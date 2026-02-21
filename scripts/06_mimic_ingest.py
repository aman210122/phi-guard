#!/usr/bin/env python3
"""
06_mimic_ingest.py — MIMIC-IV-Note Ingestion + PHI Re-injection Pipeline

PURPOSE:
  When MIMIC-IV-Note access arrives, run this script to create a genuinely
  challenging 4-tier dataset. The key innovation is PHI RE-INJECTION:
  
  MIMIC notes are deidentified (PHI replaced with ___). We re-inject
  synthetic PHI from Synthea patients to create T3 samples that are
  linguistically IDENTICAL to T2 samples except for the presence of
  identifiers. This forces the classifier to learn actual PHI detection
  rather than surface-level document-type heuristics.

TIERS CONSTRUCTED:
  T3 (Restricted): MIMIC discharge summaries WITH re-injected synthetic PHI
  T2 (Limited):    SAME MIMIC notes with ___ markers intact (deidentified)
  T0 (Public):     MedQA questions + MTSamples (no patient context)
  T1 (Internal):   Operational templates (scheduling, admin)

WHY THIS IS HARD:
  T3 and T2 come from the EXACT SAME source documents. The only difference
  is whether ___ has been replaced with "John Smith" / "MRN 12345678" etc.
  A classifier must learn to detect PHI entities, not document structure.

USAGE:
  python scripts/06_mimic_ingest.py \
    --mimic-notes /path/to/discharge.csv.gz \
    --synthea-fhir /path/to/synthea/output/fhir/ \
    --output data/mimic_dataset.json
"""

import argparse
import csv
import gzip
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta

import yaml

ROOT = Path(__file__).resolve().parent.parent
with open(ROOT / "configs/config.yaml") as f:
    cfg = yaml.safe_load(f)

SEED = cfg["dataset"]["random_seed"]
random.seed(SEED)


# ═══════════════════════════════════════════════════════════════════════════
# SYNTHEA PATIENT POOL (for PHI re-injection)
# ═══════════════════════════════════════════════════════════════════════════

# Common US names for fallback if no Synthea FHIR available
FIRST_NAMES_M = ["James", "Robert", "Michael", "David", "Richard", "Joseph",
                 "Thomas", "Charles", "Daniel", "Matthew", "Anthony", "Mark",
                 "Steven", "Paul", "Andrew", "Joshua", "Kenneth", "Kevin"]
FIRST_NAMES_F = ["Mary", "Patricia", "Jennifer", "Linda", "Barbara", "Elizabeth",
                 "Susan", "Jessica", "Sarah", "Karen", "Lisa", "Nancy",
                 "Margaret", "Sandra", "Ashley", "Dorothy", "Kimberly", "Emily"]
LAST_NAMES = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia",
              "Miller", "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez",
              "Gonzalez", "Wilson", "Anderson", "Thomas", "Taylor", "Moore",
              "Jackson", "Martin", "Lee", "Perez", "Thompson", "White"]


def load_synthea_patients(fhir_dir: Optional[Path]) -> List[Dict]:
    """Load patient demographics from Synthea FHIR bundles."""
    patients = []

    if fhir_dir and fhir_dir.exists():
        for f in sorted(fhir_dir.glob("*.json"))[:500]:
            try:
                with open(f) as fh:
                    bundle = json.load(fh)
                for entry in bundle.get("entry", []):
                    res = entry.get("resource", {})
                    if res.get("resourceType") == "Patient":
                        name = res.get("name", [{}])[0]
                        given = " ".join(name.get("given", ["Unknown"]))
                        family = name.get("family", "Unknown")
                        dob = res.get("birthDate", "1970-01-01")
                        gender = res.get("gender", "unknown")
                        addr = res.get("address", [{}])[0] if res.get("address") else {}
                        patients.append({
                            "first": given, "last": family,
                            "dob": dob, "gender": gender,
                            "mrn": f"{random.randint(1000000, 9999999)}",
                            "ssn_last4": f"{random.randint(1000, 9999)}",
                            "phone": f"({random.randint(200,999)}) {random.randint(200,999)}-{random.randint(1000,9999)}",
                            "address": f"{random.randint(1,9999)} {random.choice(['Main','Oak','Elm','Park','Cedar'])} St",
                            "city": addr.get("city", "Boston"),
                            "state": addr.get("state", "MA"),
                            "zip": addr.get("postalCode", "02101"),
                        })
            except Exception:
                continue

    # Fallback: generate synthetic patients if not enough from FHIR
    while len(patients) < 200:
        gender = random.choice(["male", "female"])
        first = random.choice(FIRST_NAMES_M if gender == "male" else FIRST_NAMES_F)
        last = random.choice(LAST_NAMES)
        year = random.randint(1940, 2005)
        month = random.randint(1, 12)
        day = random.randint(1, 28)
        patients.append({
            "first": first, "last": last,
            "dob": f"{year}-{month:02d}-{day:02d}",
            "gender": gender,
            "mrn": f"{random.randint(1000000, 9999999)}",
            "ssn_last4": f"{random.randint(1000, 9999)}",
            "phone": f"({random.randint(200,999)}) {random.randint(200,999)}-{random.randint(1000,9999)}",
            "address": f"{random.randint(1,9999)} {random.choice(['Main','Oak','Elm','Park','Cedar'])} St",
            "city": random.choice(["Boston", "Worcester", "Springfield", "Cambridge"]),
            "state": "MA",
            "zip": f"0{random.randint(1000,2999)}",
        })

    return patients


# ═══════════════════════════════════════════════════════════════════════════
# PHI RE-INJECTION ENGINE
# ═══════════════════════════════════════════════════════════════════════════

def reinject_phi(text: str, patient: Dict) -> Tuple[str, List[Dict]]:
    """
    Replace ___ markers in MIMIC deidentified text with synthetic PHI.
    Returns (reinjected_text, list_of_injected_entities).
    
    MIMIC uses '___' (three underscores) for ALL PHI types.
    We use context clues to determine what type of PHI each ___ represents
    and inject appropriate synthetic values.
    """
    entities = []
    result = text
    
    # Track positions for entity annotation
    phi_count = text.count("___")
    if phi_count == 0:
        return text, []
    
    # Context-based replacement patterns
    # Each pattern: (regex_around_blank, replacement_function, entity_type)
    patterns = [
        # Names after titles
        (r'((?:Mr|Mrs|Ms|Miss|Dr|Prof)\.\s+)___', 
         lambda p: f"{p['first']} {p['last']}", "NAME"),
        # Names after "Patient:" or "Name:"
        (r'((?:Patient|Name|Pt)\s*:\s*)___',
         lambda p: f"{p['last']}, {p['first']}", "NAME"),
        # Dates (after "Date:", "DOB:", "Admitted:", "Discharged:")
        (r'((?:Date|DOB|Admitted|Discharged|date of birth)\s*:\s*)___',
         lambda p: _random_date(), "DATE"),
        # MRN
        (r'((?:MRN|Medical Record|Record Number|Acct)\s*[:#]?\s*)___',
         lambda p: p['mrn'], "MRN"),
        # Phone
        (r'((?:Phone|Tel|Contact|Call)\s*[:#]?\s*)___',
         lambda p: p['phone'], "PHONE"),
        # Address patterns
        (r'((?:Address|Lives at|resides at|home)\s*[:#]?\s*)___',
         lambda p: f"{p['address']}, {p['city']}, {p['state']} {p['zip']}", "ADDRESS"),
        # Age patterns
        (r'((?:\b\d+\s*(?:year|yr|y/?o)\b.*?))___',
         lambda p: str(random.randint(18, 95)), "AGE"),
    ]
    
    # First pass: context-aware replacement
    for pattern, replacer, etype in patterns:
        def make_replacer(r, e, pat):
            def repl(match):
                val = r(patient)
                entities.append({"type": e, "value": val, "context": match.group(0)[:50]})
                return match.group(1) + val
            return repl
        result = re.sub(pattern, make_replacer(replacer, etype, pattern), result, 
                        count=3, flags=re.IGNORECASE)
    
    # Second pass: remaining ___ get contextual replacements
    remaining = result.count("___")
    if remaining > 0:
        # Cycle through common PHI types for remaining blanks
        replacements = [
            patient["first"] + " " + patient["last"],  # NAME
            _random_date(),                              # DATE
            patient["mrn"],                              # MRN  
            f"Dr. {random.choice(LAST_NAMES)}",         # PROVIDER
            _random_date(),                              # DATE
            patient["phone"],                            # PHONE
            f"{patient['city']}, {patient['state']}",   # LOCATION
            _random_date(),                              # DATE
        ]
        
        idx = 0
        while "___" in result and idx < len(replacements) * 3:
            replacement = replacements[idx % len(replacements)]
            etype = ["NAME", "DATE", "MRN", "PROVIDER", "DATE", 
                     "PHONE", "LOCATION", "DATE"][idx % 8]
            entities.append({"type": etype, "value": replacement})
            result = result.replace("___", replacement, 1)
            idx += 1
        
        # Any still remaining get generic replacement
        while "___" in result:
            val = random.choice([patient["first"], _random_date(), 
                                f"Dr. {random.choice(LAST_NAMES)}"])
            result = result.replace("___", val, 1)
    
    return result, entities


def _random_date() -> str:
    """Generate a random clinical date."""
    base = datetime(2020, 1, 1)
    offset = timedelta(days=random.randint(0, 1500))
    d = base + offset
    formats = ["%Y-%m-%d", "%m/%d/%Y", "%B %d, %Y", "%m/%d/%y"]
    return d.strftime(random.choice(formats))


# ═══════════════════════════════════════════════════════════════════════════
# MIMIC-IV-NOTE LOADER
# ═══════════════════════════════════════════════════════════════════════════

def load_mimic_notes(mimic_path: Path, max_notes: int = 3000) -> List[Dict]:
    """
    Load discharge summaries from MIMIC-IV-Note discharge.csv.gz
    
    Schema: note_id, subject_id, hadm_id, note_type, note_seq, 
            charttime, storetime, text
    """
    notes = []
    
    open_fn = gzip.open if str(mimic_path).endswith('.gz') else open
    mode = 'rt' if str(mimic_path).endswith('.gz') else 'r'
    
    print(f"  Loading MIMIC notes from {mimic_path}...")
    
    with open_fn(mimic_path, mode, encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= max_notes * 3:  # Read extra, filter later
                break
            
            text = row.get('text', '')
            if len(text) < 200:  # Skip very short notes
                continue
            if text.count('___') < 2:  # Need PHI markers to reinject
                continue
                
            notes.append({
                "note_id": row.get('note_id', str(i)),
                "subject_id": row.get('subject_id', ''),
                "hadm_id": row.get('hadm_id', ''),
                "text": text,
                "phi_marker_count": text.count('___'),
            })
    
    # Sample if we have more than needed
    if len(notes) > max_notes:
        # Stratified: prefer notes with more PHI markers (more interesting)
        notes.sort(key=lambda n: n['phi_marker_count'], reverse=True)
        # Take top 60% by PHI density + random 40%
        top_n = int(max_notes * 0.6)
        rest_n = max_notes - top_n
        selected = notes[:top_n]
        remaining = notes[top_n:]
        random.shuffle(remaining)
        selected.extend(remaining[:rest_n])
        random.shuffle(selected)
        notes = selected
    
    print(f"  Loaded {len(notes)} discharge summaries "
          f"(avg {sum(n['phi_marker_count'] for n in notes)/len(notes):.1f} PHI markers/note)")
    
    return notes


# ═══════════════════════════════════════════════════════════════════════════
# T0/T1 DATA (same sources as v3, loaded here for completeness)
# ═══════════════════════════════════════════════════════════════════════════

def load_t0_public(max_samples: int = 800) -> List[Dict]:
    """Load T0 from MedQA + MTSamples."""
    samples = []
    
    # Try MedQA from HuggingFace datasets
    try:
        from datasets import load_dataset
        ds = load_dataset("openlifescienceai/medqa", split="test")
        for item in ds:
            if len(samples) >= max_samples // 2:
                break
            q = item.get("question", "")
            options = item.get("options", {})
            if isinstance(options, dict):
                opts = " ".join(f"({k}) {v}" for k, v in options.items())
            else:
                opts = str(options)
            text = f"{q} {opts}".strip()
            if len(text) > 50:
                samples.append({"text": text, "tier": 0, "source": "medqa"})
    except Exception as e:
        print(f"  Warning: MedQA load failed ({e}), using templates")
    
    # Try MTSamples
    try:
        from datasets import load_dataset
        ds = load_dataset("harishnair04/mtsamples", split="train")
        for item in ds:
            if len(samples) >= max_samples:
                break
            text = item.get("transcription", item.get("description", ""))
            if text and len(text) > 100:
                # MTSamples are deidentified medical transcriptions
                samples.append({"text": text[:500], "tier": 0, "source": "mtsamples"})
    except Exception as e:
        print(f"  Warning: MTSamples load failed ({e})")
    
    # Template fallback
    t0_templates = [
        "What are the common side effects of metformin in type 2 diabetes management?",
        "Explain the pathophysiology of congestive heart failure and its staging criteria.",
        "What is the recommended screening schedule for colorectal cancer?",
        "Describe the difference between Type 1 and Type 2 diabetes mellitus.",
        "What are the clinical guidelines for hypertension management in adults?",
        "How does chronic kidney disease progress through its five stages?",
        "What laboratory values indicate diabetic ketoacidosis?",
        "Explain the mechanism of action of ACE inhibitors in heart failure.",
    ]
    while len(samples) < max_samples:
        t = random.choice(t0_templates)
        samples.append({"text": t, "tier": 0, "source": "template"})
    
    return samples[:max_samples]


def generate_t1_internal(max_samples: int = 600) -> List[Dict]:
    """Generate T1 operational/internal queries."""
    templates = [
        "Schedule a follow-up appointment for patient in cardiology clinic next week.",
        "Request lab panel CBC and BMP for upcoming annual wellness visit.",
        "Submit prior authorization for MRI lumbar spine to insurance.",
        "Notify nursing station about bed assignment change for incoming transfer.",
        "Update the department staffing schedule for next month's ICU rotation.",
        "Generate monthly report on average length of stay for cardiac patients.",
        "Coordinate ambulance transport for patient discharge to skilled nursing facility.",
        "Process medication reconciliation checklist for new admission protocol.",
        "Send referral request to endocrinology for diabetes management consultation.",
        "Update infection control dashboard with this week's hand hygiene compliance data.",
        "Submit quality measure data for CMS reporting on readmission rates.",
        "Request maintenance for EHR system downtime scheduled for Saturday.",
    ]
    
    samples = []
    for _ in range(max_samples):
        text = random.choice(templates)
        # Add light variation
        if random.random() < 0.3:
            text = text.lower()
        if random.random() < 0.2:
            text = "URGENT: " + text
        samples.append({"text": text, "tier": 1, "source": "template"})
    
    return samples


# ═══════════════════════════════════════════════════════════════════════════
# DATASET CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════

def build_dataset(mimic_notes: List[Dict], patients: List[Dict],
                  t0_samples: List[Dict], t1_samples: List[Dict],
                  t3_count: int = 1200, t2_count: int = 800) -> List[Dict]:
    """
    Build 4-tier dataset with MIMIC-sourced T3/T2.
    
    KEY DESIGN: T3 and T2 come from the SAME source notes.
    - T3: MIMIC note with synthetic PHI re-injected
    - T2: SAME MIMIC note with ___ markers (deidentified)
    
    This creates MATCHED PAIRS that test whether the classifier can
    detect the PRESENCE of PHI rather than document structure.
    """
    dataset = []
    
    # ── T3: MIMIC notes with re-injected PHI ─────────────────────────────
    print(f"\n  Building T3 (PHI-present): {t3_count} samples from MIMIC + Synthea PHI")
    t3_notes = mimic_notes[:t3_count]
    
    for i, note in enumerate(t3_notes):
        patient = patients[i % len(patients)]
        reinjected_text, entities = reinject_phi(note["text"], patient)
        
        # Truncate to reasonable length for classification
        if len(reinjected_text) > 1500:
            # Keep first and last portions (header + assessment/plan)
            reinjected_text = reinjected_text[:800] + "\n...\n" + reinjected_text[-500:]
        
        dataset.append({
            "text": reinjected_text,
            "tier": 3,
            "source": "mimic_reinjected",
            "note_id": note["note_id"],
            "phi_entities": entities,
            "phi_count": len(entities),
            "matched_t2_idx": len(t0_samples) + len(t1_samples) + t3_count + i,  # Index of T2 pair
        })
    
    # ── T2: SAME MIMIC notes with ___ markers (deidentified) ─────────────
    print(f"  Building T2 (deidentified): {t2_count} samples from same MIMIC notes")
    t2_notes = mimic_notes[:t2_count]
    
    for note in t2_notes:
        text = note["text"]
        if len(text) > 1500:
            text = text[:800] + "\n...\n" + text[-500:]
        
        dataset.append({
            "text": text,
            "tier": 2,
            "source": "mimic_deidentified",
            "note_id": note["note_id"],
            "phi_marker_count": note["phi_marker_count"],
        })
    
    # ── T0 and T1 ────────────────────────────────────────────────────────
    dataset.extend(t0_samples)
    dataset.extend(t1_samples)
    
    # Shuffle
    random.shuffle(dataset)
    
    return dataset


def split_dataset(data: List[Dict], train: float = 0.5, val: float = 0.1,
                  test: float = 0.4) -> Tuple[List, List, List]:
    """Split ensuring no matched T3/T2 pairs leak across splits."""
    # Group by note_id to keep matched pairs together
    note_groups = {}
    ungrouped = []
    
    for item in data:
        nid = item.get("note_id")
        if nid:
            if nid not in note_groups:
                note_groups[nid] = []
            note_groups[nid].append(item)
        else:
            ungrouped.append(item)
    
    # Shuffle groups
    group_list = list(note_groups.values())
    random.shuffle(group_list)
    random.shuffle(ungrouped)
    
    # Split groups
    n_groups = len(group_list)
    train_end = int(n_groups * train)
    val_end = int(n_groups * (train + val))
    
    train_data = []
    val_data = []
    test_data = []
    
    for i, group in enumerate(group_list):
        if i < train_end:
            train_data.extend(group)
        elif i < val_end:
            val_data.extend(group)
        else:
            test_data.extend(group)
    
    # Split ungrouped
    n_ug = len(ungrouped)
    ug_train_end = int(n_ug * train)
    ug_val_end = int(n_ug * (train + val))
    train_data.extend(ungrouped[:ug_train_end])
    val_data.extend(ungrouped[ug_train_end:ug_val_end])
    test_data.extend(ungrouped[ug_val_end:])
    
    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)
    
    return train_data, val_data, test_data


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="MIMIC-IV Note Ingestion Pipeline")
    parser.add_argument("--mimic-notes", type=str, required=True,
                        help="Path to discharge.csv.gz from MIMIC-IV-Note")
    parser.add_argument("--synthea-fhir", type=str, default=None,
                        help="Path to Synthea FHIR output directory")
    parser.add_argument("--output", type=str, default="data/mimic_dataset.json",
                        help="Output dataset path")
    parser.add_argument("--t3-count", type=int, default=1200)
    parser.add_argument("--t2-count", type=int, default=800)
    parser.add_argument("--t0-count", type=int, default=800)
    parser.add_argument("--t1-count", type=int, default=600)
    args = parser.parse_args()

    print("=" * 60)
    print("MIMIC-IV Note Ingestion + PHI Re-injection Pipeline")
    print("=" * 60)

    # ── Load Synthea patients ─────────────────────────────────────────────
    fhir_dir = Path(args.synthea_fhir) if args.synthea_fhir else None
    print("\nStage 1: Loading Synthea patient pool...")
    patients = load_synthea_patients(fhir_dir)
    print(f"  {len(patients)} synthetic patient identities loaded")

    # ── Load MIMIC notes ──────────────────────────────────────────────────
    print("\nStage 2: Loading MIMIC-IV discharge summaries...")
    mimic_path = Path(args.mimic_notes)
    if not mimic_path.exists():
        print(f"ERROR: {mimic_path} not found!")
        print("Download discharge.csv.gz from physionet.org/content/mimic-iv-note/2.2/")
        sys.exit(1)
    
    max_notes = max(args.t3_count, args.t2_count)
    mimic_notes = load_mimic_notes(mimic_path, max_notes=max_notes)

    # ── Load T0/T1 ────────────────────────────────────────────────────────
    print("\nStage 3: Loading T0 (public) and T1 (internal) data...")
    t0_samples = load_t0_public(args.t0_count)
    t1_samples = generate_t1_internal(args.t1_count)
    print(f"  T0: {len(t0_samples)}, T1: {len(t1_samples)}")

    # ── Build dataset ─────────────────────────────────────────────────────
    print("\nStage 4: Building 4-tier dataset with PHI re-injection...")
    dataset = build_dataset(mimic_notes, patients, t0_samples, t1_samples,
                            args.t3_count, args.t2_count)

    # ── Statistics ────────────────────────────────────────────────────────
    tier_counts = {}
    for item in dataset:
        t = item["tier"]
        tier_counts[t] = tier_counts.get(t, 0) + 1
    
    print(f"\n  Dataset: {len(dataset)} total samples")
    for t in sorted(tier_counts):
        labels = {0: "T0_Public", 1: "T1_Internal", 2: "T2_Limited", 3: "T3_Restricted"}
        print(f"    {labels[t]}: {tier_counts[t]}")
    
    # Count matched pairs
    matched = sum(1 for item in dataset if item.get("matched_t2_idx"))
    print(f"  Matched T3/T2 pairs: {matched}")

    # ── Split ─────────────────────────────────────────────────────────────
    print("\nStage 5: Train/Val/Test split (no pair leakage)...")
    train, val, test = split_dataset(dataset)
    print(f"  Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

    # ── Save ──────────────────────────────────────────────────────────────
    out_dir = ROOT / Path(args.output).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    
    out_path = ROOT / args.output
    with open(out_path, "w") as f:
        json.dump(dataset, f, indent=2, default=str)
    
    # Save splits
    for name, split in [("train", train), ("val", val), ("test", test)]:
        p = out_dir / f"{name}.json"
        with open(p, "w") as f:
            json.dump(split, f, indent=2, default=str)
    
    print(f"\nSaved to {out_path}")
    print(f"Splits: {out_dir}/{{train,val,test}}.json")
    
    # ── Sample outputs ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SAMPLE OUTPUTS")
    print("=" * 60)
    
    for tier in [3, 2, 0]:
        sample = next((s for s in dataset if s["tier"] == tier), None)
        if sample:
            labels = {0: "T0_Public", 1: "T1_Internal", 2: "T2_Limited", 3: "T3_Restricted"}
            print(f"\n--- {labels[tier]} ({sample.get('source', 'unknown')}) ---")
            print(sample["text"][:300] + "...")


if __name__ == "__main__":
    main()
