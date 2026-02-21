#!/usr/bin/env python3
"""
06_mimic_ingest_v2.py — MIMIC-IV Ingestion with PARTIAL PHI Re-injection

KEY CHANGE FROM V1:
  V1 replaced ALL ___ markers → classifier learned "has ___ = T2, no ___ = T3"
  V2 replaces only 30-60% of markers randomly → BOTH T3 and T2 contain ___
  This forces the classifier to detect ACTUAL PHI entities, not ___ patterns.

ADDITIONAL IMPROVEMENTS:
  - T0 includes PubMed-style clinical abstracts that look like notes
  - T3-hard samples: PHI embedded naturally in narrative
  - Configurable difficulty via --replace-pct flag
  - Text augmentation: abbreviations, typos, case variation

USAGE:
  python scripts/06_mimic_ingest_v2.py \
    --mimic-notes /path/to/discharge.csv.gz \
    --synthea-fhir /path/to/synthea/output/fhir/ \
    --replace-pct 0.45 \
    --output data/mimic_dataset.json
"""

import argparse
import csv
import gzip
import json
import os
import random
import re
import string
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
# SYNTHEA PATIENT POOL
# ═══════════════════════════════════════════════════════════════════════════

FIRST_NAMES_M = ["James", "Robert", "Michael", "David", "Richard", "Joseph",
                 "Thomas", "Charles", "Daniel", "Matthew", "Anthony", "Mark",
                 "Steven", "Paul", "Andrew", "Joshua", "Kenneth", "Kevin",
                 "Brian", "George", "Timothy", "Ronald", "Edward", "Jason"]
FIRST_NAMES_F = ["Mary", "Patricia", "Jennifer", "Linda", "Barbara", "Elizabeth",
                 "Susan", "Jessica", "Sarah", "Karen", "Lisa", "Nancy",
                 "Margaret", "Sandra", "Ashley", "Dorothy", "Kimberly", "Emily",
                 "Donna", "Michelle", "Carol", "Amanda", "Melissa", "Deborah"]
LAST_NAMES = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia",
              "Miller", "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez",
              "Gonzalez", "Wilson", "Anderson", "Thomas", "Taylor", "Moore",
              "Jackson", "Martin", "Lee", "Perez", "Thompson", "White",
              "Harris", "Clark", "Lewis", "Robinson", "Walker", "Young"]


def load_synthea_patients(fhir_dir: Optional[Path]) -> List[Dict]:
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
                        addr = res.get("address", [{}])[0] if res.get("address") else {}
                        patients.append({
                            "first": given, "last": family,
                            "dob": res.get("birthDate", "1970-01-01"),
                            "gender": res.get("gender", "unknown"),
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

    while len(patients) < 200:
        gender = random.choice(["male", "female"])
        first = random.choice(FIRST_NAMES_M if gender == "male" else FIRST_NAMES_F)
        last = random.choice(LAST_NAMES)
        patients.append({
            "first": first, "last": last,
            "dob": f"{random.randint(1940,2005)}-{random.randint(1,12):02d}-{random.randint(1,28):02d}",
            "gender": gender,
            "mrn": f"{random.randint(1000000, 9999999)}",
            "ssn_last4": f"{random.randint(1000, 9999)}",
            "phone": f"({random.randint(200,999)}) {random.randint(200,999)}-{random.randint(1000,9999)}",
            "address": f"{random.randint(1,9999)} {random.choice(['Main','Oak','Elm','Park','Cedar'])} St",
            "city": random.choice(["Boston", "Worcester", "Springfield", "Cambridge"]),
            "state": "MA", "zip": f"0{random.randint(1000,2999)}",
        })
    return patients


# ═══════════════════════════════════════════════════════════════════════════
# PARTIAL PHI RE-INJECTION (THE KEY FIX)
# ═══════════════════════════════════════════════════════════════════════════

def _random_date() -> str:
    base = datetime(2020, 1, 1) + timedelta(days=random.randint(0, 1500))
    return base.strftime(random.choice(["%Y-%m-%d", "%m/%d/%Y", "%B %d, %Y", "%m/%d/%y"]))

def _get_replacement(patient: Dict, marker_idx: int) -> Tuple[str, str]:
    """Get a PHI replacement value and its entity type based on position."""
    cycle = [
        (f"{patient['first']} {patient['last']}", "NAME"),
        (_random_date(), "DATE"),
        (patient['mrn'], "MRN"),
        (f"Dr. {random.choice(LAST_NAMES)}", "PROVIDER"),
        (_random_date(), "DATE"),
        (patient['phone'], "PHONE"),
        (f"{patient['city']}, {patient['state']}", "LOCATION"),
        (_random_date(), "DATE"),
    ]
    return cycle[marker_idx % len(cycle)]


def partial_reinject_phi(text: str, patient: Dict, replace_pct: float = 0.45) -> Tuple[str, List[Dict], int]:
    """
    Replace only a FRACTION of ___ markers with synthetic PHI.
    
    This is the critical fix: if we replace all markers, T3 has zero "___" 
    and classifier trivially separates T3/T2. By replacing only 30-60%,
    BOTH tiers contain ___ markers, forcing the classifier to detect
    actual PHI tokens instead of marker patterns.
    
    Returns: (text, entities, n_replaced)
    """
    # Find all ___ positions
    markers = [(m.start(), m.end()) for m in re.finditer(r'___', text)]
    if not markers:
        return text, [], 0
    
    n_total = len(markers)
    n_replace = max(1, int(n_total * replace_pct))
    
    # Randomly select which markers to replace
    chosen_indices = sorted(random.sample(range(n_total), min(n_replace, n_total)))
    
    entities = []
    result = text
    offset = 0  # Track position shift from replacements
    
    for idx, marker_num in enumerate(chosen_indices):
        start, end = markers[marker_num]
        start += offset
        end += offset
        
        replacement, etype = _get_replacement(patient, idx)
        
        result = result[:start] + replacement + result[end:]
        offset += len(replacement) - (end - start)
        
        entities.append({"type": etype, "value": replacement, "position": start})
    
    return result, entities, len(chosen_indices)


# ═══════════════════════════════════════════════════════════════════════════
# MIMIC-IV-NOTE LOADER
# ═══════════════════════════════════════════════════════════════════════════

def load_mimic_notes(mimic_path: Path, max_notes: int = 3000) -> List[Dict]:
    notes = []
    open_fn = gzip.open if str(mimic_path).endswith('.gz') else open
    mode = 'rt' if str(mimic_path).endswith('.gz') else 'r'
    
    print(f"  Loading MIMIC notes from {mimic_path}...")
    with open_fn(mimic_path, mode, encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= max_notes * 3:
                break
            text = row.get('text', '')
            if len(text) < 200:
                continue
            if text.count('___') < 3:  # Need enough markers for partial replacement
                continue
            notes.append({
                "note_id": row.get('note_id', str(i)),
                "subject_id": row.get('subject_id', ''),
                "hadm_id": row.get('hadm_id', ''),
                "text": text,
                "phi_marker_count": text.count('___'),
            })
    
    if len(notes) > max_notes:
        notes.sort(key=lambda n: n['phi_marker_count'], reverse=True)
        top_n = int(max_notes * 0.6)
        rest_n = max_notes - top_n
        selected = notes[:top_n]
        remaining = notes[top_n:]
        random.shuffle(remaining)
        selected.extend(remaining[:rest_n])
        random.shuffle(selected)
        notes = selected
    
    print(f"  Loaded {len(notes)} discharge summaries "
          f"(avg {sum(n['phi_marker_count'] for n in notes)/max(len(notes),1):.1f} PHI markers/note)")
    return notes


# ═══════════════════════════════════════════════════════════════════════════
# HARD T0: Clinical text that LOOKS like notes but has NO PHI
# ═══════════════════════════════════════════════════════════════════════════

CLINICAL_VIGNETTES = [
    # These mimic discharge note style but contain zero PHI
    "ASSESSMENT AND PLAN:\nThe patient presents with acute exacerbation of chronic obstructive pulmonary disease. Chest X-ray shows hyperinflation without consolidation. Started on nebulized albuterol and ipratropium every 4 hours. IV methylprednisolone 125mg daily. Supplemental oxygen via nasal cannula at 2L/min to maintain SpO2 >92%. Continue home medications including tiotropium and fluticasone/salmeterol. Pulmonology consulted for possible BiPAP if clinical deterioration.",
    
    "HOSPITAL COURSE:\nPatient admitted with new-onset atrial fibrillation with rapid ventricular response, heart rate 142bpm. Hemodynamically stable. Troponin negative x2. TSH within normal limits. Echocardiogram showed preserved ejection fraction at 55-60%, no valvular abnormalities. Started on diltiazem drip, successfully rate controlled to HR 70-80s. Transitioned to oral metoprolol succinate 50mg daily. CHADS2-VASc score calculated, initiated apixaban 5mg BID for stroke prophylaxis.",
    
    "DISCHARGE SUMMARY:\nAdmitted for elective left total knee arthroplasty. Surgery performed without complications under spinal anesthesia. Estimated blood loss 200mL. Post-operatively, patient progressed well with physical therapy. Achieved 90 degrees flexion by post-op day 2. Pain managed with multimodal approach including scheduled acetaminophen, celecoxib, and PRN oxycodone. DVT prophylaxis with enoxaparin 40mg SQ daily. Discharged to skilled nursing facility for continued rehabilitation.",
    
    "CLINICAL NOTE:\nFollow-up visit for type 2 diabetes mellitus. HbA1c improved from 9.2% to 7.8% over past 3 months on current regimen of metformin 1000mg BID and empagliflozin 10mg daily. Renal function stable with eGFR 62. Urine albumin-to-creatinine ratio mildly elevated at 45 mg/g. Blood pressure at goal 128/76 on lisinopril 20mg. Continued diabetic retinopathy screening recommended. Foot exam unremarkable. Reinforced dietary counseling and exercise goals.",
    
    "PROGRESS NOTE:\nCritical care day 3. Patient remains intubated on mechanical ventilation, FiO2 40%, PEEP 8. Sedation with propofol infusion. Hemodynamics stable on norepinephrine 0.05 mcg/kg/min, being weaned. Lactate trending down from 4.2 to 1.8. Blood cultures from admission growing gram-negative rods, speciation pending. Broad spectrum antibiotics continued with meropenem and vancomycin. Nutrition via nasogastric tube, tolerating feeds at goal rate. Renal function improving, creatinine 1.4 from peak 2.1.",
    
    "CONSULTATION NOTE:\nRequested for evaluation of acute kidney injury in setting of sepsis. Baseline creatinine 0.9, now 3.2. Urine output has been 15mL/hr over past 6 hours. Urine sodium 8 mEq/L consistent with prerenal etiology. No muddy brown casts on urinalysis. Renal ultrasound shows normal sized kidneys without hydronephrosis. Recommend aggressive fluid resuscitation with lactated Ringer's, holding nephrotoxic agents including ACE inhibitor and NSAIDs. Will follow daily chemistry. Dialysis not indicated at this time.",
    
    "OPERATIVE NOTE:\nProcedure: Laparoscopic cholecystectomy. Indication: Symptomatic cholelithiasis with recurrent biliary colic. Findings: Chronically inflamed gallbladder with multiple pigmented stones. No evidence of choledocholithiasis. Critical view of safety achieved. Cystic duct and cystic artery identified, clipped, and divided. Gallbladder dissected from liver bed using electrocautery. Specimen removed via umbilical port in endocatch bag. Hemostasis confirmed. All ports removed under direct visualization.",
    
    "EMERGENCY DEPARTMENT NOTE:\nChief complaint: Chest pain. History: Sudden onset substernal chest pressure radiating to left arm, associated with diaphoresis and nausea. Onset 45 minutes ago. Risk factors include hypertension, hyperlipidemia, and 30 pack-year smoking history. ECG shows ST elevations in leads II, III, aVF consistent with inferior STEMI. Aspirin 325mg and heparin bolus administered. Cardiology emergently consulted for primary percutaneous coronary intervention. Patient hemodynamically stable, pain 8/10.",
    
    "PSYCHIATRIC EVALUATION:\nPresenting concern: Worsening depression and passive suicidal ideation without plan or intent. PHQ-9 score 22 indicating severe depression. Currently on sertraline 100mg which was titrated 6 weeks ago. Denies active suicidal or homicidal ideation. No history of suicide attempts. Sleep disrupted with early morning awakening. Appetite decreased with 8-pound weight loss over 2 months. Assessment: Major depressive disorder, recurrent, severe, without psychotic features. Plan: Increase sertraline to 150mg, add trazodone 50mg QHS for insomnia, safety plan reviewed.",
    
    "RADIOLOGY REPORT:\nCT Chest with contrast. Clinical indication: Shortness of breath, rule out pulmonary embolism. Findings: No pulmonary embolism identified. Bilateral small pleural effusions, left greater than right. Dependent atelectasis at bilateral lung bases. Mild cardiomegaly. No mediastinal lymphadenopathy. Aorta normal in caliber. Incidental 4mm ground glass nodule right lower lobe. Impression: 1. No pulmonary embolism. 2. Bilateral pleural effusions. 3. Incidental pulmonary nodule, recommend follow-up CT in 12 months per Fleischner criteria.",
]


def load_t0_public(max_samples: int = 800) -> List[Dict]:
    """Load T0 from MedQA + MTSamples + clinical vignettes."""
    samples = []
    
    # Clinical vignettes that look like notes (HARD for classifier)
    for v in CLINICAL_VIGNETTES:
        for _ in range(max_samples // (len(CLINICAL_VIGNETTES) * 3)):
            # Add variation
            text = v
            if random.random() < 0.3:
                text = text.replace("Patient", random.choice(["Pt", "patient", "The patient"]))
            if random.random() < 0.2:
                text = text.lower()
            samples.append({"text": text, "tier": 0, "source": "clinical_vignette"})
    
    # MedQA
    try:
        from datasets import load_dataset
        ds = load_dataset("openlifescienceai/medqa", split="test")
        for item in ds:
            if len(samples) >= max_samples * 0.7:
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
        print(f"  Warning: MedQA load failed ({e})")
    
    # MTSamples
    try:
        from datasets import load_dataset
        ds = load_dataset("harishnair04/mtsamples", split="train")
        for item in ds:
            if len(samples) >= max_samples:
                break
            text = item.get("transcription", item.get("description", ""))
            if text and len(text) > 100:
                samples.append({"text": text[:500], "tier": 0, "source": "mtsamples"})
    except Exception as e:
        print(f"  Warning: MTSamples load failed ({e})")
    
    # Template fallback
    t0_templates = [
        "What are the common side effects of metformin in type 2 diabetes management?",
        "Explain the pathophysiology of congestive heart failure and its staging criteria.",
        "What is the recommended screening schedule for colorectal cancer?",
        "How does chronic kidney disease progress through its five stages?",
        "What laboratory values indicate diabetic ketoacidosis?",
    ]
    while len(samples) < max_samples:
        samples.append({"text": random.choice(t0_templates), "tier": 0, "source": "template"})
    
    random.shuffle(samples)
    return samples[:max_samples]


def generate_t1_internal(max_samples: int = 600) -> List[Dict]:
    templates = [
        "Schedule a follow-up appointment for patient in cardiology clinic next week.",
        "Request lab panel CBC and BMP for upcoming annual wellness visit.",
        "Submit prior authorization for MRI lumbar spine to insurance.",
        "Notify nursing station about bed assignment change for incoming transfer.",
        "Update the department staffing schedule for next month's ICU rotation.",
        "Generate monthly report on average length of stay for cardiac patients.",
        "Process medication reconciliation checklist for new admission protocol.",
        "Send referral request to endocrinology for diabetes management consultation.",
        "Submit quality measure data for CMS reporting on readmission rates.",
        "Request maintenance for EHR system downtime scheduled for Saturday.",
    ]
    samples = []
    for _ in range(max_samples):
        text = random.choice(templates)
        if random.random() < 0.3:
            text = text.lower()
        if random.random() < 0.2:
            text = "URGENT: " + text
        samples.append({"text": text, "tier": 1, "source": "template"})
    return samples


# ═══════════════════════════════════════════════════════════════════════════
# DATASET CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════

def truncate_note(text: str, max_len: int = 1500) -> str:
    if len(text) > max_len:
        return text[:800] + "\n...\n" + text[-500:]
    return text


def strip_header(text: str) -> str:
    """
    Remove the structured header from MIMIC discharge notes.
    
    The header contains fields like Name, Unit No, DOB, Admission Date,
    Discharge Date, Sex, Service, Attending — these are where the most
    OBVIOUS PHI lives. By stripping them, we force the classifier to
    detect PHI embedded in clinical narrative prose.
    
    We keep text starting from the first clinical section:
    - Chief Complaint
    - History of Present Illness
    - Major Surgical or Invasive Procedure
    - Past Medical History
    - Allergies (when followed by content)
    """
    # Try to find the start of clinical content
    clinical_starts = [
        r'(?i)chief\s+complaint',
        r'(?i)history\s+of\s+present\s+illness',
        r'(?i)major\s+surgical',
        r'(?i)past\s+medical\s+history',
        r'(?i)reason\s+for\s+admission',
        r'(?i)present\s+illness',
        r'(?i)hospital\s+course',
        r'(?i)brief\s+hospital\s+course',
    ]
    
    earliest_pos = len(text)  # default: keep everything
    for pattern in clinical_starts:
        match = re.search(pattern, text)
        if match and match.start() < earliest_pos:
            earliest_pos = match.start()
    
    # If we found a clinical section, strip everything before it
    if earliest_pos < len(text) and earliest_pos > 50:
        return text[earliest_pos:]
    
    # Fallback: try to skip past the first ~10 lines (header block)
    lines = text.split('\n')
    if len(lines) > 12:
        # Skip lines that look like header fields
        start_idx = 0
        for i, line in enumerate(lines[:15]):
            if re.match(r'(?i)^\s*(name|unit\s*no|admission|discharge|date\s*of\s*birth|sex|service|attending|allergies)\s*:', line.strip()):
                start_idx = i + 1
            elif line.strip() == '' and i > 5:
                start_idx = i + 1
                break
        if start_idx > 0:
            return '\n'.join(lines[start_idx:])
    
    return text


def build_dataset(mimic_notes: List[Dict], patients: List[Dict],
                  t0_samples: List[Dict], t1_samples: List[Dict],
                  replace_pct: float = 0.45,
                  t3_count: int = 1200, t2_count: int = 800) -> List[Dict]:
    """
    Build 4-tier dataset with PARTIAL PHI re-injection.
    
    T3: MIMIC note with ~45% of ___ replaced with PHI (still has ___ too!)
    T2: Same MIMIC note with ALL ___ intact
    """
    dataset = []
    
    # ── T3: MIMIC notes with PARTIAL PHI re-injection ─────────────────────
    print(f"\n  Building T3: {t3_count} samples, {replace_pct*100:.0f}% PHI replacement")
    t3_notes = mimic_notes[:t3_count]
    
    t3_phi_counts = []
    for i, note in enumerate(t3_notes):
        patient = patients[i % len(patients)]
        reinjected, entities, n_replaced = partial_reinject_phi(
            note["text"], patient, replace_pct
        )
        t3_phi_counts.append(n_replaced)
        
        processed_text = truncate_note(strip_header(reinjected))
        dataset.append({
            "text": processed_text,
            "tier": 3,
            "source": "mimic_partial_phi",
            "note_id": note["note_id"],
            "phi_entities": entities,
            "phi_count": len(entities),
            "markers_remaining": processed_text.count("___"),
            "replace_pct_actual": n_replaced / max(note["phi_marker_count"], 1),
        })
    
    avg_phi = sum(t3_phi_counts) / max(len(t3_phi_counts), 1)
    print(f"    Avg PHI entities injected: {avg_phi:.1f}")
    print(f"    Avg ___ remaining in T3: {sum(d['markers_remaining'] for d in dataset if d['tier']==3)/max(t3_count,1):.1f}")
    
    # ── T2: SAME MIMIC notes, ALL ___ intact ──────────────────────────────
    print(f"  Building T2: {t2_count} samples (deidentified, all ___ intact)")
    t2_notes = mimic_notes[:t2_count]
    
    for note in t2_notes:
        dataset.append({
            "text": truncate_note(strip_header(note["text"])),
            "tier": 2,
            "source": "mimic_deidentified",
            "note_id": note["note_id"],
            "phi_marker_count": note["phi_marker_count"],
        })
    
    # ── T0 and T1 ────────────────────────────────────────────────────────
    print(f"  Adding T0: {len(t0_samples)}, T1: {len(t1_samples)}")
    dataset.extend(t0_samples)
    dataset.extend(t1_samples)
    
    random.shuffle(dataset)
    return dataset


def split_dataset(data, train=0.5, val=0.1, test=0.4):
    """Split ensuring no matched T3/T2 pairs leak across splits."""
    note_groups = {}
    ungrouped = []
    for item in data:
        nid = item.get("note_id")
        if nid:
            note_groups.setdefault(nid, []).append(item)
        else:
            ungrouped.append(item)
    
    group_list = list(note_groups.values())
    random.shuffle(group_list)
    random.shuffle(ungrouped)
    
    n = len(group_list)
    train_end = int(n * train)
    val_end = int(n * (train + val))
    
    train_data, val_data, test_data = [], [], []
    for i, group in enumerate(group_list):
        if i < train_end: train_data.extend(group)
        elif i < val_end: val_data.extend(group)
        else: test_data.extend(group)
    
    n_ug = len(ungrouped)
    train_data.extend(ungrouped[:int(n_ug * train)])
    val_data.extend(ungrouped[int(n_ug * train):int(n_ug * (train + val))])
    test_data.extend(ungrouped[int(n_ug * (train + val)):])
    
    for d in [train_data, val_data, test_data]:
        random.shuffle(d)
    
    return train_data, val_data, test_data


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mimic-notes", type=str, required=True)
    parser.add_argument("--synthea-fhir", type=str, default=None)
    parser.add_argument("--output", type=str, default="data/mimic_dataset.json")
    parser.add_argument("--replace-pct", type=float, default=0.45,
                        help="Fraction of ___ to replace with PHI (0.3-0.6 recommended)")
    parser.add_argument("--t3-count", type=int, default=1200)
    parser.add_argument("--t2-count", type=int, default=800)
    parser.add_argument("--t0-count", type=int, default=800)
    parser.add_argument("--t1-count", type=int, default=600)
    args = parser.parse_args()

    print("=" * 60)
    print("MIMIC-IV Ingestion v2: PARTIAL PHI Re-injection")
    print(f"Replace percentage: {args.replace_pct*100:.0f}%")
    print("=" * 60)

    fhir_dir = Path(args.synthea_fhir) if args.synthea_fhir else None
    print("\nStage 1: Loading Synthea patients...")
    patients = load_synthea_patients(fhir_dir)
    print(f"  {len(patients)} patients loaded")

    print("\nStage 2: Loading MIMIC-IV discharge summaries...")
    mimic_notes = load_mimic_notes(Path(args.mimic_notes), max(args.t3_count, args.t2_count))

    print("\nStage 3: Loading T0 + T1...")
    t0 = load_t0_public(args.t0_count)
    t1 = generate_t1_internal(args.t1_count)
    
    # Count clinical vignettes in T0
    n_vignettes = sum(1 for s in t0 if s["source"] == "clinical_vignette")
    print(f"  T0: {len(t0)} ({n_vignettes} clinical vignettes that mimic note style)")
    print(f"  T1: {len(t1)}")

    print("\nStage 4: Building dataset with partial PHI re-injection...")
    dataset = build_dataset(mimic_notes, patients, t0, t1,
                            args.replace_pct, args.t3_count, args.t2_count)

    tier_counts = {}
    for item in dataset:
        tier_counts[item["tier"]] = tier_counts.get(item["tier"], 0) + 1
    
    labels = {0: "T0_Public", 1: "T1_Internal", 2: "T2_Limited", 3: "T3_Restricted"}
    print(f"\n  Total: {len(dataset)} samples")
    for t in sorted(tier_counts):
        print(f"    {labels[t]}: {tier_counts[t]}")
    
    # Key diagnostic: how many ___ markers in each tier?
    for t in [2, 3]:
        tier_samples = [d for d in dataset if d["tier"] == t]
        avg_markers = sum(d["text"].count("___") for d in tier_samples) / max(len(tier_samples), 1)
        print(f"  Avg ___ markers in {labels[t]}: {avg_markers:.1f}")

    print("\nStage 5: Splitting (no pair leakage)...")
    train, val, test = split_dataset(dataset)
    print(f"  Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

    out_dir = ROOT / Path(args.output).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(ROOT / args.output, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, default=str)
    
    for name, split in [("train", train), ("val", val), ("test", test)]:
        with open(out_dir / f"{name}.json", "w", encoding="utf-8") as f:
            json.dump(split, f, indent=2, default=str)
    
    print(f"\nSaved to {ROOT / args.output}")

    # Samples
    print("\n" + "=" * 60)
    print("SAMPLE COMPARISON (T3 vs T2 from SAME note)")
    print("=" * 60)
    
    # Find a matched pair
    for d3 in dataset:
        if d3["tier"] == 3:
            nid = d3.get("note_id")
            d2 = next((d for d in dataset if d["tier"] == 2 and d.get("note_id") == nid), None)
            if d2:
                print(f"\n--- T3 (partial PHI, note {nid}) ---")
                print(d3["text"][:400])
                print(f"\n--- T2 (deidentified, note {nid}) ---")
                print(d2["text"][:400])
                break


if __name__ == "__main__":
    main()
