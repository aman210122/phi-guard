#!/usr/bin/env python3
"""
20_hardest_dataset.py — Maximum difficulty dataset

PROBLEM:
  Script 18 with 45% injection rate yielded 99.93% T3 recall (1 miss).
  Still too easy. DistilBERT finds any real name → T3.

THREE ADDITIONAL TECHNIQUES to force misclassification:

  1. LOWER INJECTION RATE (20%): T3 notes have only 2-5 PHI entities
     buried in long clinical text. Many ___ markers remain.

  2. NAME POLLUTION IN T2: Add provider names, hospital names, and 
     generic person references to T2 notes. "Dr. Smith ordered labs",
     "per Dr. Johnson's assessment", "transferred from Mass General".
     Now T2 also has real-looking names — the classifier must learn
     PATIENT names vs PROVIDER names, which is genuinely hard.

  3. SHORTER SNIPPETS: Truncate notes to 400-600 chars. Less context
     means fewer PHI signals per sample and harder classification.

  4. CROSS-CONTAMINATION SET: 10% of T2 samples get 1 random name
     injected (simulating imperfect de-identification in real data).
     10% of T3 samples are notes with very few ___ markers where only
     1 marker gets replaced (minimal PHI signal).

EXPECTED T3 RECALL: 85-93%
"""

import argparse
import csv
import gzip
import json
import random
import re
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
SEED = 42
random.seed(SEED)

# ═══════════════════════════════════════════════════════════════
# NAME POOLS
# ═══════════════════════════════════════════════════════════════

FIRST_NAMES_M = ["James", "Robert", "Michael", "David", "Richard", "Joseph",
                 "Thomas", "Charles", "Daniel", "Matthew", "Anthony", "Mark",
                 "Steven", "Paul", "Andrew", "Joshua", "Kenneth", "Kevin",
                 "Brian", "George", "Timothy", "Ronald", "Edward", "Jason",
                 "Jeffrey", "Ryan", "Jacob", "Nicholas", "Gary", "Eric"]
FIRST_NAMES_F = ["Mary", "Patricia", "Jennifer", "Linda", "Barbara", "Elizabeth",
                 "Susan", "Jessica", "Sarah", "Karen", "Lisa", "Nancy",
                 "Margaret", "Sandra", "Ashley", "Dorothy", "Kimberly", "Emily",
                 "Donna", "Michelle", "Carol", "Amanda", "Melissa", "Deborah",
                 "Stephanie", "Rebecca", "Sharon", "Laura", "Cynthia", "Amy"]
LAST_NAMES = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia",
              "Miller", "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez",
              "Gonzalez", "Wilson", "Anderson", "Thomas", "Taylor", "Moore",
              "Jackson", "Martin", "Lee", "Perez", "Thompson", "White",
              "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson",
              "Walker", "Young", "Allen", "King", "Wright", "Scott"]

# Provider/hospital names for T2 pollution
PROVIDER_PHRASES = [
    "Dr. {last} ordered",
    "per Dr. {last}'s assessment",
    "Dr. {first} {last} evaluated the patient",
    "attending Dr. {last}",
    "discussed with Dr. {last}",
    "Dr. {last} was consulted",
    "seen by Dr. {first} {last} in consultation",
    "Dr. {last} recommended",
    "per {first} {last}, RN",
    "nurse {first} administered",
    "{first} {last}, PA-C assessed",
    "Dr. {last} performed the procedure",
    "spoke with Dr. {last} regarding plan",
    "Dr. {last} agrees with the assessment",
]

HOSPITAL_PHRASES = [
    "transferred from Massachusetts General",
    "previously seen at Beth Israel Deaconess",
    "follow up at Brigham and Women's",
    "referred from Mount Sinai",
    "outside records from Johns Hopkins",
    "prior admission to Cleveland Clinic",
    "evaluated at Memorial Sloan Kettering",
    "records from Mayo Clinic reviewed",
]


def generate_patients(n=500):
    patients = []
    for _ in range(n):
        gender = random.choice(["male", "female"])
        first = random.choice(FIRST_NAMES_M if gender == "male" else FIRST_NAMES_F)
        last = random.choice(LAST_NAMES)
        year = random.randint(1940, 2005)
        patients.append({
            "first": first, "last": last,
            "dob": f"{year}-{random.randint(1,12):02d}-{random.randint(1,28):02d}",
            "mrn": f"{random.randint(1000000, 9999999)}",
            "phone": f"({random.randint(200,999)}) {random.randint(200,999)}-{random.randint(1000,9999)}",
            "ssn_last4": f"{random.randint(1000, 9999)}",
        })
    return patients


def random_date():
    d = datetime(2018, 1, 1) + timedelta(days=random.randint(0, 2000))
    return d.strftime(random.choice(["%Y-%m-%d", "%m/%d/%Y", "%B %d, %Y", "%m/%d/%y"]))


# ═══════════════════════════════════════════════════════════════
# HEADER STRIPPING
# ═══════════════════════════════════════════════════════════════

def strip_header(text):
    markers = [
        r"Chief Complaint", r"History of Present Illness", r"HPI",
        r"HISTORY OF PRESENT", r"Reason for Admission", r"Present Illness",
        r"CC:", r"Major Surgical", r"Brief Hospital Course",
    ]
    match = re.search("|".join(markers), text, re.IGNORECASE)
    if match:
        return text[match.start():]
    return text[500:] if len(text) > 800 else text


# ═══════════════════════════════════════════════════════════════
# T3: LOW INJECTION RATE (20%) + SHORT SNIPPETS
# ═══════════════════════════════════════════════════════════════

def build_t3_minimal(text, patient, injection_rate=0.20):
    """
    Replace only 20% of ___ markers with patient PHI.
    This means T3 notes have mostly ___ with just a few real identifiers.
    """
    positions = [m.start() for m in re.finditer(r'___', text)]
    if not positions:
        return text, [], 0
    
    n_replace = max(1, int(len(positions) * injection_rate))
    replace_set = set(random.sample(range(len(positions)), min(n_replace, len(positions))))
    
    phi_pool = [
        (f"{patient['first']} {patient['last']}", "NAME"),
        (random_date(), "DATE"),
        (patient["mrn"], "MRN"),
        (f"{patient['last']}, {patient['first']}", "NAME"),
        (random_date(), "DATE"),
        (patient["phone"], "PHONE"),
        (f"{random.randint(100,999)}-{random.randint(10,99)}-{patient['ssn_last4']}", "SSN"),
        (random_date(), "DATE"),
    ]
    
    entities = []
    result = text
    phi_idx = 0
    for i in reversed(range(len(positions))):
        pos = positions[i]
        if i in replace_set:
            val, etype = phi_pool[phi_idx % len(phi_pool)]
            result = result[:pos] + val + result[pos+3:]
            entities.append({"type": etype, "value": val})
            phi_idx += 1
    
    residual = result.count("___")
    return result, entities, residual


# ═══════════════════════════════════════════════════════════════
# T2: MARKER REPLACEMENT + NAME POLLUTION
# ═══════════════════════════════════════════════════════════════

def build_t2_polluted(text):
    """
    Replace ___ with generic placeholders AND inject provider/hospital names.
    This makes T2 contain real names (providers, hospitals) that look like
    patient names to a naive classifier.
    """
    generic = [
        "the patient", "[REDACTED]", "the individual",
        "[NAME]", "[DATE]", "a family member",
        "the provider", "[LOCATION]", "their physician",
        "the hospital", "[ID]", "a specialist",
        "2023", "recently", "the clinic",
        "[CONTACT]", "the referring", "their doctor",
        "the family", "the team", "an outside provider",
    ]
    
    result = text
    while "___" in result:
        result = result.replace("___", random.choice(generic), 1)
    
    # Inject 2-4 provider name references into the text
    n_injections = random.randint(2, 4)
    sentences = result.split('. ')
    
    for _ in range(n_injections):
        if len(sentences) < 3:
            break
        phrase_template = random.choice(PROVIDER_PHRASES)
        phrase = phrase_template.format(
            first=random.choice(FIRST_NAMES_M + FIRST_NAMES_F),
            last=random.choice(LAST_NAMES),
        )
        # Insert at random position in the text
        insert_idx = random.randint(1, len(sentences) - 1)
        sentences.insert(insert_idx, phrase)
    
    result = '. '.join(sentences)
    
    # 30% chance: also add a hospital reference
    if random.random() < 0.3:
        result += " " + random.choice(HOSPITAL_PHRASES) + "."
    
    return result


# ═══════════════════════════════════════════════════════════════
# CROSS-CONTAMINATION: blur the T3/T2 boundary further
# ═══════════════════════════════════════════════════════════════

def contaminate_t2_with_name(text):
    """Add a single patient-like name to a T2 sample (simulating imperfect deidentification)."""
    first = random.choice(FIRST_NAMES_M + FIRST_NAMES_F)
    last = random.choice(LAST_NAMES)
    
    patterns = [
        f"patient {first} {last}",
        f"Mr. {last}",
        f"Mrs. {last}",
        f"{last}, {first}",
    ]
    
    insertion = random.choice(patterns)
    # Replace one generic reference with the name
    for placeholder in ["the patient", "the individual", "[NAME]", "[REDACTED]"]:
        if placeholder in text:
            return text.replace(placeholder, insertion, 1)
    
    return text  # If no placeholder found, return as-is


def make_minimal_t3(text, patient):
    """Create a T3 sample with absolute minimum PHI (just 1 identifier)."""
    positions = [m.start() for m in re.finditer(r'___', text)]
    if not positions:
        return text, [], 0
    
    # Replace exactly 1 marker
    replace_idx = random.choice(range(len(positions)))
    pos = positions[replace_idx]
    
    val = f"{patient['first']} {patient['last']}"
    result = text[:pos] + val + text[pos+3:]
    
    # Replace remaining ___ with generic text
    generic = ["the patient", "[REDACTED]", "the provider", "[DATE]", "the clinic",
               "their physician", "the team", "recently", "[NAME]", "a specialist"]
    while "___" in result:
        result = result.replace("___", random.choice(generic), 1)
    
    return result, [{"type": "NAME", "value": val}], 0


# ═══════════════════════════════════════════════════════════════
# MIMIC LOADER
# ═══════════════════════════════════════════════════════════════

def load_mimic(path, max_notes=5000):
    notes = []
    open_fn = gzip.open if str(path).endswith('.gz') else open
    mode = 'rt' if str(path).endswith('.gz') else 'r'
    
    print(f"  Loading from {path}...")
    with open_fn(path, mode, encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if len(notes) >= max_notes:
                break
            text = row.get('text', '')
            if len(text) < 500 or text.count('___') < 3:
                continue
            notes.append({
                "note_id": row.get('note_id', str(i)),
                "subject_id": row.get('subject_id', ''),
                "text": text,
                "phi_markers": text.count('___'),
            })
    
    print(f"  {len(notes)} notes loaded")
    return notes


# ═══════════════════════════════════════════════════════════════
# T0 / T1 
# ═══════════════════════════════════════════════════════════════

def generate_t0(count=2500):
    templates = [
        "What are the common side effects of {drug} in {cond} management?",
        "Explain the pathophysiology of {cond} and its staging criteria.",
        "What is the recommended screening schedule for {cond}?",
        "What are the clinical guidelines for {cond} management in adults?",
        "How does {cond} progress through its stages?",
        "What laboratory values indicate {cond}?",
        "Explain the mechanism of action of {drug} in {cond}.",
        "What is the first-line treatment for {cond}?",
        "Describe the diagnostic criteria for {cond}.",
        "What are the risk factors for developing {cond}?",
        "How should {drug} dosage be adjusted for renal impairment?",
        "What imaging studies are indicated for suspected {cond}?",
        "What are the contraindications for {drug} use?",
        "Describe the prognosis for patients with {cond}.",
        "Compare {drug} and {drug2} for treatment of {cond}.",
    ]
    drugs = ["metformin", "lisinopril", "atorvastatin", "amlodipine", "omeprazole",
             "levothyroxine", "metoprolol", "losartan", "gabapentin", "sertraline",
             "warfarin", "insulin", "furosemide", "prednisone", "amoxicillin"]
    conds = ["type 2 diabetes", "hypertension", "heart failure", "COPD",
             "chronic kidney disease", "atrial fibrillation", "pneumonia",
             "acute MI", "stroke", "sepsis", "asthma", "DVT", "PE", "cirrhosis"]
    
    samples = []
    for _ in range(count):
        t = random.choice(templates)
        text = t.format(drug=random.choice(drugs), cond=random.choice(conds),
                        drug2=random.choice(drugs))
        if random.random() < 0.3:
            text = "Question: " + text
        samples.append({"text": text, "tier": 0, "source": "template"})
    return samples


def generate_t1(count=1500):
    templates = [
        "Schedule follow-up in {spec} clinic next {day}.",
        "Request lab panel CBC and BMP for wellness visit.",
        "Submit prior authorization for MRI lumbar spine.",
        "Notify nursing about bed assignment change.",
        "Update staffing schedule for next month's ICU rotation.",
        "Generate monthly LOS report for cardiac patients.",
        "Coordinate transport for discharge to SNF.",
        "Process medication reconciliation for admission.",
        "Send referral to {spec} for {reason}.",
        "Update infection control dashboard with hand hygiene data.",
        "Submit quality measures for CMS readmission reporting.",
        "Request EHR maintenance for scheduled downtime on {day}.",
        "Order surgical supplies for {spec} inventory.",
        "Schedule IDT meeting for discharge planning.",
        "File incident report for medication delay on unit {unit}.",
    ]
    specs = ["cardiology", "neurology", "orthopedics", "pulmonology",
             "gastroenterology", "oncology", "nephrology", "endocrinology"]
    reasons = ["evaluation", "management", "second opinion", "post-op care"]
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    units = ["3A", "4B", "ICU", "5C", "ED", "7West"]
    
    samples = []
    for _ in range(count):
        t = random.choice(templates)
        text = t.format(spec=random.choice(specs), reason=random.choice(reasons),
                        day=random.choice(days), unit=random.choice(units))
        if random.random() < 0.15:
            text = "URGENT: " + text
        samples.append({"text": text, "tier": 1, "source": "template"})
    return samples


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mimic-notes", required=True)
    parser.add_argument("--t3-count", type=int, default=3500)
    parser.add_argument("--t2-count", type=int, default=2500)
    parser.add_argument("--t0-count", type=int, default=2500)
    parser.add_argument("--t1-count", type=int, default=1500)
    parser.add_argument("--injection-rate", type=float, default=0.20)
    parser.add_argument("--t2-contamination-rate", type=float, default=0.10)
    parser.add_argument("--t3-minimal-rate", type=float, default=0.10)
    parser.add_argument("--max-chars", type=int, default=600,
                        help="Max text length per sample (shorter = harder)")
    args = parser.parse_args()
    
    print("=" * 65)
    print("HARDEST DATASET: Low injection + Name pollution + Short snippets")
    print("=" * 65)
    print(f"  Injection rate: {args.injection_rate*100:.0f}%")
    print(f"  T2 contamination: {args.t2_contamination_rate*100:.0f}%")
    print(f"  T3 minimal (1 PHI only): {args.t3_minimal_rate*100:.0f}%")
    print(f"  Max text length: {args.max_chars} chars")
    
    patients = generate_patients(500)
    
    # Load MIMIC
    print(f"\nLoading MIMIC notes...")
    notes = load_mimic(Path(args.mimic_notes), max_notes=max(args.t3_count, args.t2_count) + 500)
    random.shuffle(notes)
    
    # ── Build T3 ──
    print(f"\nBuilding T3 ({args.t3_count} samples)...")
    t3_samples = []
    n_minimal = int(args.t3_count * args.t3_minimal_rate)
    
    for i, note in enumerate(notes[:args.t3_count]):
        patient = patients[i % len(patients)]
        stripped = strip_header(note["text"])
        
        if i < n_minimal:
            # Minimal T3: exactly 1 PHI entity, all other ___ replaced with generic
            text, entities, residual = make_minimal_t3(stripped, patient)
            source = "mimic_minimal_phi"
        else:
            # Standard T3: 20% injection rate
            text, entities, residual = build_t3_minimal(stripped, patient, args.injection_rate)
            source = "mimic_partial_phi"
        
        # Truncate to max_chars
        if len(text) > args.max_chars:
            # Try to keep the portion with PHI entities
            text = text[:args.max_chars]
        
        t3_samples.append({
            "text": text,
            "tier": 3,
            "source": source,
            "note_id": note["note_id"],
            "phi_count": len(entities),
            "residual_markers": text.count("___"),
        })
    
    avg_phi = sum(s["phi_count"] for s in t3_samples) / len(t3_samples)
    avg_residual = sum(s["residual_markers"] for s in t3_samples) / len(t3_samples)
    minimal_count = sum(1 for s in t3_samples if s["source"] == "mimic_minimal_phi")
    print(f"  Avg PHI entities: {avg_phi:.1f}")
    print(f"  Avg residual ___: {avg_residual:.1f}")
    print(f"  Minimal (1 PHI): {minimal_count}")
    
    # ── Build T2 ──
    print(f"\nBuilding T2 ({args.t2_count} samples)...")
    t2_samples = []
    n_contaminated = int(args.t2_count * args.t2_contamination_rate)
    
    for i, note in enumerate(notes[:args.t2_count]):
        stripped = strip_header(note["text"])
        text = build_t2_polluted(stripped)
        
        # Cross-contamination: add a patient name to some T2 samples
        if i < n_contaminated:
            text = contaminate_t2_with_name(text)
            source = "mimic_deidentified_contaminated"
        else:
            source = "mimic_deidentified_polluted"
        
        if len(text) > args.max_chars:
            text = text[:args.max_chars]
        
        t2_samples.append({
            "text": text,
            "tier": 2,
            "source": source,
            "note_id": note["note_id"],
        })
    
    contaminated_count = sum(1 for s in t2_samples if s["source"] == "mimic_deidentified_contaminated")
    print(f"  Polluted with provider names: {len(t2_samples)}")
    print(f"  Contaminated with patient name: {contaminated_count}")
    
    # ── T0, T1 ──
    print(f"\nBuilding T0 ({args.t0_count}) and T1 ({args.t1_count})...")
    t0 = generate_t0(args.t0_count)
    t1 = generate_t1(args.t1_count)
    
    # ── Combine ──
    dataset = t3_samples + t2_samples + t0 + t1
    random.shuffle(dataset)
    
    total = len(dataset)
    print(f"\nTOTAL: {total}")
    for tier in range(4):
        c = sum(1 for s in dataset if s["tier"] == tier)
        print(f"  T{tier}: {c}")
    
    # ── Split ──
    print(f"\nSplitting (no pair leakage)...")
    note_groups = {}
    ungrouped = []
    for item in dataset:
        nid = item.get("note_id")
        if nid:
            note_groups.setdefault(nid, []).append(item)
        else:
            ungrouped.append(item)
    
    groups = list(note_groups.values())
    random.shuffle(groups)
    random.shuffle(ungrouped)
    
    ng = len(groups)
    t_end = int(ng * 0.5)
    v_end = int(ng * 0.6)
    
    train, val, test = [], [], []
    for i, g in enumerate(groups):
        (train if i < t_end else val if i < v_end else test).extend(g)
    
    nug = len(ungrouped)
    train.extend(ungrouped[:int(nug*0.5)])
    val.extend(ungrouped[int(nug*0.5):int(nug*0.6)])
    test.extend(ungrouped[int(nug*0.6):])
    
    for s in [train, val, test]:
        random.shuffle(s)
    
    print(f"  Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    for name, split in [("Train", train), ("Val", val), ("Test", test)]:
        dist = {}
        for s in split:
            dist[s["tier"]] = dist.get(s["tier"], 0) + 1
        print(f"  {name}: {', '.join(f'T{t}:{c}' for t,c in sorted(dist.items()))}")
    
    # ── Save ──
    out_dir = ROOT / "data"
    out_dir.mkdir(exist_ok=True)
    for name, split in [("train", train), ("val", val), ("test", test)]:
        with open(out_dir / f"{name}.json", "w") as f:
            json.dump(split, f, indent=2, default=str)
    
    print(f"\nSaved to {out_dir}/")
    
    # ── Samples ──
    print(f"\n{'='*65}")
    print("SAMPLES (notice how similar T3 and T2 look)")
    print(f"{'='*65}")
    
    t3_min = next((s for s in dataset if s.get("source") == "mimic_minimal_phi"), None)
    t3_std = next((s for s in dataset if s.get("source") == "mimic_partial_phi"), None)
    t2_con = next((s for s in dataset if s.get("source") == "mimic_deidentified_contaminated"), None)
    t2_std = next((s for s in dataset if s.get("source") == "mimic_deidentified_polluted"), None)
    
    for label, sample in [("T3 MINIMAL (1 PHI)", t3_min), ("T3 STANDARD (20% PHI)", t3_std),
                           ("T2 CONTAMINATED (fake patient name)", t2_con), ("T2 STANDARD (provider names)", t2_std)]:
        if sample:
            print(f"\n--- {label} ---")
            print(sample["text"][:300] + "...")
    
    print(f"\n{'='*65}")
    print("DIFFICULTY FEATURES")
    print(f"{'='*65}")
    print(f"  T3 has {avg_phi:.1f} avg PHI entities (was 24.4 in Script 18)")
    print(f"  T3 still has {avg_residual:.1f} avg ___ markers (looks like T2)")
    print(f"  T2 has provider names: 'Dr. Smith ordered...' (looks like T3)")
    print(f"  {contaminated_count} T2 samples have actual patient-like names")
    print(f"  {minimal_count} T3 samples have only 1 PHI entity")
    print(f"  Max text length: {args.max_chars} chars (less context to work with)")
    print(f"\n  Expected T3 recall: 85-93%")
    print(f"\nNext: python scripts/02_train_classifier.py")


if __name__ == "__main__":
    main()
