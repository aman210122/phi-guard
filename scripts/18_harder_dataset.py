#!/usr/bin/env python3
"""
18_harder_dataset.py — Create a GENUINELY HARD dataset

PROBLEM:
  With standard PHI re-injection, T3 recall = 100%. DistilBERT trivially 
  learns "real names present → T3, ___ present → T2". This means CompTS
  has no errors to protect against, and the paper has no story.

SOLUTION (3 techniques to make classification realistically hard):

  1. HEADER STRIPPING: Remove structured header fields (Name, Unit No, DOB,
     Admission/Discharge Date) from ALL notes. Only keep clinical narrative
     starting from "Chief Complaint" or "History of Present Illness".
     This prevents the classifier from using header structure as a shortcut.

  2. PARTIAL PHI RE-INJECTION: For T3 samples, only replace 30-50% of ___
     markers with synthetic PHI. The remaining ___ stay as-is. This means
     T3 notes still have some ___ markers (like T2), but ALSO have real
     names/dates/MRNs scattered in the clinical narrative.

  3. MARKER REPLACEMENT IN T2: Replace ___ in T2 samples with plausible
     but NON-PHI tokens like "[REDACTED]", "[NAME]", "the patient", 
     generic dates "2023", etc. This prevents the classifier from using
     the literal string "___" as a T2 indicator.

  The result: T3 has some real PHI embedded in clinical text alongside
  residual ___ markers. T2 has generic placeholders. The classifier must
  learn to detect ACTUAL PHI entities, not formatting artifacts.

EXPECTED T3 RECALL: 90-95% (realistic for production)

USAGE:
  python scripts/18_harder_dataset.py \
    --mimic-notes "C:\path\to\discharge.csv.gz" \
    --t3-count 3500 --t2-count 2500 --t0-count 2500 --t1-count 1500
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
from typing import Dict, List, Tuple, Optional

ROOT = Path(__file__).resolve().parent.parent
SEED = 42
random.seed(SEED)

# ═══════════════════════════════════════════════════════════════
# SYNTHETIC PATIENT POOL
# ═══════════════════════════════════════════════════════════════

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
              "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson"]


def generate_patients(n=500):
    patients = []
    for _ in range(n):
        gender = random.choice(["male", "female"])
        first = random.choice(FIRST_NAMES_M if gender == "male" else FIRST_NAMES_F)
        last = random.choice(LAST_NAMES)
        year = random.randint(1940, 2005)
        month = random.randint(1, 12)
        day = random.randint(1, 28)
        patients.append({
            "first": first, "last": last,
            "dob": f"{year}-{month:02d}-{day:02d}",
            "mrn": f"{random.randint(1000000, 9999999)}",
            "ssn_last4": f"{random.randint(1000, 9999)}",
            "phone": f"({random.randint(200,999)}) {random.randint(200,999)}-{random.randint(1000,9999)}",
            "address": f"{random.randint(1,9999)} {random.choice(['Main','Oak','Elm','Park','Cedar','Maple','Pine'])} St",
            "city": random.choice(["Boston", "Worcester", "Springfield", "Cambridge",
                                   "Chicago", "Houston", "Phoenix", "Dallas"]),
            "state": random.choice(["MA", "IL", "TX", "AZ", "CA", "NY"]),
        })
    return patients


def random_date():
    base = datetime(2018, 1, 1)
    d = base + timedelta(days=random.randint(0, 2000))
    fmt = random.choice(["%Y-%m-%d", "%m/%d/%Y", "%B %d, %Y", "%m/%d/%y"])
    return d.strftime(fmt)


# ═══════════════════════════════════════════════════════════════
# HEADER STRIPPING — the key to making classification hard
# ═══════════════════════════════════════════════════════════════

def strip_header(text):
    """
    Remove structured header from MIMIC discharge notes.
    Keep only clinical narrative starting from Chief Complaint,
    History of Present Illness, or similar clinical sections.
    """
    # Common clinical section starts
    section_markers = [
        r"Chief Complaint",
        r"History of Present Illness",
        r"HPI",
        r"HISTORY OF PRESENT",
        r"Reason for Admission",
        r"Present Illness",
        r"CC:",
        r"Major Surgical",
        r"Brief Hospital Course",
    ]
    
    pattern = "|".join(section_markers)
    match = re.search(pattern, text, re.IGNORECASE)
    
    if match:
        return text[match.start():]
    
    # Fallback: skip first 500 chars (usually header)
    if len(text) > 800:
        return text[500:]
    
    return text


# ═══════════════════════════════════════════════════════════════
# PARTIAL PHI RE-INJECTION (T3) — only replace some markers
# ═══════════════════════════════════════════════════════════════

def partial_reinject_phi(text, patient, injection_rate=0.45):
    """
    Replace only injection_rate fraction of ___ markers with synthetic PHI.
    The rest stay as ___ (making T3 look partially like T2).
    
    This is the key to realistic classification difficulty:
    T3 notes have SOME real PHI mixed with SOME ___ markers.
    """
    entities = []
    
    # Find all ___ positions
    positions = [m.start() for m in re.finditer(r'___', text)]
    if not positions:
        return text, []
    
    # Decide which ones to replace
    n_replace = max(2, int(len(positions) * injection_rate))  # At least 2 PHI entities
    replace_indices = set(random.sample(range(len(positions)), min(n_replace, len(positions))))
    
    # Build replacement list
    phi_values = [
        (patient["first"] + " " + patient["last"], "NAME"),
        (random_date(), "DATE"),
        (patient["mrn"], "MRN"),
        (f"Dr. {random.choice(LAST_NAMES)}", "PROVIDER"),
        (random_date(), "DATE"),
        (patient["phone"], "PHONE"),
        (patient["last"] + ", " + patient["first"], "NAME"),
        (random_date(), "DATE"),
        (patient["address"], "ADDRESS"),
        (patient["city"] + ", " + patient["state"], "LOCATION"),
        (random_date(), "DATE"),
        (f"{random.randint(100,999)}-{random.randint(10,99)}-{patient['ssn_last4']}", "SSN_PARTIAL"),
    ]
    
    # Replace from end to start (to preserve positions)
    result = text
    phi_idx = 0
    for i in reversed(range(len(positions))):
        pos = positions[i]
        if i in replace_indices:
            val, etype = phi_values[phi_idx % len(phi_values)]
            result = result[:pos] + val + result[pos+3:]  # 3 = len("___")
            entities.append({"type": etype, "value": val})
            phi_idx += 1
        # else: leave ___ as-is
    
    return result, entities


# ═══════════════════════════════════════════════════════════════
# T2 MARKER REPLACEMENT — prevent ___ from being a trivial feature
# ═══════════════════════════════════════════════════════════════

def replace_markers_t2(text):
    """
    Replace ___ in T2 with varied generic placeholders.
    This prevents the classifier from learning "___ → T2".
    """
    replacements = [
        "the patient", "[REDACTED]", "the individual", 
        "[NAME]", "[DATE]", "a family member",
        "the provider", "[LOCATION]", "their physician",
        "the hospital", "[ID]", "a specialist",
        "2023", "recently", "the clinic",
        "[CONTACT]", "the referring", "their doctor",
    ]
    
    result = text
    while "___" in result:
        replacement = random.choice(replacements)
        result = result.replace("___", replacement, 1)
    
    return result


# ═══════════════════════════════════════════════════════════════
# MIMIC LOADER
# ═══════════════════════════════════════════════════════════════

def load_mimic_notes(path, max_notes=5000):
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
            if len(text) < 500:  # Need substantial notes
                continue
            if text.count('___') < 3:  # Need PHI markers
                continue
            notes.append({
                "note_id": row.get('note_id', str(i)),
                "subject_id": row.get('subject_id', ''),
                "text": text,
                "phi_markers": text.count('___'),
            })
    
    print(f"  Loaded {len(notes)} notes (avg {sum(n['phi_markers'] for n in notes)/max(1,len(notes)):.1f} PHI markers)")
    return notes


# ═══════════════════════════════════════════════════════════════
# T0 / T1 GENERATION
# ═══════════════════════════════════════════════════════════════

def generate_t0(count=2500):
    """T0: General medical knowledge — no patient context."""
    templates = [
        "What are the common side effects of {drug} in {condition} management?",
        "Explain the pathophysiology of {condition} and its staging criteria.",
        "What is the recommended screening schedule for {condition}?",
        "Describe the difference between {condition} and {condition2}.",
        "What are the clinical guidelines for {condition} management in adults?",
        "How does {condition} progress through its stages?",
        "What laboratory values indicate {condition}?",
        "Explain the mechanism of action of {drug} in {condition}.",
        "What is the first-line treatment for {condition}?",
        "Describe the diagnostic criteria for {condition}.",
        "What are the risk factors for developing {condition}?",
        "How should {drug} dosage be adjusted for renal impairment?",
        "What imaging studies are indicated for suspected {condition}?",
        "Explain the role of {drug} in the treatment of {condition}.",
        "What are the contraindications for {drug} use?",
        "Describe the prognosis for patients with {condition}.",
    ]
    drugs = ["metformin", "lisinopril", "atorvastatin", "amlodipine", "omeprazole",
             "levothyroxine", "metoprolol", "losartan", "gabapentin", "sertraline",
             "warfarin", "insulin glargine", "furosemide", "prednisone", "amoxicillin",
             "clopidogrel", "pantoprazole", "hydrochlorothiazide", "acetaminophen"]
    conditions = ["type 2 diabetes", "hypertension", "heart failure", "COPD",
                  "chronic kidney disease", "atrial fibrillation", "pneumonia",
                  "acute myocardial infarction", "stroke", "sepsis", "asthma",
                  "deep vein thrombosis", "pulmonary embolism", "cirrhosis",
                  "pancreatitis", "diverticulitis", "cellulitis", "UTI",
                  "osteoarthritis", "rheumatoid arthritis", "lupus"]
    
    samples = []
    for _ in range(count):
        t = random.choice(templates)
        text = t.format(
            drug=random.choice(drugs),
            condition=random.choice(conditions),
            condition2=random.choice(conditions),
        )
        # Add variation
        if random.random() < 0.3:
            text = "Question: " + text
        if random.random() < 0.2:
            text += f" Consider patients over {random.randint(40,80)} years old."
        samples.append({"text": text, "tier": 0, "source": "template_medical"})
    
    return samples


def generate_t1(count=1500):
    """T1: Internal/operational — scheduling, admin, logistics."""
    templates = [
        "Schedule a follow-up appointment for the cardiology clinic next {day}.",
        "Request lab panel CBC and BMP for upcoming annual wellness visit.",
        "Submit prior authorization for MRI lumbar spine to insurance.",
        "Notify nursing station about bed assignment change for incoming transfer.",
        "Update the department staffing schedule for next month's ICU rotation.",
        "Generate monthly report on average length of stay for cardiac patients.",
        "Coordinate ambulance transport for discharge to skilled nursing facility.",
        "Process medication reconciliation checklist for new admission protocol.",
        "Send referral request to {specialty} for {reason} consultation.",
        "Update infection control dashboard with this week's hand hygiene data.",
        "Submit quality measure data for CMS reporting on readmission rates.",
        "Request maintenance for EHR system downtime scheduled for {day}.",
        "Order surgical supplies for {specialty} department inventory replenishment.",
        "Schedule interdisciplinary team meeting for discharge planning review.",
        "File incident report for medication administration delay on unit {unit}.",
        "Prepare conference room for {specialty} department grand rounds presentation.",
    ]
    specialties = ["endocrinology", "cardiology", "neurology", "orthopedics",
                   "pulmonology", "gastroenterology", "oncology", "nephrology"]
    reasons = ["diabetes management", "cardiac evaluation", "pain management",
               "post-operative care", "medication adjustment", "second opinion"]
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    units = ["3A", "4B", "ICU", "5C", "ED", "2North", "7West"]
    
    samples = []
    for _ in range(count):
        t = random.choice(templates)
        text = t.format(
            specialty=random.choice(specialties),
            reason=random.choice(reasons),
            day=random.choice(days),
            unit=random.choice(units),
        )
        if random.random() < 0.15:
            text = "URGENT: " + text
        if random.random() < 0.2:
            text = text.lower()
        samples.append({"text": text, "tier": 1, "source": "template_operational"})
    
    return samples


# ═══════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mimic-notes", required=True)
    parser.add_argument("--t3-count", type=int, default=3500)
    parser.add_argument("--t2-count", type=int, default=2500)
    parser.add_argument("--t0-count", type=int, default=2500)
    parser.add_argument("--t1-count", type=int, default=1500)
    parser.add_argument("--injection-rate", type=float, default=0.45,
                        help="Fraction of ___ markers to replace with PHI in T3 (default 0.45)")
    parser.add_argument("--output-dir", default="data")
    args = parser.parse_args()
    
    print("=" * 65)
    print("HARDER DATASET: Header Stripping + Partial PHI + Marker Replacement")
    print("=" * 65)
    
    # Load patients
    print("\n1. Generating synthetic patient pool...")
    patients = generate_patients(500)
    print(f"   {len(patients)} patients")
    
    # Load MIMIC
    print("\n2. Loading MIMIC discharge notes...")
    mimic_path = Path(args.mimic_notes)
    if not mimic_path.exists():
        print(f"ERROR: {mimic_path} not found")
        sys.exit(1)
    
    max_needed = max(args.t3_count, args.t2_count)
    notes = load_mimic_notes(mimic_path, max_notes=max_needed + 500)
    random.shuffle(notes)
    
    # Build T3: Strip header → Partial PHI re-injection
    print(f"\n3. Building T3 ({args.t3_count} samples): header strip + {args.injection_rate*100:.0f}% PHI injection")
    t3_samples = []
    for i, note in enumerate(notes[:args.t3_count]):
        patient = patients[i % len(patients)]
        
        # Strip header first
        stripped = strip_header(note["text"])
        
        # Partial PHI re-injection
        reinjected, entities = partial_reinject_phi(stripped, patient, args.injection_rate)
        
        # Truncate
        if len(reinjected) > 1500:
            reinjected = reinjected[:800] + "\n...\n" + reinjected[-500:]
        
        t3_samples.append({
            "text": reinjected,
            "tier": 3,
            "source": "mimic_partial_phi",
            "note_id": note["note_id"],
            "phi_count": len(entities),
            "residual_markers": reinjected.count("___"),
        })
    
    avg_phi = sum(s["phi_count"] for s in t3_samples) / len(t3_samples)
    avg_residual = sum(s["residual_markers"] for s in t3_samples) / len(t3_samples)
    print(f"   Avg PHI entities injected: {avg_phi:.1f}")
    print(f"   Avg residual ___ markers: {avg_residual:.1f}")
    
    # Build T2: Strip header → Replace ___ with generic placeholders
    print(f"\n4. Building T2 ({args.t2_count} samples): header strip + marker replacement")
    t2_samples = []
    for note in notes[:args.t2_count]:
        stripped = strip_header(note["text"])
        cleaned = replace_markers_t2(stripped)
        
        if len(cleaned) > 1500:
            cleaned = cleaned[:800] + "\n...\n" + cleaned[-500:]
        
        t2_samples.append({
            "text": cleaned,
            "tier": 2,
            "source": "mimic_deidentified_cleaned",
            "note_id": note["note_id"],
        })
    
    # Build T0 and T1
    print(f"\n5. Building T0 ({args.t0_count}) and T1 ({args.t1_count})")
    t0_samples = generate_t0(args.t0_count)
    t1_samples = generate_t1(args.t1_count)
    
    # Combine
    dataset = t3_samples + t2_samples + t0_samples + t1_samples
    random.shuffle(dataset)
    
    total = len(dataset)
    print(f"\n   TOTAL: {total} samples")
    for tier in range(4):
        count = sum(1 for s in dataset if s["tier"] == tier)
        labels = {0: "T0_Public", 1: "T1_Internal", 2: "T2_Limited", 3: "T3_Restricted"}
        print(f"   {labels[tier]}: {count}")
    
    # Split (50/10/40) with no note_id leakage
    print("\n6. Splitting (no pair leakage)...")
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
    train_end = int(ng * 0.5)
    val_end = int(ng * 0.6)
    
    train, val, test = [], [], []
    for i, g in enumerate(groups):
        if i < train_end:
            train.extend(g)
        elif i < val_end:
            val.extend(g)
        else:
            test.extend(g)
    
    nug = len(ungrouped)
    train.extend(ungrouped[:int(nug*0.5)])
    val.extend(ungrouped[int(nug*0.5):int(nug*0.6)])
    test.extend(ungrouped[int(nug*0.6):])
    
    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)
    
    print(f"   Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    
    # Show tier distribution per split
    for name, split in [("Train", train), ("Val", val), ("Test", test)]:
        tier_dist = {}
        for s in split:
            tier_dist[s["tier"]] = tier_dist.get(s["tier"], 0) + 1
        dist_str = ", ".join(f"T{t}:{c}" for t, c in sorted(tier_dist.items()))
        print(f"   {name}: {dist_str}")
    
    # Save
    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for name, split in [("train", train), ("val", val), ("test", test)]:
        path = out_dir / f"{name}.json"
        with open(path, "w") as f:
            json.dump(split, f, indent=2, default=str)
    
    full_path = out_dir / "mimic_dataset_hard.json"
    with open(full_path, "w") as f:
        json.dump(dataset, f, indent=2, default=str)
    
    print(f"\n   Saved to {out_dir}/")
    
    # Show samples
    print(f"\n{'='*65}")
    print("SAMPLE COMPARISON (first 200 chars of clinical narrative)")
    print(f"{'='*65}")
    
    t3_ex = next(s for s in dataset if s["tier"] == 3)
    t2_ex = next(s for s in dataset if s["tier"] == 2)
    
    print(f"\n--- T3 (PHI present, {t3_ex.get('phi_count',0)} entities, {t3_ex.get('residual_markers',0)} residual ___) ---")
    print(t3_ex["text"][:250] + "...")
    
    print(f"\n--- T2 (no PHI, markers replaced with generic text) ---")
    print(t2_ex["text"][:250] + "...")
    
    print(f"\n{'='*65}")
    print("WHY THIS IS HARDER")
    print(f"{'='*65}")
    print(f"  - Headers stripped: classifier can't use Name/DOB/Unit No fields")
    print(f"  - Partial injection ({args.injection_rate*100:.0f}%): T3 still has {avg_residual:.0f} avg ___ markers")
    print(f"  - Marker replacement: T2 uses '[REDACTED]', 'the patient' instead of ___")
    print(f"  - Expected T3 recall: 90-95% (not 100%)")
    print(f"\nNow run: python scripts/02_train_classifier.py")


if __name__ == "__main__":
    main()
