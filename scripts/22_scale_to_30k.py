#!/usr/bin/env python3
"""
22_scale_to_30k.py — Scale dataset to 30K+ for NeurIPS

PURPOSE:
  NeurIPS items 5 & 6:
  - Scale from 10K to 30K+ queries
  - Add MIMIC-III cross-validation
  - Tighten CARES bound via larger calibration set (n≈9,000 → δ+1/(n+1)≈0.0501)

APPROACH:
  Uses the SAME hardening techniques as 20_hardest_dataset.py (low injection,
  name pollution, header stripping) but at 3× scale:
  
    T0: 2500 → 7500  (MedQA + PubMedQA + MedMCQA)
    T1: 1500 → 4500  (expanded operational templates)
    T2: 2500 → 7500  (more MIMIC-IV notes, same pollution)
    T3: 3500 → 10500 (more MIMIC-IV notes, same 20% injection)
    TOTAL: 10K → 30K
  
  Additionally creates a MIMIC-III cross-validation set (10K) to test
  generalization across hospital systems.

PREREQUISITES:
  - MIMIC-IV-Note v2.2 (you already have this)
  - MIMIC-III v1.4 Clinical Notes (request at physionet.org if not already)
  - For expanded T0: pip install datasets (for PubMedQA, MedMCQA from HuggingFace)

SPLIT STRATEGY:
  30K dataset → 60/10/30 = 18K train, 3K val, 9K test
  9K test → 30% calibration (2,700) + 70% evaluation (6,300)
  
  CARES bound: δ + 1/(n+1) = 0.05 + 1/2701 = 0.0504 (vs current 0.0508)
  With full 9K as calibration: 0.05 + 1/9001 = 0.0501

USAGE:
  # Standard 30K (uses more MIMIC-IV notes)
  python scripts/22_scale_to_30k.py --mimic-notes /path/to/discharge.csv.gz

  # With MIMIC-III cross-validation
  python scripts/22_scale_to_30k.py --mimic-notes /path/to/mimic4/discharge.csv.gz \
                                    --mimic3-notes /path/to/mimic3/NOTEEVENTS.csv.gz

  # With HuggingFace datasets for expanded T0
  python scripts/22_scale_to_30k.py --mimic-notes /path/to/discharge.csv.gz \
                                    --use-hf-datasets
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
from typing import List, Dict

ROOT = Path(__file__).resolve().parent.parent
SEED = 42
random.seed(SEED)

# ═══════════════════════════════════════════════════════════════
# Import shared components from 20_hardest_dataset.py
# ═══════════════════════════════════════════════════════════════

# We duplicate the key functions here for self-containment.
# In practice, you could import from 20_hardest_dataset.py.

FIRST_NAMES_M = ["James","Robert","Michael","David","Richard","Joseph",
    "Thomas","Charles","Daniel","Matthew","Anthony","Mark","Steven","Paul",
    "Andrew","Joshua","Kenneth","Kevin","Brian","George","Timothy","Ronald",
    "Edward","Jason","Jeffrey","Ryan","Jacob","Nicholas","Gary","Eric",
    "William","Christopher","Patrick","Frank","Raymond","Gregory","Harold"]
FIRST_NAMES_F = ["Mary","Patricia","Jennifer","Linda","Barbara","Elizabeth",
    "Susan","Jessica","Sarah","Karen","Lisa","Nancy","Margaret","Sandra",
    "Ashley","Dorothy","Kimberly","Emily","Donna","Michelle","Carol",
    "Amanda","Melissa","Deborah","Stephanie","Rebecca","Sharon","Laura",
    "Cynthia","Amy","Angela","Teresa","Helen","Gloria","Cheryl","Frances"]
LAST_NAMES = ["Smith","Johnson","Williams","Brown","Jones","Garcia","Miller",
    "Davis","Rodriguez","Martinez","Hernandez","Lopez","Gonzalez","Wilson",
    "Anderson","Thomas","Taylor","Moore","Jackson","Martin","Lee","Perez",
    "Thompson","White","Harris","Sanchez","Clark","Ramirez","Lewis",
    "Robinson","Walker","Young","Allen","King","Wright","Scott","Adams",
    "Nelson","Hill","Baker","Green","Campbell","Mitchell","Roberts","Carter"]

PROVIDER_PHRASES = [
    "Dr. {last} ordered","per Dr. {last}'s assessment",
    "Dr. {first} {last} evaluated the patient","attending Dr. {last}",
    "discussed with Dr. {last}","Dr. {last} was consulted",
    "seen by Dr. {first} {last} in consultation","Dr. {last} recommended",
    "per {first} {last}, RN","nurse {first} administered",
    "{first} {last}, PA-C assessed","Dr. {last} performed the procedure",
]

HOSPITAL_PHRASES = [
    "transferred from Massachusetts General","previously seen at Beth Israel Deaconess",
    "follow up at Brigham and Women's","referred from Mount Sinai",
    "outside records from Johns Hopkins","prior admission to Cleveland Clinic",
]


def generate_patients(n=1000):
    patients = []
    for _ in range(n):
        gender = random.choice(["male","female"])
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
    d = datetime(2018,1,1) + timedelta(days=random.randint(0,2000))
    return d.strftime(random.choice(["%Y-%m-%d","%m/%d/%Y","%B %d, %Y"]))


def strip_header(text):
    markers = [r"Chief Complaint",r"History of Present Illness",r"HPI",
               r"HISTORY OF PRESENT",r"Reason for Admission",r"Brief Hospital Course"]
    match = re.search("|".join(markers), text, re.IGNORECASE)
    if match:
        return text[match.start():]
    return text[500:] if len(text) > 800 else text


def build_t3(text, patient, injection_rate=0.20, minimal=False):
    """Build T3 with 20% PHI injection (same as script 20)."""
    positions = [m.start() for m in re.finditer(r'___', text)]
    if not positions:
        return text, [], 0
    
    if minimal:
        # Exactly 1 PHI entity
        pos = positions[random.choice(range(len(positions)))]
        val = f"{patient['first']} {patient['last']}"
        result = text[:pos] + val + text[pos+3:]
        generic = ["the patient","[REDACTED]","the provider","[DATE]","the clinic"]
        while "___" in result:
            result = result.replace("___", random.choice(generic), 1)
        return result, [{"type":"NAME","value":val}], 0
    
    n_replace = max(1, int(len(positions) * injection_rate))
    replace_set = set(random.sample(range(len(positions)), min(n_replace, len(positions))))
    
    phi_pool = [
        (f"{patient['first']} {patient['last']}","NAME"),
        (random_date(),"DATE"), (patient["mrn"],"MRN"),
        (f"{patient['last']}, {patient['first']}","NAME"),
        (random_date(),"DATE"), (patient["phone"],"PHONE"),
    ]
    
    entities = []
    result = text
    phi_idx = 0
    for i in reversed(range(len(positions))):
        pos = positions[i]
        if i in replace_set:
            val, etype = phi_pool[phi_idx % len(phi_pool)]
            result = result[:pos] + val + result[pos+3:]
            entities.append({"type":etype,"value":val})
            phi_idx += 1
    
    return result, entities, result.count("___")


def build_t2(text):
    """Build T2 with provider name pollution (same as script 20)."""
    generic = ["the patient","[REDACTED]","[NAME]","[DATE]","a specialist",
               "the provider","[LOCATION]","their physician","the hospital",
               "[ID]","the clinic","[CONTACT]","the team","recently"]
    
    result = text
    while "___" in result:
        result = result.replace("___", random.choice(generic), 1)
    
    # Inject 2-4 provider names
    sentences = result.split('. ')
    for _ in range(random.randint(2, 4)):
        if len(sentences) < 3: break
        phrase = random.choice(PROVIDER_PHRASES).format(
            first=random.choice(FIRST_NAMES_M + FIRST_NAMES_F),
            last=random.choice(LAST_NAMES))
        sentences.insert(random.randint(1, len(sentences)-1), phrase)
    
    result = '. '.join(sentences)
    if random.random() < 0.3:
        result += " " + random.choice(HOSPITAL_PHRASES) + "."
    return result


def contaminate_t2(text):
    """Add patient-like name to T2 (10% contamination)."""
    first = random.choice(FIRST_NAMES_M + FIRST_NAMES_F)
    last = random.choice(LAST_NAMES)
    name = random.choice([f"patient {first} {last}", f"Mr. {last}", f"{last}, {first}"])
    for ph in ["the patient","the individual","[NAME]","[REDACTED]"]:
        if ph in text:
            return text.replace(ph, name, 1)
    return text


def load_mimic(path, max_notes=15000):
    """Load MIMIC discharge notes."""
    notes = []
    open_fn = gzip.open if str(path).endswith('.gz') else open
    mode = 'rt' if str(path).endswith('.gz') else 'r'
    
    with open_fn(path, mode, encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if len(notes) >= max_notes: break
            text = row.get('text', '')
            if len(text) < 500 or text.count('___') < 3: continue
            notes.append({
                "note_id": row.get('note_id', str(i)),
                "subject_id": row.get('subject_id', ''),
                "text": text,
            })
    return notes


def load_mimic3(path, max_notes=10000):
    """Load MIMIC-III NOTEEVENTS (different schema from MIMIC-IV)."""
    notes = []
    open_fn = gzip.open if str(path).endswith('.gz') else open
    mode = 'rt' if str(path).endswith('.gz') else 'r'
    
    with open_fn(path, mode, encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if len(notes) >= max_notes: break
            # MIMIC-III uses CATEGORY, TEXT, SUBJECT_ID
            cat = row.get('CATEGORY', '')
            if cat not in ('Discharge summary', 'Nursing', 'Physician'): continue
            text = row.get('TEXT', '')
            if len(text) < 500: continue
            # MIMIC-III uses [**...**] for de-id markers instead of ___
            text_normalized = re.sub(r'\[\*\*[^\]]*\*\*\]', '___', text)
            if text_normalized.count('___') < 3: continue
            notes.append({
                "note_id": row.get('ROW_ID', str(i)),
                "subject_id": row.get('SUBJECT_ID', ''),
                "text": text_normalized,
            })
    return notes


# ═══════════════════════════════════════════════════════════════
# EXPANDED T0 (MedQA + PubMedQA + MedMCQA)
# ═══════════════════════════════════════════════════════════════

def generate_t0_expanded(count=7500, use_hf=False):
    """Generate T0 queries from templates + optional HuggingFace datasets."""
    samples = []
    
    # Templates (same as script 20)
    drugs = ["metformin","lisinopril","atorvastatin","amlodipine","omeprazole",
             "levothyroxine","metoprolol","losartan","gabapentin","sertraline",
             "warfarin","insulin","furosemide","prednisone","amoxicillin",
             "clopidogrel","albuterol","pantoprazole","tramadol","duloxetine"]
    conds = ["type 2 diabetes","hypertension","heart failure","COPD",
             "chronic kidney disease","atrial fibrillation","pneumonia",
             "acute MI","stroke","sepsis","asthma","DVT","PE","cirrhosis",
             "epilepsy","Parkinson's","rheumatoid arthritis","osteoporosis"]
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
        "What is the WHO classification of {cond}?",
        "Explain the pharmacokinetics of {drug}.",
        "What monitoring is required for patients on {drug}?",
    ]
    
    for _ in range(min(count, 3000)):  # templates cover first 3K
        t = random.choice(templates)
        text = t.format(drug=random.choice(drugs), cond=random.choice(conds),
                        drug2=random.choice(drugs))
        samples.append({"text": text, "tier": 0, "source": "template"})
    
    # HuggingFace datasets for diversity
    if use_hf and len(samples) < count:
        try:
            from datasets import load_dataset
            
            # PubMedQA
            print("  Loading PubMedQA from HuggingFace...")
            ds = load_dataset("pubmed_qa", "pqa_labeled", split="train", trust_remote_code=True)
            for item in ds:
                if len(samples) >= count: break
                q = item.get("question", "")
                if len(q) > 20:
                    samples.append({"text": q, "tier": 0, "source": "pubmedqa"})
            
            # MedMCQA
            print("  Loading MedMCQA from HuggingFace...")
            ds = load_dataset("openlifescienceai/medmcqa", split="train", trust_remote_code=True)
            for item in ds:
                if len(samples) >= count: break
                q = item.get("question", "")
                if len(q) > 20:
                    samples.append({"text": q, "tier": 0, "source": "medmcqa"})
        except Exception as e:
            print(f"  HuggingFace loading failed: {e}. Using templates only.")
    
    # Fill remaining with templates
    while len(samples) < count:
        t = random.choice(templates)
        text = t.format(drug=random.choice(drugs), cond=random.choice(conds),
                        drug2=random.choice(drugs))
        samples.append({"text": text, "tier": 0, "source": "template"})
    
    return samples[:count]


def generate_t1_expanded(count=4500):
    """Expanded operational templates."""
    templates = [
        "Schedule follow-up in {spec} clinic next {day}.",
        "Request lab panel CBC and BMP for wellness visit.",
        "Submit prior authorization for MRI lumbar spine.",
        "Notify nursing about bed assignment change.",
        "Update staffing schedule for next month's ICU rotation.",
        "Generate monthly LOS report for cardiac patients.",
        "Coordinate transport for discharge to SNF.",
        "Process medication reconciliation for admission.",
        "Send referral to {spec} for evaluation.",
        "Update infection control dashboard with hand hygiene data.",
        "Submit quality measures for CMS readmission reporting.",
        "Request EHR maintenance for scheduled downtime on {day}.",
        "Order surgical supplies for {spec} inventory.",
        "Schedule IDT meeting for discharge planning.",
        "File incident report for medication delay on unit {unit}.",
        "Run billing audit for {spec} department charges.",
        "Update {spec} on-call roster for the holiday weekend.",
        "Prepare department utilization dashboard for Q3.",
        "Submit compliance training records for annual review.",
        "Coordinate interpreter services for {day} clinic.",
    ]
    specs = ["cardiology","neurology","orthopedics","pulmonology",
             "gastroenterology","oncology","nephrology","endocrinology",
             "rheumatology","urology","dermatology","psychiatry"]
    days = ["Monday","Tuesday","Wednesday","Thursday","Friday"]
    units = ["3A","4B","ICU","5C","ED","7West","8North","PACU"]
    
    samples = []
    for _ in range(count):
        t = random.choice(templates)
        text = t.format(spec=random.choice(specs), day=random.choice(days),
                        unit=random.choice(units))
        samples.append({"text": text, "tier": 1, "source": "template"})
    return samples


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mimic-notes", required=True,
                        help="Path to MIMIC-IV discharge.csv.gz")
    parser.add_argument("--mimic3-notes", default=None,
                        help="Path to MIMIC-III NOTEEVENTS.csv.gz (for cross-validation)")
    parser.add_argument("--t3-count", type=int, default=10500)
    parser.add_argument("--t2-count", type=int, default=7500)
    parser.add_argument("--t0-count", type=int, default=7500)
    parser.add_argument("--t1-count", type=int, default=4500)
    parser.add_argument("--injection-rate", type=float, default=0.20)
    parser.add_argument("--max-chars", type=int, default=600)
    parser.add_argument("--use-hf-datasets", action="store_true",
                        help="Use HuggingFace datasets for expanded T0")
    args = parser.parse_args()
    
    total = args.t3_count + args.t2_count + args.t0_count + args.t1_count
    print("=" * 70)
    print(f"SCALING TO {total:,} QUERIES FOR NeurIPS")
    print("=" * 70)
    print(f"  T0={args.t0_count}, T1={args.t1_count}, T2={args.t2_count}, T3={args.t3_count}")
    
    patients = generate_patients(1000)
    
    # Load MIMIC-IV (need more notes than before)
    max_needed = max(args.t3_count, args.t2_count) + 500
    print(f"\nLoading MIMIC-IV notes (need {max_needed})...")
    notes = load_mimic(Path(args.mimic_notes), max_notes=max_needed)
    print(f"  Loaded {len(notes)} notes")
    
    if len(notes) < max_needed:
        print(f"  WARNING: Only {len(notes)} notes available. "
              f"T3/T2 counts may be reduced.")
        args.t3_count = min(args.t3_count, len(notes))
        args.t2_count = min(args.t2_count, len(notes))
    
    random.shuffle(notes)
    
    # ── Build T3 (same hardening as script 20) ──
    print(f"\nBuilding T3 ({args.t3_count})...")
    t3_samples = []
    n_minimal = int(args.t3_count * 0.10)  # 10% minimal PHI
    
    for i in range(min(args.t3_count, len(notes))):
        note = notes[i]
        patient = patients[i % len(patients)]
        stripped = strip_header(note["text"])
        
        if i < n_minimal:
            text, ent, _ = build_t3(stripped, patient, minimal=True)
            src = "mimic4_minimal"
        else:
            text, ent, _ = build_t3(stripped, patient, args.injection_rate)
            src = "mimic4_partial"
        
        if len(text) > args.max_chars:
            text = text[:args.max_chars]
        
        t3_samples.append({"text": text, "tier": 3, "source": src,
                           "note_id": note["note_id"], "phi_count": len(ent)})
    
    print(f"  Built {len(t3_samples)} T3 samples")
    
    # ── Build T2 ──
    print(f"Building T2 ({args.t2_count})...")
    t2_samples = []
    n_contam = int(args.t2_count * 0.10)
    
    for i in range(min(args.t2_count, len(notes))):
        note = notes[i]
        stripped = strip_header(note["text"])
        text = build_t2(stripped)
        
        if i < n_contam:
            text = contaminate_t2(text)
            src = "mimic4_contaminated"
        else:
            src = "mimic4_polluted"
        
        if len(text) > args.max_chars:
            text = text[:args.max_chars]
        
        t2_samples.append({"text": text, "tier": 2, "source": src,
                           "note_id": note["note_id"]})
    
    print(f"  Built {len(t2_samples)} T2 samples")
    
    # ── T0, T1 ──
    print(f"Building T0 ({args.t0_count}) and T1 ({args.t1_count})...")
    t0 = generate_t0_expanded(args.t0_count, use_hf=args.use_hf_datasets)
    t1 = generate_t1_expanded(args.t1_count)
    
    # ── Combine and split ──
    dataset = t3_samples + t2_samples + t0 + t1
    random.shuffle(dataset)
    total = len(dataset)
    
    print(f"\nTOTAL: {total:,}")
    for tier in range(4):
        c = sum(1 for s in dataset if s["tier"] == tier)
        print(f"  T{tier}: {c:,}")
    
    # Split with no matched-pair leakage
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
    
    # 60/10/30 split
    ng = len(groups)
    t_end = int(ng * 0.60)
    v_end = int(ng * 0.70)
    
    train, val, test = [], [], []
    for i, g in enumerate(groups):
        (train if i < t_end else val if i < v_end else test).extend(g)
    
    nug = len(ungrouped)
    train.extend(ungrouped[:int(nug*0.60)])
    val.extend(ungrouped[int(nug*0.60):int(nug*0.70)])
    test.extend(ungrouped[int(nug*0.70):])
    
    for s in [train, val, test]:
        random.shuffle(s)
    
    print(f"\nSplit: Train={len(train):,}, Val={len(val):,}, Test={len(test):,}")
    
    # CARES calibration math
    n_cal = int(len(test) * 0.30)
    n_eval = len(test) - n_cal
    cares_bound = 0.05 + 1.0 / (n_cal + 1)
    print(f"\nCARES calibration set: {n_cal:,} → bound = δ+1/(n+1) = {cares_bound:.6f}")
    print(f"CARES evaluation set: {n_eval:,}")
    
    # Save
    out_dir = ROOT / "data_30k"
    out_dir.mkdir(exist_ok=True)
    for name, split in [("train", train), ("val", val), ("test", test)]:
        with open(out_dir / f"{name}.json", "w") as f:
            json.dump(split, f, indent=2, default=str)
    print(f"\nSaved to {out_dir}/")
    
    # ── MIMIC-III Cross-Validation ──
    if args.mimic3_notes:
        print(f"\n{'='*70}")
        print("MIMIC-III CROSS-VALIDATION SET")
        print(f"{'='*70}")
        
        m3_notes = load_mimic3(Path(args.mimic3_notes), max_notes=8000)
        print(f"  Loaded {len(m3_notes)} MIMIC-III notes")
        
        if len(m3_notes) < 2000:
            print("  WARNING: Not enough MIMIC-III notes for cross-validation.")
        else:
            m3_t3 = []
            m3_t2 = []
            m3_patients = generate_patients(500)
            random.shuffle(m3_notes)
            
            n_m3_t3 = min(3500, len(m3_notes) // 2)
            n_m3_t2 = min(2500, len(m3_notes) // 2)
            
            for i, note in enumerate(m3_notes[:n_m3_t3]):
                patient = m3_patients[i % len(m3_patients)]
                stripped = strip_header(note["text"])
                text, ent, _ = build_t3(stripped, patient, args.injection_rate)
                if len(text) > args.max_chars:
                    text = text[:args.max_chars]
                m3_t3.append({"text": text, "tier": 3, "source": "mimic3_partial",
                              "note_id": note["note_id"]})
            
            for i, note in enumerate(m3_notes[n_m3_t3:n_m3_t3 + n_m3_t2]):
                stripped = strip_header(note["text"])
                text = build_t2(stripped)
                if len(text) > args.max_chars:
                    text = text[:args.max_chars]
                m3_t2.append({"text": text, "tier": 2, "source": "mimic3_polluted",
                              "note_id": note["note_id"]})
            
            # Add T0/T1 from templates
            m3_t0 = generate_t0_expanded(2500)
            m3_t1 = generate_t1_expanded(1500)
            
            m3_dataset = m3_t3 + m3_t2 + m3_t0 + m3_t1
            random.shuffle(m3_dataset)
            
            print(f"  MIMIC-III cross-validation set: {len(m3_dataset):,}")
            for tier in range(4):
                c = sum(1 for s in m3_dataset if s["tier"] == tier)
                print(f"    T{tier}: {c:,}")
            
            with open(out_dir / "mimic3_crossval.json", "w") as f:
                json.dump(m3_dataset, f, indent=2, default=str)
            print(f"  Saved: {out_dir}/mimic3_crossval.json")
    
    # ── Summary ──
    print(f"\n{'='*70}")
    print("NeurIPS DATASET READY")
    print(f"{'='*70}")
    print(f"  Primary dataset: {total:,} queries (MIMIC-IV)")
    print(f"  Train/Val/Test: {len(train):,}/{len(val):,}/{len(test):,}")
    print(f"  CARES bound: {cares_bound:.6f} (was 0.0508 with n=1,198)")
    if args.mimic3_notes:
        print(f"  Cross-validation: {len(m3_dataset):,} queries (MIMIC-III)")
    print(f"\nNext steps:")
    print(f"  1. python scripts/02_train_classifier.py  (retrain on 30K)")
    print(f"  2. python scripts/19b_cares_routing_fulltest.py  (re-evaluate)")
    print(f"  3. python scripts/21_pilot_barp_baselines.py  (run baselines)")


if __name__ == "__main__":
    main()
