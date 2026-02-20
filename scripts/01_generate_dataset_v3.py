#!/usr/bin/env python3
"""
01_generate_dataset_v3.py — PHI-GUARD Dataset with Realistic Clinical Text

DATA SOURCES:
  T3 (PHI):    Synthea FHIR clinical narratives annotated by obi/deid_bert_i2b2
  T2 (Pop):    De-identified population queries (templates + Synthea aggregate)
  T1 (Ops):    Operational queries (templates)
  T0 (Public): MedQA (real USMLE) + MTSamples transcriptions + clinical guidelines

PREREQUISITES:
  1. Generate Synthea patients first:
     git clone https://github.com/synthetichealth/synthea.git
     cd synthea
     ./run_synthea -p 300 --exporter.fhir.export=true
     (outputs to synthea/output/fhir/*.json)
  
  2. pip install transformers datasets torch pandas

USAGE:
  python scripts/01_generate_dataset_v3.py --synthea_dir /path/to/synthea/output/fhir
"""

import argparse
import json
import glob
import os
import random
import re
from pathlib import Path
from typing import List, Dict

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
with open(ROOT / "configs/config.yaml") as f:
    cfg = yaml.safe_load(f)

SEED = cfg["dataset"]["random_seed"]
random.seed(SEED); np.random.seed(SEED)
DATA = ROOT / cfg["dataset"]["output_dir"]; DATA.mkdir(exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# STAGE A: Extract clinical narratives from Synthea FHIR Bundles
# ═══════════════════════════════════════════════════════════════════════════
def extract_synthea_narratives(synthea_dir: str, max_patients: int = 300) -> List[Dict]:
    """
    Read Synthea FHIR Bundle JSON files and extract:
    - Patient demographics (name, DOB, address, phone, MRN)
    - Encounter narratives
    - Condition/medication/procedure descriptions
    - Compose into realistic clinical note snippets
    """
    fhir_files = sorted(glob.glob(os.path.join(synthea_dir, "*.json")))
    if not fhir_files:
        print(f"  WARNING: No FHIR JSON files found in {synthea_dir}")
        return []

    narratives = []
    for fpath in fhir_files[:max_patients]:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                bundle = json.load(f)
        except Exception as e:
            continue

        if bundle.get("resourceType") != "Bundle":
            continue

        # Extract patient info
        patient = {}
        conditions, medications, encounters, procedures = [], [], [], []

        for entry in bundle.get("entry", []):
            res = entry.get("resource", {})
            rtype = res.get("resourceType", "")

            if rtype == "Patient":
                names = res.get("name", [{}])
                if names:
                    given = " ".join(names[0].get("given", ["Unknown"]))
                    family = names[0].get("family", "Unknown")
                    patient["name"] = f"{given} {family}"
                    patient["family"] = family
                    patient["given"] = given
                patient["dob"] = res.get("birthDate", "1970-01-01")
                patient["gender"] = res.get("gender", "unknown")
                addrs = res.get("address", [{}])
                if addrs:
                    a = addrs[0]
                    lines = a.get("line", [""])
                    patient["address"] = f"{', '.join(lines)}, {a.get('city','')}, {a.get('state','')} {a.get('postalCode','')}"
                telecoms = res.get("telecom", [])
                for t in telecoms:
                    if t.get("system") == "phone":
                        patient["phone"] = t.get("value", "")
                patient["mrn"] = res.get("id", "")[:12]

            elif rtype == "Condition":
                code = res.get("code", {}).get("coding", [{}])[0]
                conditions.append(code.get("display", ""))

            elif rtype == "MedicationRequest":
                med = res.get("medicationCodeableConcept", {}).get("coding", [{}])[0]
                medications.append(med.get("display", ""))

            elif rtype == "Encounter":
                etype = res.get("type", [{}])[0].get("text", "") if res.get("type") else ""
                period = res.get("period", {})
                encounters.append({"type": etype, "start": period.get("start", "")})

            elif rtype == "Procedure":
                code = res.get("code", {}).get("coding", [{}])[0]
                procedures.append(code.get("display", ""))

        if not patient.get("name"):
            continue

        # Compose clinical note snippets from extracted data
        p = patient
        conds = [c for c in conditions if c][:5]
        meds = [m for m in medications if m][:5]
        procs = [pr for pr in procedures if pr][:3]
        encs = [e for e in encounters if e.get("type")][:3]

        # Generate multiple note styles per patient
        templates = []

        # Discharge summary style
        if conds:
            dx_list = ", ".join(conds[:3])
            med_list = ", ".join(meds[:3]) if meds else "none documented"
            templates.append(
                f"DISCHARGE SUMMARY\nPatient: {p['name']}, DOB: {p['dob']}, MRN: {p['mrn']}\n"
                f"Gender: {p['gender'].title()}\nAddress: {p.get('address','N/A')}\n"
                f"Diagnoses: {dx_list}\nMedications at discharge: {med_list}\n"
                f"Follow-up: Return to clinic in 2 weeks."
            )

        # Progress note style
        if conds and meds:
            templates.append(
                f"Progress Note — {p['name']} ({p['mrn']})\n"
                f"Date: {encs[0]['start'][:10] if encs else '2025-01-15'}\n"
                f"Chief Complaint: Follow-up for {conds[0]}.\n"
                f"Current medications include {', '.join(meds[:2])}.\n"
                f"Assessment: {conds[0]} — stable on current regimen.\n"
                f"Plan: Continue {meds[0]}, recheck labs in 3 months."
            )

        # Referral style
        if procs and conds:
            templates.append(
                f"Referral: {p['name']} (DOB {p['dob']}) referred for {procs[0]} "
                f"evaluation. PMH includes {conds[0]}. Contact: {p.get('phone','N/A')}."
            )

        # Brief query style (what a clinician might type)
        templates.append(f"What are the latest labs for {p['name']}, MRN {p['mrn']}?")
        templates.append(f"Pull up {p['family']}'s medication list — DOB {p['dob']}.")
        if conds:
            templates.append(f"Is {p['name']} still on {meds[0] if meds else 'current meds'} for {conds[0]}?")
        if p.get("phone"):
            templates.append(f"{p['name']}'s family called about follow-up. Callback: {p['phone']}.")
        if encs:
            templates.append(
                f"Checking on the patient seen {encs[0]['start'][:10]} for {encs[0]['type']} — "
                f"I think it was {p['name']} from {p.get('address','').split(',')[1].strip() if ',' in p.get('address','') else 'unknown'}."
            )

        for txt in templates:
            if len(txt) > 30:
                narratives.append({
                    "text": txt,
                    "patient": p,
                    "conditions": conds,
                    "medications": meds,
                })

    return narratives


# ═══════════════════════════════════════════════════════════════════════════
# STAGE B: Annotate PHI using obi/deid_bert_i2b2
# ═══════════════════════════════════════════════════════════════════════════
def annotate_phi(narratives: List[Dict], batch_size: int = 16) -> List[Dict]:
    """Run obi/deid_bert_i2b2 NER on narratives to get token-level PHI labels."""
    from transformers import pipeline

    print("  Loading obi/deid_bert_i2b2 model...")
    try:
        ner = pipeline("ner", model="obi/deid_bert_i2b2", aggregation_strategy="simple",
                        device=-1)  # CPU
    except Exception as e:
        print(f"  WARNING: Could not load obi/deid_bert_i2b2 ({e})")
        print("  Falling back to rule-based PHI annotation...")
        return annotate_phi_fallback(narratives)

    annotated = []
    for i in range(0, len(narratives), batch_size):
        batch = narratives[i:i+batch_size]
        texts = [n["text"][:512] for n in batch]  # model max length
        try:
            results = ner(texts)
        except Exception:
            results = [[] for _ in texts]

        for n, ents in zip(batch, results):
            phi_entities = []
            for ent in ents:
                phi_entities.append({
                    "text": ent.get("word", ""),
                    "label": ent.get("entity_group", ""),
                    "start": ent.get("start", 0),
                    "end": ent.get("end", 0),
                    "score": float(ent.get("score", 0)),
                })
            n["phi_annotations"] = phi_entities
            n["phi_count"] = len(phi_entities)
            annotated.append(n)

        if (i // batch_size) % 10 == 0:
            print(f"  Annotated {min(i+batch_size, len(narratives))}/{len(narratives)}")

    return annotated


def annotate_phi_fallback(narratives: List[Dict]) -> List[Dict]:
    """Rule-based fallback using known patient data from FHIR extraction."""
    for n in narratives:
        phi_entities = []
        p = n.get("patient", {})
        text = n["text"]
        for field in ["name", "dob", "mrn", "phone", "address"]:
            val = str(p.get(field, ""))
            if val and val in text:
                idx = text.find(val)
                phi_entities.append({
                    "text": val, "label": field.upper(),
                    "start": idx, "end": idx + len(val), "score": 1.0
                })
        n["phi_annotations"] = phi_entities
        n["phi_count"] = len(phi_entities)
    return narratives


# ═══════════════════════════════════════════════════════════════════════════
# STAGE C: Load T0 from MedQA + MTSamples
# ═══════════════════════════════════════════════════════════════════════════
def load_t0_sources() -> List[Dict]:
    """Load MedQA and MTSamples for T0 (public medical knowledge)."""
    samples = []

    # MedQA
    try:
        from datasets import load_dataset
        print("  Loading MedQA from HuggingFace...")
        ds = load_dataset("openlifescienceai/medqa", split="test")
        for x in list(ds)[:600]:
            q = x.get("question", "")
            if q and len(q) > 30:
                samples.append({"text": q, "source": "MedQA", "difficulty": "medqa"})
        print(f"    Got {len(samples)} MedQA questions")
    except Exception as e:
        print(f"    MedQA skipped: {e}")

    # MTSamples (via HuggingFace or Kaggle)
    try:
        from datasets import load_dataset
        print("  Loading MTSamples...")
        ds = load_dataset("harishnair04/mtsamples", split="train")
        count = 0
        for x in list(ds)[:800]:
            desc = x.get("description", "") or x.get("transcription", "") or ""
            if desc and len(desc) > 50:
                # Extract first 2-3 sentences for clinical vignette
                sents = desc.split(".")[:3]
                vignette = ".".join(sents).strip() + "."
                if len(vignette) > 40:
                    samples.append({"text": vignette, "source": "MTSamples",
                                    "difficulty": "hard"})
                    count += 1
        print(f"    Got {count} MTSamples vignettes")
    except Exception as e:
        print(f"    MTSamples skipped: {e}")

    # Clinical guideline questions (templates — these are fine for T0)
    guidelines = [
        "What are the ADA HbA1c targets for Type 2 Diabetes?",
        "Mechanism of action of metformin?",
        "CHADS2-VASc scoring system for atrial fibrillation?",
        "ACE inhibitor contraindications?",
        "First-line treatment for essential hypertension?",
        "DSM-5 criteria for major depressive disorder?",
        "CKD stages based on GFR?",
        "USPSTF colorectal cancer screening recommendations?",
        "Statin side effects?",
        "GOLD COPD classification?",
        "Warfarin dosing and INR targets?",
        "SGLT2 inhibitor mechanism of action?",
        "DKA clinical features and management?",
        "Ottawa ankle rules for imaging decision?",
        "Wells criteria for pulmonary embolism?",
        "CURB-65 score for community-acquired pneumonia?",
        "Anaphylaxis protocol?",
        "TNM staging for breast cancer?",
        "VTE risk factors and prophylaxis?",
        "Framingham Risk Score calculation?",
        "CAGE questionnaire for alcohol screening?",
        "Metabolic syndrome diagnostic criteria?",
        "Number needed to treat — definition and interpretation?",
        "Sepsis and qSOFA criteria?",
        "Child-Pugh score for liver disease severity?",
        "Inpatient DVT prophylaxis guidelines?",
        "ASCVD risk calculator — when to initiate statins?",
        "GINA step-up approach for asthma management?",
        "Brugada criteria for differentiating SVT from VT?",
        "Winter formula for mixed acid-base disorders?",
    ]
    for g in guidelines:
        samples.append({"text": g, "source": "template", "difficulty": "standard"})

    return samples


# ═══════════════════════════════════════════════════════════════════════════
# STAGE D: T2/T1 templates (kept from v2 — these are inherently distinct)
# ═══════════════════════════════════════════════════════════════════════════
T2_TEMPLATES = [
    "30-day readmission rate for {dx} patients?",
    "Among patients with HbA1c > 9, what percentage prescribed {med}?",
    "Age distribution of {dx} diagnoses in 2024?",
    "Average length of stay for {proc} patients?",
    "De-identified {med} adherence rates across population?",
    "{dx} mortality rate by age group?",
    "Population-level {lab} trend over 3 years?",
    "Our {proc} complication rate vs CMS benchmarks?",
    "Risk score distribution for {dx} cohort?",
    "Kaplan-Meier survival curve for {dx} by treatment arm?",
    "In the cohort of 847 patients aged 45-65 with {dx}, median {lab} at 6 months?",
    "Among 234 patients discharged on {med}, 30-day readmission count?",
    "{dx} prevalence by Springfield, Riverside, and Salem regions?",
    "Cost per episode: {dx} with {proc} vs medical management?",
    "{dx} outcomes by race/ethnicity from de-identified quality database?",
]

T1_TEMPLATES = [
    "Current bed occupancy for cardiac ICU?",
    "Night shift nurse staffing for ED tomorrow?",
    "Average door-to-doctor time in ED this month?",
    "Monthly HCAHPS quality scorecard?",
    "MRI scanner utilization rate this month?",
    "Daily census by service line?",
    "O-negative blood bank inventory level?",
    "Code blue events on medical floor this month?",
    "CLABSI rate dashboard?",
    "ED volume versus same period last year?",
    "Our {dx} readmission penalty risk for CMS reporting?",
    "{proc} cancellations due to staffing this month?",
    "Sepsis bundle compliance in ED — hitting 3-hour target?",
    "{med} medication errors reported this month?",
    "Antibiotic days of therapy per 1000 patient-days?",
]

DX = ["Type 2 Diabetes","Hypertension","COPD","CHF","Atrial Fibrillation","CKD Stage 3",
      "Pneumonia","Sepsis","DVT","Acute Pancreatitis","Pulmonary Embolism","Asthma"]
MEDS = ["Metformin","Lisinopril","Atorvastatin","Metoprolol","Warfarin","Apixaban",
        "Furosemide","Sertraline","Omeprazole","Insulin Glargine"]
PROCS = ["cardiac catheterization","colonoscopy","CT abdomen","MRI brain","echocardiogram",
         "stress test","knee arthroscopy","bronchoscopy"]
LABS = ["HbA1c","Creatinine","TSH","LDL","INR","BNP","Troponin","WBC"]

def gen_tier_samples(templates, tier, label, n):
    samples = []
    for _ in range(n):
        t = random.choice(templates)
        txt = t.format(dx=random.choice(DX), med=random.choice(MEDS),
                       proc=random.choice(PROCS), lab=random.choice(LABS))
        samples.append({"text": txt, "tier": tier, "tier_label": label,
                        "is_compound": False, "phi_present": False, "phi_entities": [],
                        "difficulty": "standard", "source": "template"})
    return samples


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthea_dir", type=str, required=True,
                        help="Path to synthea/output/fhir directory")
    parser.add_argument("--skip_model", action="store_true",
                        help="Skip obi/deid_bert model, use rule-based annotation")
    args = parser.parse_args()

    print("="*60 + "\nPHI-GUARD Dataset v3 (Synthea FHIR + deid_bert + MedQA)\n" + "="*60)

    # ── T3: Synthea FHIR narratives ───────────────────────────────────────
    print("\n[T3] Extracting Synthea FHIR narratives...")
    narratives = extract_synthea_narratives(args.synthea_dir, max_patients=300)
    print(f"  Extracted {len(narratives)} clinical text snippets")

    if not args.skip_model:
        print("\n[T3] Annotating PHI with obi/deid_bert_i2b2...")
        narratives = annotate_phi(narratives)
    else:
        print("\n[T3] Using rule-based PHI annotation (--skip_model)")
        narratives = annotate_phi_fallback(narratives)

    # Convert to dataset format
    t3_samples = []
    for n in narratives:
        phi_ents = [e["text"] for e in n.get("phi_annotations", []) if e.get("score", 0) > 0.5]
        t3_samples.append({
            "text": n["text"],
            "tier": 3, "tier_label": "T3_Restricted",
            "is_compound": False,
            "phi_present": len(phi_ents) > 0,
            "phi_entities": phi_ents,
            "phi_annotations": n.get("phi_annotations", []),
            "difficulty": "synthea_fhir",
            "source": "synthea",
        })
    print(f"  T3 samples: {len(t3_samples)} ({sum(s['phi_present'] for s in t3_samples)} with detected PHI)")

    # ── T0: MedQA + MTSamples ────────────────────────────────────────────
    print("\n[T0] Loading MedQA + MTSamples...")
    t0_raw = load_t0_sources()
    t0_samples = [{"text": s["text"], "tier": 0, "tier_label": "T0_Public",
                    "is_compound": False, "phi_present": False, "phi_entities": [],
                    "difficulty": s.get("difficulty", "standard"),
                    "source": s.get("source", "template")}
                   for s in t0_raw]
    print(f"  T0 samples: {len(t0_samples)}")

    # ── T2/T1: Templates ─────────────────────────────────────────────────
    print("\n[T2/T1] Generating operational/population queries...")
    t2_samples = gen_tier_samples(T2_TEMPLATES, 2, "T2_Limited", 1500)
    t1_samples = gen_tier_samples(T1_TEMPLATES, 1, "T1_Internal", 1100)

    # ── Combine and split ─────────────────────────────────────────────────
    all_s = t3_samples + t2_samples + t1_samples + t0_samples
    random.shuffle(all_s)
    for i, s in enumerate(all_s):
        s["id"] = f"q_{i:05d}"
        s["token_count"] = max(20, int(len(s["text"].split()) * 1.3 + np.random.normal(0, 5)))
        # Remove heavy annotation field for training (keep tier/entities)
        s.pop("phi_annotations", None)

    # Split: 50/10/40
    n = len(all_s)
    te = int(n * 0.50); ve = te + int(n * 0.10)
    splits = {"train": all_s[:te], "val": all_s[te:ve], "test": all_s[ve:]}

    for name, data in splits.items():
        p = DATA / f"{name}.json"
        with open(p, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  {name}: {len(data):>5} -> {p}")

    print(f"\nDistribution:")
    for t in range(4):
        ct = sum(1 for s in all_s if s["tier"] == t)
        src = {}
        for s in all_s:
            if s["tier"] == t:
                src[s.get("source", "?")] = src.get(s.get("source", "?"), 0) + 1
        print(f"  T{t}: {ct:>5} | sources: {dict(sorted(src.items()))}")
    print(f"Total: {len(all_s)} | Test: {len(splits['test'])}")
    print(f"PHI detected in T3: {sum(s['phi_present'] for s in t3_samples)}/{len(t3_samples)}")

if __name__ == "__main__":
    main()
