#!/usr/bin/env python3
"""01 — Generate PHI-GUARD dataset v2 with adversarial/hard examples.

Changes from v1:
  - 20% hard/adversarial examples per tier (blur boundaries)
  - T0 has clinical vignettes (looks like patient data but isn't)
  - T2 has specific numbers/locations (looks like PHI but is aggregate)
  - T1 has clinical terms (looks like T0/T2)
  - Scales to ~7000 total -> ~2800 test set (matches paper)
  - Token count noise for realistic cost variance
"""

import json, random
from pathlib import Path
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
with open(ROOT / "configs/config.yaml") as f:
    cfg = yaml.safe_load(f)

SEED = cfg["dataset"]["random_seed"]
random.seed(SEED); np.random.seed(SEED)
DATA = ROOT / cfg["dataset"]["output_dir"]; DATA.mkdir(exist_ok=True)

# ── PHI building blocks ───────────────────────────────────────────────────
FIRST = ["James","Mary","Robert","Patricia","John","Jennifer","Michael","Linda",
    "David","Elizabeth","William","Barbara","Richard","Susan","Joseph","Jessica",
    "Thomas","Sarah","Christopher","Karen","Daniel","Lisa","Matthew","Nancy",
    "Alejandro","Priya","Wei","Fatima","Dmitri","Aisha","Hiroshi","Mohammed",
    "Raj","Carmen","Oluwaseun","Mei","Anthony","Mark","Donald","Steven",
    "Ashley","Dorothy","Kimberly","Emily","Donna","Michelle","Carol","Amanda"]
LAST = ["Smith","Johnson","Williams","Brown","Jones","Garcia","Miller","Davis",
    "Rodriguez","Martinez","Hernandez","Lopez","Wilson","Anderson","Thomas",
    "Taylor","Moore","Jackson","Martin","Lee","Perez","Nguyen","Chen","Patel",
    "Kim","Tanaka","Singh","O'Brien","Kowalski","Torres","Rivera","Campbell"]
STREETS = ["Main St","Oak Ave","Elm Dr","Cedar Ln","Maple Rd","Pine St",
    "Washington Blvd","Park Ave","Lake Dr","River Rd","Hill St","Forest Ave"]
CITIES = ["Springfield","Riverside","Fairview","Georgetown","Salem","Franklin",
    "Clinton","Greenville","Bristol","Oakland","Burlington","Madison"]
STATES = ["MA","CT","NY","PA","NJ","OH","IL","CA","TX","FL","WA","CO","VA","GA"]
DX = ["Type 2 Diabetes Mellitus","Essential Hypertension","Major Depressive Disorder",
    "COPD","Congestive Heart Failure","Atrial Fibrillation","CKD Stage 3",
    "Osteoarthritis","GERD","Asthma","Coronary Artery Disease","Hypothyroidism",
    "Hyperlipidemia","Anxiety Disorder","Obesity","Iron Deficiency Anemia",
    "Rheumatoid Arthritis","Migraine","Sleep Apnea","Peripheral Neuropathy",
    "Acute Pancreatitis","Pulmonary Embolism","Cellulitis","Pneumonia","Sepsis"]
MEDS = ["Metformin 1000mg","Lisinopril 20mg","Atorvastatin 40mg","Amlodipine 5mg",
    "Omeprazole 20mg","Metoprolol 50mg","Levothyroxine 75mcg","Sertraline 100mg",
    "Gabapentin 300mg","Losartan 50mg","Furosemide 40mg","Warfarin 5mg",
    "Albuterol inhaler","Insulin Glargine 20 units","Apixaban 5mg"]
PROCS = ["cardiac catheterization","colonoscopy","upper endoscopy","CT abdomen/pelvis",
    "MRI brain","chest X-ray","echocardiogram","stress test","bronchoscopy",
    "knee arthroscopy","thyroid ultrasound","bone density scan","sleep study"]
LABS = [("HbA1c","7.2%"),("Creatinine","1.8 mg/dL"),("TSH","6.2 mIU/L"),
    ("LDL","162 mg/dL"),("INR","2.8"),("Hemoglobin","9.2 g/dL"),
    ("Potassium","5.8 mEq/L"),("BNP","890 pg/mL"),("ALT","78 U/L"),
    ("Glucose","186 mg/dL"),("Troponin","0.42 ng/mL"),("WBC","14.2 K/uL")]
UNITS = ["cardiac ICU","medical-surgical","ED","NICU","oncology","progressive care",
    "burn unit","transplant","rehab","psych","labor and delivery"]

def p():
    f,l = random.choice(FIRST), random.choice(LAST)
    lb = random.choice(LABS)
    return dict(name=f"{f} {l}", first=f, last=l,
        mrn=f"MRN-{random.randint(10000,99999)}",
        dob=f"{random.randint(1940,2005)}-{random.randint(1,12):02d}-{random.randint(1,28):02d}",
        age=random.randint(22,88),
        ssn=f"{random.randint(100,999)}-{random.randint(10,99)}-{random.randint(1000,9999)}",
        phone=f"({random.randint(200,999)}) {random.randint(200,999)}-{random.randint(1000,9999)}",
        addr=f"{random.randint(1,9999)} {random.choice(STREETS)}, {random.choice(CITIES)}, {random.choice(STATES)}",
        dos=f"2025-{random.randint(1,12):02d}-{random.randint(1,28):02d}",
        provider=f"Dr. {random.choice(FIRST)} {random.choice(LAST)}",
        dx=random.choice(DX), med=random.choice(MEDS), proc=random.choice(PROCS),
        lab=lb[0], labv=lb[1], unit=random.choice(UNITS),
        bed=f"{random.choice('ABCDEFGH')}{random.randint(1,30)}")

# ── Templates per tier ─────────────────────────────────────────────────────
# T3: Direct PHI
T3S = [
    "Patient {name} ({mrn}), DOB {dob}, admitted {dos} with {dx}. {provider} ordered {med}.",
    "DISCHARGE: {name}, {age}yo, MRN {mrn}, discharged {dos} after {dx}. Follow-up {provider}.",
    "{provider} performed {proc} on {name} (DOB: {dob}) on {dos}. {lab}: {labv}.",
    "Progress Note: {name} ({mrn}) seen for {dx}. {lab}: {labv}. Adjusted {med}.",
    "Retrieve labs for {name}, MRN {mrn}.",
    "What meds is {name} (DOB: {dob}) currently taking?",
    "Show discharge summary for {mrn} from {dos}.",
    "{name}'s daughter called re {proc} on {dos}. Phone: {phone}.",
    "Chart review: {last}, {first} - SSN ending {ssn}. Hx of {dx}.",
    "SOAP: {name} ({mrn}), worsening {dx}. Poor adherence to {med}. {lab} at {labv}.",
    "Referral from {provider}: {name} (DOB {dob}) for {proc}. {lab}: {labv}.",
    "ED Triage: {name}, {age}yo, {mrn}, CC chest pain. Hx: {dx}. Meds: {med}.",
    "Transfer note: {name} from {unit} bed {bed} to ICU. {dx} worsening.",
    "Consent: {name} ({mrn}): {proc} scheduled {dos} by {provider}.",
    "Nursing: {name} bed {bed} reports pain 8/10. {provider} notified at {phone}.",
    "Lab callback: {name}'s {lab} critical at {labv}. {provider} contacted.",
    "Home health: {name}, {addr}, follow-up for {dx}. PCP: {provider}.",
    "Has {mrn} been compliant with {med}?",
    "{lab} trend for {name} over 6 months?",
    "Pull up {name}'s {proc} results from last visit.",
]
T3H = [  # Hard: implicit/embedded PHI
    "The patient with {dx} seen by {provider} on {dos} — I believe it was {name} — needs follow-up {proc}.",
    "Checking on the {age}-year-old in {unit} bed {bed} who came through ED with {dx}. Family: {phone}.",
    "We discussed {dx} options. The patient from {addr} prefers conservative over {proc}.",
    "During rounds {provider} mentioned the {dx} case — pt {last} — has {lab} trending at {labv}.",
    "Can you check if the {dx} patient admitted {dos} has insurance? Name is {name}.",
    "The elderly patient with {dx}, {lab} of {labv}, lives alone at {addr}, needs social work.",
    "Pt dx {dx} adm {dos} {name} on {med} q12h {lab} {labv} f/u {provider}",
    "Spoke with {first} about {dx} prognosis. Will call back at {phone}.",
    "The {dx} case from {dos} — records for {first} {last}, DOB around {dob}.",
    "Interesting case: {age}yo from {addr}, {lab} was {labv} on admission, now on {med}.",
]
T3C = [  # Compound
    "What are ADA A1C targets, and how does {name} ({mrn}) compare with {lab} of {labv}?",
    "Compare standard {med} dosing with what was prescribed for {name} on {dos}.",
    "Guidelines for {dx} — and check if {name}'s plan follows them.",
    "Normal range for {lab}? Where does {mrn}'s {labv} fall?",
    "Summarize {med} side effects and flag any reported by {name}.",
    "Review {dx} pathway and determine if {name} meets discharge criteria.",
    "Evidence for {proc} in {dx}? Also pull {name}'s imaging from {dos}.",
    "Standard {med} taper — and what dose is {name} ({mrn}) currently on?",
]

# T2: De-identified population
T2S = [
    "30-day readmission rate for {dx} patients?",
    "Among patients with HbA1c > 9, what % prescribed {med}?",
    "Age distribution of {dx} patients in 2024?",
    "Average LOS for {proc} patients?",
    "{lab} improvement across treatment cohorts for {dx}?",
    "ED visit rate for {dx} cohort past year?",
    "{dx} mortality by age group?",
    "De-identified {med} adherence report.",
    "% of diabetic population with {lab} below target?",
    "Aggregate outcomes: {proc} vs conservative for {dx}?",
    "Risk-stratify {dx} population by comorbidity.",
    "Average time from {dx} to first {proc}?",
    "Readmission: {med} vs alternatives for {dx}?",
    "{dx} demographics by zip code?",
    "Population {lab} trend over 3 years?",
    "Our {proc} complication rate vs CMS benchmarks?",
    "{lab} normalization rate within 6mo of {med}?",
    "Risk score distribution for {dx}?",
    "KM curve for {dx} by treatment arm?",
    "{lab} vs 90-day outcomes in {dx} cohort?",
]
T2H = [  # Hard: specific numbers/locations that look PHI-adjacent
    "In the cohort of 847 patients aged 45-65 with {dx}, median {lab} at 6 months?",
    "For patients in {unit} Jan-Mar 2025, average {lab}?",
    "Among 234 patients discharged on {med}, 30-day readmission count?",
    "{dx} prevalence by Springfield, Riverside, and Salem regions?",
    "Compare {lab} distributions between {med} cohorts — de-identified only.",
    "Patients over 65 with {dx} who got {proc} last quarter? Aggregate counts.",
    "% of {dx} patients in {unit} achieving target {lab} within 90 days?",
    "De-identified survival analysis for {dx} by initial {lab} quartile.",
    "Heatmap of {dx} incidence by zip code from de-identified registry.",
    "Cost per episode: {dx} with {proc} vs medical management?",
    "Across {unit} admissions 2024, median time to {lab} normalization?",
    "{dx} outcomes by race/ethnicity from quality database.",
]

# T1: Operational
T1S = [
    "Current bed occupancy for {unit}?",
    "Night shift nurse count for ED tomorrow?",
    "ED boarding time this quarter?",
    "Surgical backlog for orthopedics?",
    "Weekly staffing report for {unit}?",
    "Door-to-doctor time in ED this month?",
    "ORs scheduled for maintenance next week?",
    "Hospitalist productivity dashboard?",
    "Central lab turnaround time?",
    "Monthly HCAHPS scorecard?",
    "Available beds in {unit}?",
    "Radiology order wait time?",
    "Q4 budget variance by department?",
    "Pending physician credentialing apps?",
    "MRI utilization rate this month?",
    "Daily census by service line?",
    "Discharge lounge patient count?",
    "O-negative blood bank inventory?",
    "Surgical supply costs vs last quarter?",
    "Patient satisfaction by unit?",
    "Code blue events on medical floor this month?",
    "P&T formulary update?",
    "OR booking to procedure average time?",
    "CLABSI rate dashboard?",
    "ED volume vs same period last year?",
]
T1H = [  # Hard: clinical terms that look like T0/T2
    "Our {dx} readmission penalty risk for next CMS period?",
    "{proc} cancellations due to staffing this month?",
    "{med} utilization report — meeting formulary targets?",
    "Average cost of {dx} admission vs state benchmark?",
    "{proc} conversion rate to open surgery in {unit} this quarter?",
    "{lab} STAT turnaround SLA compliance?",
    "Sepsis bundle compliance in ED — hitting 3-hour target?",
    "{dx} case mix index vs budget?",
    "{unit} nurse-to-patient ratio past 30 days?",
    "{med} medication errors reported this month?",
    "{proc} volume by surgeon — anonymized productivity?",
    "Antibiotic days of therapy per 1000 patient-days for {unit}?",
]

# T0: General knowledge
T0S = [
    "ADA HbA1c targets for Type 2 Diabetes?",
    "Mechanism of action of metformin?",
    "CHADS2-VASc scoring system?",
    "ACE inhibitor contraindications?",
    "CHF pathophysiology?",
    "First-line treatment for hypertension?",
    "DSM-5 criteria for MDD?",
    "CKD stages by GFR?",
    "USPSTF colorectal cancer screening?",
    "Systolic vs diastolic heart failure?",
    "Statin side effects?",
    "GOLD COPD classification?",
    "Warfarin dosing and INR targets?",
    "AHA BP management guidelines?",
    "SGLT2 inhibitor mechanism?",
    "DKA clinical features?",
    "Ottawa ankle rules?",
    "Wells criteria for PE?",
    "CABG indications?",
    "CURB-65 for CAP?",
    "Adult 65+ immunizations?",
    "AKI pathophysiology?",
    "Anaphylaxis protocol?",
    "Opioid prescribing guidelines?",
    "TNM staging breast cancer?",
    "VTE risk factors?",
    "New-onset seizure workup?",
    "Antibiotic stewardship guidelines?",
    "Framingham Risk Score?",
    "First-line GAD treatment?",
    "CAGE questionnaire?",
    "Stable angina management?",
    "Metabolic syndrome criteria?",
    "NNT concept?",
    "Perioperative beta-blockers?",
    "Sepsis and qSOFA?",
    "Thyroid nodule evaluation?",
    "NICE T2DM guidelines?",
    "Child-Pugh score?",
    "Inpatient DVT prophylaxis?",
]
T0H = [  # Hard: clinical vignettes (looks like patient data but is textbook)
    "A 65-year-old male presents with chest pain and diaphoresis. Differential diagnosis?",
    "Patient has creatinine 2.4 and GFR 28. What CKD stage?",
    "Recommended HbA1c target for a 72-year-old with multiple comorbidities?",
    "Management algorithm for acute PE presentation?",
    "Patient on warfarin, INR 5.2, no bleeding. What to do?",
    "SIRS vs sepsis vs septic shock diagnostic criteria?",
    "MELD score calculation and components?",
    "Evidence for aspirin in primary CV prevention?",
    "GINA step-up approach for asthma?",
    "CKD patient with K+ 6.1 — acute management?",
    "Thrombolysis contraindications in STEMI?",
    "ASCVD calculator — when to start statins?",
    "Triple therapy duration for H. pylori?",
    "Brugada criteria for SVT vs VT?",
    "Immunocompromised adult vaccination schedule?",
    "Pediatric head trauma imaging decision rules?",
    "Tumor lysis syndrome pathophysiology and management?",
    "Anticoagulation for mechanical heart valves?",
    "Winter formula for mixed acid-base disorders?",
    "Therapeutic hypothermia post-cardiac arrest — indications and risks?",
]

# ── Build dataset ──────────────────────────────────────────────────────────
def make(std, hard, tier, label, n, comp=None, comp_frac=0.10):
    samples = []
    nc = int(n * comp_frac) if comp else 0
    nh = int(n * 0.20)
    ns = n - nc - nh
    is_phi = tier == 3

    def ents(d):
        return ([d["name"],d["mrn"],d["dob"],d["ssn"],d["phone"],d["addr"],
                 d["provider"],d["dos"],str(d["age"])] if is_phi else [])

    for _ in range(ns):
        d = p()
        samples.append(dict(text=random.choice(std).format(**d), tier=tier,
            tier_label=label, is_compound=False, phi_present=is_phi,
            phi_entities=ents(d), difficulty="standard"))
    for _ in range(nh):
        d = p()
        samples.append(dict(text=random.choice(hard).format(**d), tier=tier,
            tier_label=label, is_compound=False, phi_present=is_phi,
            phi_entities=ents(d), difficulty="hard"))
    for _ in range(nc):
        d = p()
        samples.append(dict(text=random.choice(comp).format(**d), tier=tier,
            tier_label=label, is_compound=True, phi_present=is_phi,
            phi_entities=[d["name"],d["mrn"],d["dob"],d["provider"]] if is_phi else [],
            difficulty="compound"))
    return samples

def main():
    print("="*60+"\nPHI-GUARD Dataset Generator v2 (adversarial)\n"+"="*60)
    s3 = make(T3S, T3H, 3, "T3_Restricted", 2400, T3C)
    s2 = make(T2S, T2H, 2, "T2_Limited", 1500)
    s1 = make(T1S, T1H, 1, "T1_Internal", 1100)
    s0 = make(T0S, T0H, 0, "T0_Public", 2000)

    try:
        from datasets import load_dataset
        print("Downloading MedQA...")
        ds = load_dataset("openlifescienceai/medqa", split="test")
        mq = [dict(text=x.get("question",""), tier=0, tier_label="T0_Public",
                    is_compound=False, phi_present=False, phi_entities=[],
                    difficulty="medqa") for x in list(ds)[:500] if len(x.get("question",""))>30]
        if mq: s0 = s0[:2000-len(mq)] + mq; print(f"  Added {len(mq)} MedQA")
    except Exception as e:
        print(f"  MedQA skipped ({e})")

    all_s = s3+s2+s1+s0; random.shuffle(all_s)
    for i,s in enumerate(all_s):
        s["id"]=f"q_{i:05d}"
        s["token_count"]=max(20,int(len(s["text"].split())*1.3+np.random.normal(0,5)))

    # 50/10/40 split -> ~2800 test
    n=len(all_s); te=int(n*0.50); ve=te+int(n*0.10)
    splits={"train":all_s[:te],"val":all_s[te:ve],"test":all_s[ve:]}
    for nm,data in splits.items():
        with open(DATA/f"{nm}.json","w") as f: json.dump(data,f,indent=2)
        print(f"  {nm}: {len(data):>5} -> {DATA/f'{nm}.json'}")

    print(f"\nDistribution:")
    for t in range(4):
        ct=sum(1 for s in all_s if s["tier"]==t)
        hd=sum(1 for s in all_s if s["tier"]==t and s.get("difficulty")=="hard")
        cm=sum(1 for s in all_s if s["tier"]==t and s["is_compound"])
        print(f"  T{t}: {ct:>5} ({hd:>4} hard, {cm:>4} compound)")
    print(f"Total: {len(all_s)} | Test: {len(splits['test'])} | PHI: {sum(s['phi_present'] for s in all_s)}")

if __name__=="__main__":
    main()
