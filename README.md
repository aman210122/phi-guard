# PHI-GUARD: Compliance-Aware LLM Routing for Healthcare with Distribution-Free Safety Guarantees

[![arXiv](https://img.shields.io/badge/arXiv-2602.XXXXX-b31b1b.svg)](https://arxiv.org/abs/2602.XXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![MIMIC-IV](https://img.shields.io/badge/data-MIMIC--IV-green.svg)](https://physionet.org/content/mimic-iv-note/2.2/)

> **Healthcare LLM routing where violations are forbidden, not penalized.**

PHI-GUARD is a compliance-aware routing framework that enforces hard data-residency constraints when routing clinical queries across cloud and on-premises LLM platforms. Unlike existing routers (RouteLLM, FrugalGPT, PILOT, BaRP) that optimize cost vs. quality, PHI-GUARD ensures that queries containing Protected Health Information (PHI) are **never** sent to unauthorized platforms — with provable, distribution-free safety guarantees.

---

## Key Results

| Strategy | Violations | Violation % | Cloud % | Guarantee |
|----------|-----------|-------------|---------|-----------|
| Greedy | 6,771 | 75.3% | 100.0% | None |
| PILOT-style | 83 | 0.92% | 65.0% | None |
| BaRP-style | 82 | 0.91% | 64.7% | None |
| StaticILP | 83 | 0.92% | 65.2% | None |
| SafeTS (ε=0.02) | 7 | 0.078% | 63.1% | Invalid* |
| **CARES (δ=0.005)** | **0** | **0.000%** | **35.6%** | **Valid ✓** |
| SecureDefault | 0 | 0.000% | 0.0% | Trivial |

*\*SafeTS guarantee requires classifier calibration; violated in practice (MCE=0.217).*

**CARES is the only strategy achieving zero violations with a valid distribution-free bound** (δ + 1/(n+1) = 0.005), verified across:
- 30,000 MIMIC-IV clinical queries + 10,000-query MIMIC-III cross-validation
- 3 classifier architectures (DistilBERT → TF-IDF → Regex heuristic)
- 7 drift intensities (0.0 → 0.6) without recalibration
- Real API endpoints (GPT-4o-mini + Llama-3.1-8B)

---

## The Problem

Healthcare organizations deploying LLMs face a regulatory tradeoff:

```
Cloud API  →  Cheaper, faster       →  But PHI exposure = HIPAA violation ($1.5M+ per incident)
On-Premises →  Safe, compliant      →  But 2-3× more expensive, higher latency
```

**Existing routers don't help.** RouteLLM, FrugalGPT, PILOT, and BaRP optimize cost against quality — they treat every query identically for data sensitivity. A drug interaction question and a query containing a patient's medical record number get the same routing logic.

**The core difficulty:** Query sensitivity is uncertain at routing time. Classifiers estimate whether a query contains PHI, but they err — and can be *confidently wrong*. In our experiments, 8 PHI queries are misclassified with confidence above 0.80 (the highest reaching **0.997**).

---

## Algorithms

### SafeTS (Safety-constrained Thompson Sampling)

Builds safe action sets from classifier posteriors. For each query:
1. **Confidence gate** — if classifier confidence < τ, fall back to on-premises
2. **Safe action set** — include platform only if posterior violation probability ≤ ε
3. **Thompson Sampling** — select cheapest platform from the safe set

**Limitation:** When a T3 (PHI) query is misclassified as T2 with confidence 0.997, the posterior assigns near-zero violation probability. The safe action set includes unauthorized platforms because the classifier is so confident in its wrong answer that the safety check trusts it. Result: **1–8 violations** depending on ε.

### CARES (Compliance-Aware Residual Envelope Scoring)

Replaces calibration-dependent posteriors with a **distribution-free conformal approach**:

1. **Calibration** — compute asymmetric residuals on labeled calibration data:
   ```
   R_i = λ · max(s_i - ŝ_i, 0) + μ · Φ(π̂, s_i)
   ```
   where λ penalizes sensitivity underestimation and Φ is an adversarial coupling

2. **Routing** — for a new query, compute the safety envelope:
   ```
   U(q) = ŝ + ⌈q̂_δ⌉
   ```
   Route only to platforms with clearance ≥ U(q)

**Guarantee:** For any classifier (regardless of calibration quality):
```
Pr[violation] ≤ δ + 1/(n+1)
```
This bound requires only exchangeability — not calibration, not distributional assumptions, not classifier accuracy.

---

## Installation

```bash
git clone https://github.com/<your-username>/phi-guard.git
cd phi-guard
pip install -r requirements.txt
```

- Python 3.9+
- No GPU required — all experiments run on CPU (~10 min for DistilBERT fine-tuning)

---

## Reproducing the Paper

### Pipeline Overview

```
 STAGE 1: Data Construction
 ─────────────────────────────────────────────────
 22_scale_to_30k.py             Build the 30K dataset (MIMIC-IV + Synthea + MedQA)
                                 Uses header stripping, 20% PHI injection, T2 name pollution

 STAGE 2: Classifier Training
 ─────────────────────────────────────────────────
 02_train_classifier.py         Fine-tune DistilBERT for 4-class sensitivity

 STAGE 3: Core Routing Evaluation
 ─────────────────────────────────────────────────
 19b_cares_routing_fulltest.py  CARES routing (val = calibration, full test = eval)
 05_bandit_routing.py           SafeTS (CompTS) routing
 03_evaluate_routing.py         Static baselines (Greedy, StaticILP, Threshold)
 21_pilot_barp_baselines.py     PILOT & BaRP baseline reimplementations

 STAGE 4: Analyses & Ablations
 ─────────────────────────────────────────────────
 26_classifier_ablation.py      DistilBERT vs TF-IDF+LR vs Regex+NER under CARES
 24_mimic3_crossval.py          Cross-dataset generalization (MIMIC-III, 10K queries)
 25_calibration_honest.py       ECE/MCE analysis, reliability diagrams
 07_theoretical_analysis.py     Formal proofs (Theorems 1–4) + Monte Carlo verification
 12_distribution_shift.py       Flu-season shift simulation
 27_recalibration_analysis.py   Drift robustness & recalibration schedule

 STAGE 5: Real-World Validation
 ─────────────────────────────────────────────────
 09_real_endpoint_eval.py       GPT-4o-mini + Ollama/Llama-3.1 real endpoints
 23_real_routing_loop.py        End-to-end routing loop with real APIs
 11_quality_comparison.py       BERTScore quality preservation across platforms

 STAGE 6: Figures & Tables
 ─────────────────────────────────────────────────
 04_generate_figures.py         Paper-ready figures and LaTeX tables
```

### Quick Start

```bash
# 1 — Build 30K dataset (requires MIMIC-IV credentialed access)
python scripts/22_scale_to_30k.py \
    --mimic-notes /path/to/mimic-iv-note/discharge.csv \
    --synthea-dir /path/to/synthea/output/fhir

# 2 — Train classifier
python scripts/02_train_classifier.py

# 3 — Run CARES routing (main result → Table 2)
python scripts/19b_cares_routing_fulltest.py

# 4 — Run all baselines
python scripts/05_bandit_routing.py
python scripts/03_evaluate_routing.py
python scripts/21_pilot_barp_baselines.py

# 5 — Classifier ablation (→ Table 3)
python scripts/26_classifier_ablation.py

# 6 — MIMIC-III cross-validation (→ Table 4)
python scripts/24_mimic3_crossval.py

# 7 — Generate figures
python scripts/04_generate_figures.py
```

---

## Full Script Reference

| # | Script | Purpose | Paper |
|---|--------|---------|-------|
| 01 | `01_generate_dataset.py` | Initial dataset with adversarial examples (~7K) | superseded |
| 01b | `01_generate_dataset_v3.py` | Dataset from Synthea FHIR + MedQA + MTSamples | superseded |
| **02** | **`02_train_classifier.py`** | **Fine-tune DistilBERT (4-class sensitivity)** | **§6.1** |
| **03** | **`03_evaluate_routing.py`** | **Static routing baselines** | **Table 2** |
| **04** | **`04_generate_figures.py`** | **Paper figures and LaTeX tables** | **Figs 1–5** |
| **05** | **`05_bandit_routing.py`** | **SafeTS (CompTS) routing** | **Table 2, §4.1** |
| 06 | `06_mimic_ingest.py` | MIMIC-IV ingestion + full PHI re-injection (v1) | §6.1 |
| 06b | `06_mimic_ingest_v2.py` | Partial PHI re-injection (prevents shortcut learning) | §6.1 |
| **07** | **`07_theoretical_analysis.py`** | **Formal proofs + Monte Carlo verification** | **§5** |
| 08 | `08_calibration_analysis.py` | ECE/MCE + temperature scaling | §6.5 |
| **09** | **`09_real_endpoint_eval.py`** | **Real endpoint validation (GPT-4o-mini + Ollama)** | **Table 5** |
| 10 | `10_baseline_classifier.py` | TF-IDF + Logistic Regression baseline | Table 3 |
| 11 | `11_quality_comparison.py` | BERTScore cross-platform quality | Appendix |
| **12** | **`12_distribution_shift.py`** | **Flu-season distribution shift simulation** | **Table 6** |
| 13 | `13_threshold_baseline_and_real_costs.py` | Threshold baselines + real cost model | Table 2 |
| 13b | `13_threshold_baseline_and_real_costs_v2.py` | Amortized on-prem infrastructure costs | Discussion |
| 14 | `14_hybrid_classifier.py` | DistilBERT + rule-based PHI ensemble | exploration |
| 15 | `15_diagnose_routing.py` | Debug: why SafeTS routes everything on-prem | diagnostic |
| 16 | `16_fix_tier_probs_and_reeval.py` | Reconstruct softmax from predicted_tier + confidence | data fix |
| 17 | `17_scale_and_rerun.py` | Scale to 10K+ and retrain | superseded |
| 18 | `18_harder_dataset.py` | Header stripping + adversarial T2/T3 boundary | §6.1 |
| 19 | `19_cares_routing.py` | CARES algorithm (split test evaluation) | §4.2 |
| **19b** | **`19b_cares_routing_fulltest.py`** | **CARES with full test eval (val=calibration)** | **Table 2** |
| 20 | `20_hardest_dataset.py` | Max difficulty: 20% injection + name pollution | §6.1 |
| **21** | **`21_pilot_barp_baselines.py`** | **PILOT (LinUCB) & BaRP (REINFORCE) baselines** | **Table 2** |
| **22** | **`22_scale_to_30k.py`** | **Scale to 30K dataset** | **§6.1** |
| **23** | **`23_real_routing_loop.py`** | **End-to-end routing with real LLM endpoints** | **Table 5** |
| **24** | **`24_mimic3_crossval.py`** | **MIMIC-III cross-dataset validation (10K queries)** | **Table 4** |
| **25** | **`25_calibration_honest.py`** | **ECE/MCE analysis, SafeTS invalidity proof** | **§6.5, Fig 4** |
| **26** | **`26_classifier_ablation.py`** | **Classifier ablation (DistilBERT/TF-IDF/Regex)** | **Table 3** |
| **27** | **`27_recalibration_analysis.py`** | **Drift robustness & recalibration schedule** | **Table 8** |

> **Bold** = scripts producing final paper results. Others show the iterative research process.

---

## Data

### Obtaining the Data

**MIMIC-IV-Note v2.2** (requires credentialed access):
1. Complete CITI training at https://about.citiprogram.org/
2. Sign the PhysioNet Data Use Agreement
3. Request access at https://physionet.org/content/mimic-iv-note/2.2/

**MIMIC-III** (cross-validation): same credentialing — https://physionet.org/content/mimiciii/

**Synthea** (synthetic PHI, Apache 2.0):
```bash
git clone https://github.com/synthetichealth/synthea.git
cd synthea && ./run_synthea -p 1000 --exporter.fhir.export=true
```

**MedQA** (open access):
```python
from datasets import load_dataset
dataset = load_dataset("bigbio/med_qa")
```

### Dataset Hardening Techniques

The T2/T3 boundary is deliberately made difficult through three techniques (scripts 18, 20, 22):

1. **Header stripping** — remove structured fields (Name, Unit No, DOB) from all notes; only clinical narrative from "Chief Complaint" onward remains
2. **Low-rate PHI injection** (20%) — T3 notes have only 2–5 real entities buried in long text with remaining `___` markers intact
3. **T2 name pollution** — provider names, hospital names, and 10% patient-like names added to T2 notes

This forces the classifier to learn actual PHI detection rather than surface heuristics like "has `___` markers = T2."

---

## Hyperparameters

| Component | Parameter | Value |
|-----------|-----------|-------|
| **Classifier** | Model | DistilBERT-base-uncased |
| | Learning rate | 2 × 10⁻⁵ |
| | Batch size | 16 |
| | Max sequence length | 256 |
| | Epochs | 4 (early stopping patience 3) |
| **SafeTS** | ε (violation bound) | 0.02 |
| | τ (confidence threshold) | 0.70 |
| | α₀ (Dirichlet prior) | **1** |
| | β (exploration noise) | 0.01 |
| **CARES** | δ (conformal level) | 0.005 |
| | λ (underestimation penalty) | 1.0 |
| | μ (adversarial coupling) | 0.5 |
| | τ (confidence threshold) | 0.70 |
| | Calibration set | Validation set (n = 2,976) |
| **Dataset** | Seed | 42 |
| | PHI injection rate | 20% |
| | Split | 18,033 / 2,976 / 8,991 |

---

## Real Endpoint Validation

```bash
# Set up endpoints
export OPENAI_API_KEY="your-key-here"
ollama pull llama3.1:8b && ollama serve

# Run validation (Table 5)
python scripts/09_real_endpoint_eval.py
python scripts/23_real_routing_loop.py
python scripts/11_quality_comparison.py --n-queries 100
```

---

## Extending to Other Domains

PHI-GUARD generalizes to any domain with tiered data sensitivity:

| Domain | Tiers | Constraint |
|--------|-------|-----------|
| **HIPAA** | Public → Internal → De-identified → PHI | Platform clearance ≥ query sensitivity |
| **GDPR** | Non-personal → Pseudonymized → Personal → Special category | EU-authorized processors only |
| **PCI-DSS** | Public → Internal → Cardholder → Full track | PCI compliance level |
| **Government** | Unclassified → CUI → Secret → Top Secret | Facility security clearance |

CARES calibrates against *any* classifier's empirical error distribution — the conformal envelope adapts automatically.

---

## Research Development Log

The numbered scripts document the full research journey:

- **01–06** — Dataset construction: synthetic → MIMIC-IV ingestion → PHI re-injection
- **07–08** — Theory + calibration: discovered MCE=0.217 invalidates SafeTS guarantee
- **09–13** — Validation: real endpoints, threshold baselines, cost model corrections
- **14–16** — Debugging: why SafeTS defaults to all on-prem, tier_probs reconstruction
- **17–20** — Dataset hardening: header stripping → name pollution → 20% injection
- **19** — **CARES development**: the core algorithmic contribution
- **21–22** — Scaling to 30K with PILOT/BaRP baselines
- **23–27** — Publication analyses: real routing loop, MIMIC-III crossval, ablation, drift

---

## Citation

```bibtex
@article{sharma2026phiguard,
  title={{PHI-GUARD}: Compliance-Aware {LLM} Routing for Healthcare 
         with Distribution-Free Safety Guarantees},
  author={Sharma, Aman},
  journal={arXiv preprint arXiv:2602.XXXXX},
  year={2026}
}
```

---

## Ethics & Compliance

- **Data:** MIMIC-IV-Note v2.2 under PhysioNet DUA. Exempt from IRB under 45 CFR §46.104(d)(4)
- **PHI:** All patient information is **synthetic** ([Synthea](https://github.com/synthetichealth/synthea), Apache 2.0)
- **Deployment:** Research framework — clinical use requires IRB approval + BAA agreements
- **AI Disclosure:** AI tools (Claude, Anthropic) assisted with code development, mathematical formulation, and manuscript preparation. Problem identification, experimental design, data sourcing, domain expertise, and all research decisions were made by the author

## License

MIT — see [LICENSE](LICENSE). Data: MIMIC-IV (PhysioNet License 1.5.0) · Synthea (Apache 2.0) · MedQA (open access)

## Acknowledgments

Independent research at Colorado Technical University. Does not represent the views, policies, or endorsement of Blue Shield of California. No proprietary data, systems, or resources were used.
