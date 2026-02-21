# Data Access Instructions

## MIMIC-IV-Note v2.2 (Required)

MIMIC-IV clinical notes require credentialed access through PhysioNet.

### Steps:
1. Create a PhysioNet account: https://physionet.org/register/
2. Complete the CITI training course for human subjects research
3. Submit credentialing application: https://physionet.org/settings/credentialing/
4. Sign the MIMIC-IV Data Use Agreement
5. Download MIMIC-IV-Note v2.2: https://physionet.org/content/mimic-iv-note/2.2/

### Expected file structure:
```
data/
├── mimic-iv/
│   └── note/
│       └── discharge.csv.gz
├── mimic-iii/          # Optional, for cross-dataset evaluation
│   └── NOTEEVENTS.csv.gz
└── medqa/              # Optional, auto-generated if missing
    └── questions.json
```

**License:** PhysioNet Credentialed Health Data License v1.5.0
**DOI:** 10.13026/1n74-ne17

## MIMIC-III (Optional, for cross-dataset evaluation)

Required only for Table VI (cross-dataset generalization).
Access: https://physionet.org/content/mimiciii/1.4/

## MedQA (Optional)

Public medical knowledge questions for T0 tier.
If not provided, the pipeline auto-generates equivalent queries.
Available at: https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options

## Synthea (Synthetic PHI generation)

Open-source synthetic patient generator (Apache 2.0).
PHI entities are generated programmatically — no separate download needed.
Source: https://github.com/synthetichealth/synthea

## Important Notes

- **Do NOT commit MIMIC data to this repository** — it violates the DUA
- All PHI in experiments is synthetic (Synthea-generated)
- The dataset construction pipeline handles all transformations automatically
