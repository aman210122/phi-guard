#!/usr/bin/env python3
"""02 — Train PHI-GUARD sensitivity classifier (ClinicalBERT + joint NER/tier heads)."""

import json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
with open(ROOT / "configs/config.yaml") as f:
    cfg = yaml.safe_load(f)

DATA = ROOT / cfg["dataset"]["output_dir"]
MDIR = ROOT / cfg["classifier"]["model_dir"]; MDIR.mkdir(exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BACKBONE = cfg["classifier"]["backbone"] if torch.cuda.is_available() else cfg["classifier"]["backbone_fallback"]
print(f"Device: {DEVICE} | Backbone: {BACKBONE}")

tokenizer = AutoTokenizer.from_pretrained(BACKBONE)
base_model = AutoModel.from_pretrained(BACKBONE)
H = base_model.config.hidden_size

# ── Dataset ────────────────────────────────────────────────────────────────
class DS(Dataset):
    def __init__(self, samples, tok, maxlen=256):
        self.samples, self.tok, self.ml = samples, tok, maxlen
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        s = self.samples[i]
        enc = self.tok(s["text"], max_length=self.ml, padding="max_length",
                       truncation=True, return_tensors="pt", return_offsets_mapping=True)
        ids = enc["input_ids"].squeeze(0)
        mask = enc["attention_mask"].squeeze(0)
        offsets = enc["offset_mapping"].squeeze(0)
        ner = torch.zeros(self.ml, dtype=torch.long)
        if s.get("phi_entities"):
            tl = s["text"].lower()
            for ent in s["phi_entities"]:
                st = tl.find(str(ent).lower())
                if st < 0: continue
                en = st + len(str(ent))
                for ti in range(self.ml):
                    ts, te = offsets[ti].tolist()
                    if te == 0: continue
                    if ts >= st and te <= en: ner[ti] = 1
        return dict(input_ids=ids, attention_mask=mask,
                    tier_label=torch.tensor(s["tier"], dtype=torch.long), ner_labels=ner)

# ── Model ──────────────────────────────────────────────────────────────────
class Classifier(nn.Module):
    def __init__(self, base, h, n_tiers=4, n_ner=2):
        super().__init__()
        self.base = base
        self.drop = nn.Dropout(0.1)
        self.tier_head = nn.Sequential(nn.Linear(h, 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, n_tiers))
        self.ner_head = nn.Sequential(nn.Linear(h, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, n_ner))
    def forward(self, ids, mask):
        out = self.base(input_ids=ids, attention_mask=mask).last_hidden_state
        return self.tier_head(self.drop(out[:, 0, :])), self.ner_head(self.drop(out))

# ── Train / Eval ───────────────────────────────────────────────────────────
def train_epoch(model, dl, opt, alpha):
    model.train(); total = 0
    ce_tier, ce_ner = nn.CrossEntropyLoss(), nn.CrossEntropyLoss(ignore_index=-100)
    for b in tqdm(dl, desc="  Train", leave=False):
        ids, mask = b["input_ids"].to(DEVICE), b["attention_mask"].to(DEVICE)
        tl, nl = b["tier_label"].to(DEVICE), b["ner_labels"].to(DEVICE)
        nl2 = nl.clone(); nl2[mask == 0] = -100
        opt.zero_grad()
        tlog, nlog = model(ids, mask)
        loss = alpha * ce_ner(nlog.view(-1, 2), nl2.view(-1)) + (1-alpha) * ce_tier(tlog, tl)
        loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        total += loss.item()
    return total / len(dl)

def evaluate(model, dl):
    model.eval()
    tp, tl, tc, np_, nl_ = [], [], [], [], []
    with torch.no_grad():
        for b in tqdm(dl, desc="  Eval", leave=False):
            ids, mask = b["input_ids"].to(DEVICE), b["attention_mask"].to(DEVICE)
            tlog, nlog = model(ids, mask)
            probs = torch.softmax(tlog, dim=-1)
            tp.extend(tlog.argmax(1).cpu().numpy())
            tl.extend(b["tier_label"].numpy())
            tc.extend(probs.max(1).values.cpu().numpy())
            npred = nlog.argmax(-1)
            for i in range(len(npred)):
                m = mask[i].bool().cpu()
                np_.extend(npred[i][m].cpu().numpy())
                nl_.extend(b["ner_labels"][i][m].numpy())
    return dict(tier_preds=np.array(tp), tier_labels=np.array(tl),
                tier_probs=np.array(tc), ner_preds=np.array(np_), ner_labels=np.array(nl_))

# ── Main ───────────────────────────────────────────────────────────────────
def main():
    print("="*60 + "\nPHI-GUARD Classifier Training\n" + "="*60)
    train = json.load(open(DATA / "train.json"))
    val = json.load(open(DATA / "val.json"))
    test = json.load(open(DATA / "test.json"))
    print(f"Train:{len(train)} Val:{len(val)} Test:{len(test)}")

    ml, bs = cfg["classifier"]["max_length"], cfg["classifier"]["batch_size"]
    trl = DataLoader(DS(train, tokenizer, ml), batch_size=bs, shuffle=True, num_workers=0)
    vl = DataLoader(DS(val, tokenizer, ml), batch_size=bs, num_workers=0)
    tel = DataLoader(DS(test, tokenizer, ml), batch_size=bs, num_workers=0)

    model = Classifier(base_model, H).to(DEVICE)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    opt = AdamW(model.parameters(), lr=cfg["classifier"]["learning_rate"],
                weight_decay=cfg["classifier"]["weight_decay"])
    alpha = cfg["classifier"]["multitask_alpha"]
    best_f1, patience, pat_cnt = 0, cfg["classifier"]["early_stopping_patience"], 0

    for ep in range(1, cfg["classifier"]["epochs"] + 1):
        print(f"\nEpoch {ep}")
        loss = train_epoch(model, trl, opt, alpha)
        print(f"  Loss: {loss:.4f}")
        vr = evaluate(model, vl)
        vf1 = f1_score(vr["tier_labels"], vr["tier_preds"], average="macro")
        print(f"  Val F1: {vf1:.4f}")
        if vf1 > best_f1:
            best_f1 = vf1; pat_cnt = 0
            torch.save(dict(state=model.state_dict(), backbone=BACKBONE, h=H, f1=vf1, ep=ep),
                       MDIR / "best_classifier.pt")
            print(f"  ✓ Saved (F1={vf1:.4f})")
        else:
            pat_cnt += 1; print(f"  No improvement ({pat_cnt}/{patience})")
            if pat_cnt >= patience: print("  Early stop!"); break

    # ── Test evaluation ────────────────────────────────────────────────────
    print("\n" + "="*60 + "\nTest Evaluation\n" + "="*60)
    ck = torch.load(MDIR / "best_classifier.pt", map_location=DEVICE, weights_only=False)
    model.load_state_dict(ck["state"])
    tr = evaluate(model, tel)

    names = ["T0_Public","T1_Internal","T2_Limited","T3_Restricted"]
    report = classification_report(tr["tier_labels"], tr["tier_preds"],
                                    target_names=names, output_dict=True, zero_division=0)
    print("\n" + classification_report(tr["tier_labels"], tr["tier_preds"],
                                       target_names=names, zero_division=0))

    # T3 safety
    t3m = tr["tier_labels"] == 3
    t3_recall = (tr["tier_preds"][t3m] == 3).mean() if t3m.any() else 0
    t3_mis = t3m & (tr["tier_preds"] != 3)
    tau = cfg["routing"]["confidence_threshold"]
    print(f"=== SAFETY ===")
    print(f"T3 Recall: {t3_recall:.4f} ({t3_mis.sum()}/{t3m.sum()} misclassified)")
    if t3_mis.any():
        mp = tr["tier_probs"][t3_mis]
        print(f"  Misclassified confidence: mean={mp.mean():.3f} max={mp.max():.3f}")
        print(f"  Caught by τ={tau}: {(mp < tau).sum()}/{len(mp)}")
    correct = tr["tier_preds"] == tr["tier_labels"]
    print(f"Confidence: correct={tr['tier_probs'][correct].mean():.3f} "
          f"incorrect={tr['tier_probs'][~correct].mean():.3f}" if (~correct).any() else
          f"Confidence: correct={tr['tier_probs'][correct].mean():.3f} | No errors!")
    ner_f1 = f1_score(tr["ner_labels"], tr["ner_preds"], average="binary", zero_division=0)
    print(f"NER F1: {ner_f1:.4f}")

    # Save results
    results = dict(backbone=BACKBONE, epoch=ck["ep"],
        test_macro_f1=float(report["macro avg"]["f1-score"]),
        t3_recall=float(t3_recall), t3_misclassified=int(t3_mis.sum()), t3_total=int(t3m.sum()),
        ner_f1=float(ner_f1),
        per_tier={n: dict(precision=report[n]["precision"], recall=report[n]["recall"],
                          f1=report[n]["f1-score"], support=report[n]["support"])
                  for n in names if n in report},
        confidence_threshold=tau,
        confusion_matrix=confusion_matrix(tr["tier_labels"], tr["tier_preds"]).tolist())
    json.dump(results, open(MDIR / "test_results.json", "w"), indent=2)

    # Save predictions for routing
    preds = [{**test[i], "predicted_tier": int(tr["tier_preds"][i]),
              "confidence": float(tr["tier_probs"][i])}
             for i in range(min(len(test), len(tr["tier_preds"])))]
    json.dump(preds, open(DATA / "test_with_predictions.json", "w"), indent=2)
    print(f"\nSaved: {MDIR}/test_results.json, {DATA}/test_with_predictions.json")

if __name__ == "__main__":
    main()
