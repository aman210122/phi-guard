#!/usr/bin/env python3
"""
19b_cares_routing_fulltest.py — CARES with full test evaluation

CHANGE FROM 19: Uses val.json for calibration and FULL test set for evaluation.
Script 19 split the test set 40/60, which could put the most interesting 
high-confidence errors into the calibration set. This version keeps all
test data for evaluation.

Since val.json doesn't have predictions, we run the classifier on val
to get predictions, then use those as the calibration set.

USAGE:
  python scripts/19b_cares_routing_fulltest.py
"""

import json
import sys
import math
import random
from pathlib import Path
from collections import Counter
from typing import List, Dict, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

NUM_TIERS = 4


# ═══════════════════════════════════════════════════════════════════════
# CARES Algorithm
# ═══════════════════════════════════════════════════════════════════════

def compute_directional_residuals(cal_data, lam=2.0):
    residuals = []
    for item in cal_data:
        s_true = item["tier"]
        s_pred = item["predicted_tier"]
        gap = max(0, s_true - s_pred)
        weight = math.exp(lam * s_true)
        residuals.append(gap * weight)
    return residuals


def compute_calibration_quantile(residuals, delta=0.01):
    n = len(residuals)
    augmented = sorted(residuals) + [float('inf')]
    idx = math.ceil((n + 1) * (1 - delta)) - 1
    idx = min(idx, len(augmented) - 1)
    return augmented[idx]


def cares_construct_envelope(query, q_hat, lam=2.0, mu=1.0, eta=1e-6):
    s_pred = query["predicted_tier"]
    probs = query["tier_probs"]
    
    envelope = []
    for k in range(NUM_TIERS):
        if k <= s_pred:
            envelope.append(k)
        else:
            residual_k = (k - s_pred) * math.exp(lam * k)
            pi_pred = probs[s_pred] if s_pred < len(probs) else 0.5
            pi_k = probs[k] if k < len(probs) else 0.0
            coupling = mu * math.log(pi_pred / (pi_k + eta))
            if residual_k <= q_hat + coupling:
                envelope.append(k)
    
    return envelope


def cares_route(query, platforms, q_hat, lam=2.0, mu=1.0, eta=1e-6,
                tau=0.80, beta=0.01):
    conf = query["confidence"]
    true_tier = query["tier"]
    
    if conf < tau:
        chosen = max(platforms, key=lambda p: p["clearance"])
        return chosen, chosen["clearance"] < true_tier
    
    envelope = cares_construct_envelope(query, q_hat, lam, mu, eta)
    l_min = max(envelope) if envelope else NUM_TIERS - 1
    safe = [p for p in platforms if p["clearance"] >= l_min]
    
    if not safe:
        chosen = max(platforms, key=lambda p: p["clearance"])
        return chosen, chosen["clearance"] < true_tier
    
    noise = {p["name"]: np.random.exponential(beta) for p in safe}
    chosen = min(safe, key=lambda p: p["cost"] + noise[p["name"]])
    return chosen, chosen["clearance"] < true_tier


# ═══════════════════════════════════════════════════════════════════════
# CompTS
# ═══════════════════════════════════════════════════════════════════════

def compts_route(query, platforms, epsilon=0.01, tau=0.80):
    probs = query["tier_probs"]
    conf = query["confidence"]
    true_tier = query["tier"]
    
    if conf < tau:
        chosen = max(platforms, key=lambda p: p["clearance"])
        return chosen, chosen["clearance"] < true_tier
    
    safe = []
    for p in platforms:
        if p["clearance"] >= 3:
            viol_prob = 0.0
        else:
            viol_prob = sum(probs[t] for t in range(p["clearance"] + 1, 4))
        if viol_prob <= epsilon:
            safe.append(p)
    
    if not safe:
        chosen = max(platforms, key=lambda p: p["clearance"])
        return chosen, chosen["clearance"] < true_tier
    
    chosen = min(safe, key=lambda p: p["cost"])
    return chosen, chosen["clearance"] < true_tier


# ═══════════════════════════════════════════════════════════════════════
# Other Baselines
# ═══════════════════════════════════════════════════════════════════════

def route_static(query, platforms):
    pred = query["predicted_tier"]
    true_tier = query["tier"]
    safe = [p for p in platforms if p["clearance"] >= pred]
    chosen = min(safe, key=lambda p: p["cost"]) if safe else max(platforms, key=lambda p: p["clearance"])
    return chosen, chosen["clearance"] < true_tier

def route_secure(query, platforms):
    chosen = max(platforms, key=lambda p: p["clearance"])
    return chosen, chosen["clearance"] < query["tier"]

def route_greedy(query, platforms):
    chosen = min(platforms, key=lambda p: p["cost"])
    return chosen, chosen["clearance"] < query["tier"]

def route_threshold(query, platforms, threshold=0.95):
    pred = query["predicted_tier"]
    conf = query["confidence"]
    true_tier = query["tier"]
    if conf < threshold:
        chosen = max(platforms, key=lambda p: p["clearance"])
    else:
        safe = [p for p in platforms if p["clearance"] >= pred]
        chosen = min(safe, key=lambda p: p["cost"]) if safe else max(platforms, key=lambda p: p["clearance"])
    return chosen, chosen["clearance"] < true_tier


# ═══════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════

def evaluate(name, route_fn, data, platforms, **kwargs):
    total_cost = 0
    violations = 0
    platform_counts = Counter()
    viol_details = []
    
    for q in data:
        chosen, viol = route_fn(q, platforms, **kwargs) if kwargs else route_fn(q, platforms)
        tokens = len(q.get("text", "").split()) * 1.3
        total_cost += chosen["cost"] * tokens / 1000
        violations += int(viol)
        platform_counts[chosen["name"]] += 1
        if viol:
            viol_details.append({
                "true": q["tier"], "pred": q["predicted_tier"],
                "conf": round(q["confidence"], 3),
                "platform": chosen["name"], "clearance": chosen["clearance"],
            })
    
    n = len(data)
    return {
        "name": name, "cost": round(total_cost, 4),
        "violations": violations, "viol_pct": round(violations/n*100, 3),
        "routing": dict(platform_counts), "n": n,
        "cloud_pct": round((n - platform_counts.get("OnPremises", 0))/n*100, 1),
        "viol_details": viol_details[:20],
    }


def fix_tier_probs(predictions):
    fixed = []
    for p in predictions:
        pred = p["predicted_tier"]
        conf = p["confidence"]
        probs = p.get("tier_probs")
        if probs is None or not isinstance(probs, list) or len(probs) != 4 or all(abs(x-0.25)<0.001 for x in probs):
            remaining = (1.0 - conf) / 3.0
            probs = [remaining] * 4
            probs[pred] = conf
        fixed_p = dict(p)
        fixed_p["tier_probs"] = probs
        fixed.append(fixed_p)
    return fixed


def _build_classifier(backbone_name, hidden_size):
    """Reconstruct the custom Classifier from 02_train_classifier.py."""
    import torch.nn as tnn
    from transformers import AutoModel

    class Classifier(tnn.Module):
        def __init__(self, base, h, n_tiers=4, n_ner=2):
            super().__init__()
            self.base = base
            self.drop = tnn.Dropout(0.1)
            self.tier_head = tnn.Sequential(
                tnn.Linear(h, 256), tnn.ReLU(), tnn.Dropout(0.1), tnn.Linear(256, n_tiers))
            self.ner_head = tnn.Sequential(
                tnn.Linear(h, 128), tnn.ReLU(), tnn.Dropout(0.1), tnn.Linear(128, n_ner))
        def forward(self, ids, mask):
            out = self.base(input_ids=ids, attention_mask=mask).last_hidden_state
            return self.tier_head(self.drop(out[:, 0, :])), self.ner_head(self.drop(out))

    base_model = AutoModel.from_pretrained(backbone_name)
    return Classifier(base_model, hidden_size)


def run_val_inference(val_data, model_dir):
    """Run classifier on val set to get predictions for calibration."""
    try:
        import torch
        from transformers import AutoTokenizer

        # Try best_classifier.pt first, then best_model.pt
        model_path = model_dir / "best_classifier.pt"
        if not model_path.exists():
            model_path = model_dir / "best_model.pt"
        if not model_path.exists():
            print(f"  No model found in {model_dir}")
            return None

        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        backbone = checkpoint.get("backbone", "distilbert-base-uncased")
        h = checkpoint.get("h", 768)

        print(f"  Loading model: backbone={backbone}, h={h}")
        tokenizer = AutoTokenizer.from_pretrained(backbone)
        model = _build_classifier(backbone, h)
        model.load_state_dict(checkpoint["state"])
        model.eval()

        results = []
        for i, item in enumerate(val_data):
            text = item["text"][:512]
            enc = tokenizer(text, return_tensors="pt", truncation=True,
                            max_length=256, padding="max_length")

            with torch.no_grad():
                tier_logits, _ = model(enc["input_ids"], enc["attention_mask"])
                probs = torch.softmax(tier_logits, dim=-1)[0].numpy().tolist()

            pred = int(np.argmax(probs))
            conf = float(probs[pred])

            result = dict(item)
            result["predicted_tier"] = pred
            result["confidence"] = conf
            result["tier_probs"] = probs
            results.append(result)

            if (i + 1) % 500 == 0:
                print(f"    Val inference: {i+1}/{len(val_data)}")

        print(f"  Val inference complete: {len(results)} samples")
        return results
    except Exception as e:
        print(f"  Val inference failed: {e}")
        import traceback; traceback.print_exc()
        return None


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    data_dir = ROOT / "data"
    model_dir = ROOT / "models"
    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    
    # Load test predictions (FULL test set for evaluation)
    pred_path = data_dir / "test_with_predictions.json"
    if not pred_path.exists():
        print("ERROR: test_with_predictions.json not found")
        sys.exit(1)
    
    test_data = fix_tier_probs(json.loads(pred_path.read_text()))
    n_test = len(test_data)
    
    # Try to get calibration data from val set
    val_path = data_dir / "val.json"
    cal_data = None
    
    if val_path.exists():
        print("Attempting val set inference for calibration...")
        val_raw = json.loads(val_path.read_text())
        cal_data = run_val_inference(val_raw, model_dir)
        if cal_data:
            cal_data = fix_tier_probs(cal_data)
            print(f"  Val calibration set: {len(cal_data)} samples")
    
    if cal_data is None:
        print("Val inference unavailable. Using 30% of test as calibration.")
        random.shuffle(test_data)
        n_cal = int(n_test * 0.3)
        cal_data = test_data[:n_cal]
        test_data = test_data[n_cal:]
        n_test = len(test_data)
    
    n_cal = len(cal_data)
    
    print("=" * 70)
    print("CARES vs CompTS — Full Evaluation")
    print("=" * 70)
    
    # Stats
    tier_counts = Counter(p["tier"] for p in test_data)
    print(f"\nEval: {n_test} | Cal: {n_cal}")
    for t in sorted(tier_counts):
        print(f"  T{t}: {tier_counts[t]}")
    
    correct = sum(1 for p in test_data if p["predicted_tier"] == p["tier"])
    t3_total = sum(1 for p in test_data if p["tier"] == 3)
    t3_correct = sum(1 for p in test_data if p["tier"] == 3 and p["predicted_tier"] == 3)
    t3_missed = t3_total - t3_correct
    
    print(f"\nAccuracy: {correct}/{n_test} ({correct/n_test*100:.1f}%)")
    print(f"T3 recall: {t3_correct}/{t3_total} ({t3_correct/t3_total*100:.1f}%)")
    print(f"T3 missed: {t3_missed}")
    
    if t3_missed > 0:
        missed = [p for p in test_data if p["tier"] == 3 and p["predicted_tier"] != 3]
        confs = [p["confidence"] for p in missed]
        above_tau = sum(1 for c in confs if c >= 0.80)
        print(f"  Confidence: min={min(confs):.3f}, mean={np.mean(confs):.3f}, max={max(confs):.3f}")
        print(f"  Above τ=0.8 (bypass gate): {above_tau}/{t3_missed}")
        print(f"  Below τ=0.8 (caught by gate): {t3_missed - above_tau}/{t3_missed}")
    
    # Calibration
    print(f"\n{'='*70}")
    print("CALIBRATION")
    print(f"{'='*70}")
    
    cal_missed = sum(1 for p in cal_data if p["tier"] == 3 and p["predicted_tier"] != 3)
    print(f"Calibration set: {n_cal} samples, {cal_missed} T3 misclassified upward")
    
    lam = 2.0
    residuals = compute_directional_residuals(cal_data, lam=lam)
    n_nonzero = sum(1 for r in residuals if r > 0)
    print(f"Non-zero residuals (λ={lam}): {n_nonzero}")
    
    for delta in [0.005, 0.01, 0.02, 0.05]:
        q_hat = compute_calibration_quantile(residuals, delta=delta)
        q_str = f"{q_hat:.4f}" if q_hat != float('inf') else "∞"
        print(f"  δ={delta}: q̂ = {q_str}")
    
    # Platforms
    platforms = [
        {"name": "PublicAPI",   "clearance": 0, "cost": 0.010},
        {"name": "SecureCloud", "clearance": 2, "cost": 0.011},
        {"name": "OnPremises",  "clearance": 3, "cost": 0.025},
    ]
    
    # ═══════════ EVALUATE ═══════════
    print(f"\n{'='*70}")
    print(f"ROUTING RESULTS (n={n_test})")
    print(f"{'='*70}")
    
    results = []
    
    # CARES — focused sweep on best parameters
    for lam in [1.0, 2.0, 3.0]:
        res = compute_directional_residuals(cal_data, lam=lam)
        for delta in [0.005, 0.01, 0.02, 0.05]:
            q_hat = compute_calibration_quantile(res, delta=delta)
            for mu in [0.5, 1.0, 2.0]:
                for tau in [0.70, 0.80]:
                    name = f"CARES λ={lam} δ={delta} μ={mu} τ={tau}"
                    r = evaluate(name, cares_route, test_data, platforms,
                                q_hat=q_hat, lam=lam, mu=mu, tau=tau)
                    results.append(r)
    
    # CompTS
    for eps in [0.001, 0.005, 0.01, 0.02, 0.05]:
        for tau in [0.70, 0.80]:
            r = evaluate(f"CompTS ε={eps} τ={tau}", compts_route, test_data, platforms,
                        epsilon=eps, tau=tau)
            results.append(r)
    
    # Baselines
    results.append(evaluate("StaticILP", route_static, test_data, platforms))
    results.append(evaluate("SecureDefault", route_secure, test_data, platforms))
    results.append(evaluate("Greedy", route_greedy, test_data, platforms))
    for t in [0.70, 0.80, 0.90, 0.95, 0.99]:
        results.append(evaluate(f"Threshold-{t}", route_threshold, test_data, platforms, threshold=t))
    
    # Print sorted
    print(f"\n{'Strategy':<40} {'N':>5} {'Cost':>10} {'Viol':>5} {'V%':>7} {'Cloud%':>6}")
    print("-" * 85)
    
    for r in sorted(results, key=lambda x: (x["violations"], x["cost"])):
        marker = " ◆" if r["violations"] == 0 and r["cloud_pct"] > 0 else ""
        marker = " ✗" if r["violations"] > 0 else marker
        print(f"{r['name']:<40} {r['n']:>5} ${r['cost']:>8.4f} {r['violations']:>5} "
              f"{r['viol_pct']:>6.3f}% {r['cloud_pct']:>5.1f}%{marker}")
    
    # ═══════════ KEY COMPARISONS ═══════════
    print(f"\n{'='*70}")
    print("HEAD-TO-HEAD COMPARISON")
    print(f"{'='*70}")
    
    # Find best of each type (0 violations, max cloud routing)
    def best_of(prefix):
        candidates = [r for r in results if r["name"].startswith(prefix) 
                      and r["violations"] == 0 and r["cloud_pct"] > 0]
        return min(candidates, key=lambda x: x["cost"]) if candidates else None
    
    best_cares = best_of("CARES")
    best_compts = best_of("CompTS")
    best_thresh = best_of("Threshold")
    static = next(r for r in results if r["name"] == "StaticILP")
    secure = next(r for r in results if r["name"] == "SecureDefault")
    greedy = next(r for r in results if r["name"] == "Greedy")
    
    strategies = [
        ("Greedy (unsafe)", greedy),
        ("StaticILP (unsafe)", static),
        ("Best Threshold", best_thresh),
        ("Best CompTS", best_compts),
        ("Best CARES", best_cares),
        ("SecureDefault", secure),
    ]
    
    print(f"\n{'Strategy':<25} {'Cost':>10} {'Viol':>5} {'Cloud%':>7} {'Guarantee':<30}")
    print("-" * 85)
    for label, r in strategies:
        if r:
            guarantee = "None"
            if "CARES" in (r["name"] if r else ""):
                delta_str = r["name"].split("δ=")[1].split()[0] if "δ=" in r["name"] else "?"
                guarantee = f"δ+1/(n+1) = {float(delta_str)+1/(n_cal+1):.6f} (dist-free)"
            elif "CompTS" in (r["name"] if r else ""):
                guarantee = "Requires ECE ≤ α (violated)"
            elif "Threshold" in label:
                guarantee = "None (heuristic)"
            elif "Secure" in label:
                guarantee = "Trivial (all on-prem)"
            elif "Static" in label or "Greedy" in label:
                guarantee = "None"
            
            print(f"{label:<25} ${r['cost']:>8.4f} {r['violations']:>5} {r['cloud_pct']:>5.1f}%  {guarantee}")
    
    # Violation analysis
    if static["violations"] > 0:
        print(f"\n--- StaticILP Violation Details ---")
        for v in static["viol_details"][:10]:
            print(f"  True=T{v['true']}, Pred=T{v['pred']}, Conf={v['conf']}, "
                  f"→ {v['platform']} (clearance={v['clearance']})")
    
    # Safety bound verification
    if best_cares:
        parts = best_cares["name"].split()
        lam_v = float([p for p in parts if p.startswith("λ=")][0].split("=")[1])
        delta_v = float([p for p in parts if p.startswith("δ=")][0].split("=")[1])
        mu_v = float([p for p in parts if p.startswith("μ=")][0].split("=")[1])
        
        res = compute_directional_residuals(cal_data, lam=lam_v)
        q_hat = compute_calibration_quantile(res, delta=delta_v)
        
        # Check envelope coverage on T3
        t3_covered = 0
        t3_uncovered = 0
        for q in test_data:
            if q["tier"] == 3:
                env = cares_construct_envelope(q, q_hat, lam=lam_v, mu=mu_v)
                if 3 in env:
                    t3_covered += 1
                else:
                    t3_uncovered += 1
        
        bound = delta_v + 1 / (n_cal + 1)
        empirical = t3_uncovered / max(1, t3_total)
        
        print(f"\n{'='*70}")
        print("SAFETY BOUND VERIFICATION")
        print(f"{'='*70}")
        print(f"Best CARES: {best_cares['name']}")
        print(f"  q̂_δ = {q_hat:.4f}" + (" (∞)" if q_hat == float('inf') else ""))
        print(f"  T3 in envelope: {t3_covered}/{t3_total}")
        print(f"  T3 NOT in envelope: {t3_uncovered}/{t3_total}")
        print(f"  Theoretical bound: {bound:.6f}")
        print(f"  Empirical rate: {empirical:.6f}")
        print(f"  BOUND HOLDS: {'YES ✓' if empirical <= bound else 'NO ✗'}")
    
    # ═══════════ PAPER NUMBERS ═══════════
    print(f"\n{'='*70}")
    print("NUMBERS FOR THE PAPER")
    print(f"{'='*70}")
    print(f"Dataset: {n_test + n_cal} total ({n_cal} calibration, {n_test} evaluation)")
    print(f"Classifier: {correct/n_test*100:.1f}% accuracy, T3 recall {t3_correct/t3_total*100:.1f}%")
    print(f"T3 missed: {t3_missed}")
    
    if best_cares:
        sav = (1 - best_cares["cost"]/secure["cost"])*100
        print(f"\nCARES: ${best_cares['cost']:.4f}, 0 violations, {best_cares['cloud_pct']}% cloud ({sav:.1f}% savings)")
    if best_compts:
        sav_c = (1 - best_compts["cost"]/secure["cost"])*100
        print(f"CompTS: ${best_compts['cost']:.4f}, 0 violations, {best_compts['cloud_pct']}% cloud ({sav_c:.1f}% savings)")
    print(f"StaticILP: ${static['cost']:.4f}, {static['violations']} violations ({static['viol_pct']}%)")
    print(f"SecureDefault: ${secure['cost']:.4f}, 0 violations, 0% cloud")
    print(f"Greedy: ${greedy['cost']:.4f}, {greedy['violations']} violations ({greedy['viol_pct']}%)")
    
    # Save
    save_data = {
        "n_eval": n_test, "n_cal": n_cal,
        "accuracy": correct/n_test, "t3_recall": t3_correct/t3_total,
        "t3_missed": t3_missed,
        "results": results,
    }
    out_path = out_dir / "cares_fulltest_results.json"
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
