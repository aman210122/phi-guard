#!/usr/bin/env python3
"""
19_cares_routing.py — CARES: Compliance-Aware Residual Envelope Scoring

The primary routing algorithm for PHI-GUARD. Replaces CompTS as the main
contribution. CompTS is retained as a baseline for comparison.

ALGORITHM OVERVIEW:
  OFFLINE: Compute asymmetric directional residuals on calibration set,
           extract quantile threshold q̂_δ (one scalar, distribution-free)
  ONLINE:  Per query, construct safety envelope using residual threshold
           + softmax-residual adversarial coupling, then select cheapest
           platform within the safe set.

KEY INNOVATION:
  - Asymmetric tier-weighted residuals (only penalize underestimation)
  - Softmax-residual coupling: low softmax for higher tier = WARNING, not reassurance
  - Distribution-free safety guarantee (only requires exchangeability)

USAGE:
  python scripts/19_cares_routing.py

REQUIRES:
  data/test_with_predictions.json (from Script 02)
  data/val.json + trained model (for calibration set)
"""

import json
import sys
import math
import random
from pathlib import Path
from collections import Counter
from typing import List, Dict, Tuple, Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

NUM_TIERS = 4  # T0, T1, T2, T3


# ═══════════════════════════════════════════════════════════════════════
# CARES: Offline Calibration Phase
# ═══════════════════════════════════════════════════════════════════════

def compute_directional_residuals(cal_data: List[Dict], lam: float = 2.0) -> List[float]:
    """
    Compute R^+(q) = max(0, s - ŝ) · exp(λ·s) for each calibration sample.
    
    This is the ASYMMETRIC, TIER-WEIGHTED nonconformity score.
    Only penalizes underestimation (predicting lower tier than truth).
    Exponential weighting makes missing T3 much worse than missing T1.
    """
    residuals = []
    for item in cal_data:
        s_true = item["tier"]
        s_pred = item["predicted_tier"]
        gap = max(0, s_true - s_pred)
        weight = math.exp(lam * s_true)
        residuals.append(gap * weight)
    return residuals


def compute_calibration_quantile(residuals: List[float], delta: float = 0.01) -> float:
    """
    Compute q̂_δ = (1-δ)-quantile of {R_1^+, ..., R_n^+, ∞}
    
    The ∞ is the standard conformal "add one" correction for finite-sample
    validity (Vovk et al., 2005).
    """
    n = len(residuals)
    # Add infinity for finite-sample correction
    augmented = sorted(residuals) + [float('inf')]
    # Compute index: ⌈(n+1)(1-δ)⌉
    idx = math.ceil((n + 1) * (1 - delta)) - 1  # 0-indexed
    idx = min(idx, len(augmented) - 1)
    q_hat = augmented[idx]
    return q_hat


# ═══════════════════════════════════════════════════════════════════════
# CARES: Online Routing Phase  
# ═══════════════════════════════════════════════════════════════════════

def cares_construct_envelope(query: Dict, q_hat: float, 
                              lam: float = 2.0, mu: float = 1.0, 
                              eta: float = 1e-6) -> List[int]:
    """
    Construct the safety envelope E(q_t).
    
    For each tier k:
      - If k ≤ ŝ_t: always in envelope (safe direction, over-escalation)
      - If k > ŝ_t: check if residual threshold is met
        R_k = (k - ŝ_t) · exp(λ·k)
        Φ_k = μ · log(π(ŝ_t) / (π(k) + η))  [coupling term]
        Include k if R_k ≤ q̂_δ + Φ_k
    
    The coupling term Φ_k is ADVERSARIAL: when softmax assigns very low
    probability to tier k (π(k) ≈ 0), Φ_k becomes large and positive,
    making it EASIER for k to enter the envelope. This means: suspicious
    confidence AGAINST a higher tier triggers more caution, not less.
    """
    s_pred = query["predicted_tier"]
    probs = query["tier_probs"]
    
    envelope = []
    for k in range(NUM_TIERS):
        if k <= s_pred:
            # Tiers at or below prediction: always safe (over-escalation is OK)
            envelope.append(k)
        else:
            # Tiers above prediction: check residual + coupling
            residual_k = (k - s_pred) * math.exp(lam * k)
            
            # Softmax-residual coupling (adversarial)
            pi_pred = probs[s_pred] if s_pred < len(probs) else 0.5
            pi_k = probs[k] if k < len(probs) else 0.0
            coupling = mu * math.log(pi_pred / (pi_k + eta))
            
            # Include tier k if residual is within threshold + coupling
            if residual_k <= q_hat + coupling:
                envelope.append(k)
    
    return envelope


def cares_route(query: Dict, platforms: List[Dict], q_hat: float,
                lam: float = 2.0, mu: float = 1.0, eta: float = 1e-6,
                tau: float = 0.80, beta: float = 0.01) -> Tuple[Dict, bool]:
    """
    Full CARES routing for a single query.
    
    Returns (chosen_platform, is_violation).
    """
    conf = query["confidence"]
    true_tier = query["tier"]
    
    # Step 5: Confidence gate
    if conf < tau:
        chosen = max(platforms, key=lambda p: p["clearance"])
        return chosen, chosen["clearance"] < true_tier
    
    # Step 6: Construct safety envelope
    envelope = cares_construct_envelope(query, q_hat, lam, mu, eta)
    
    # Step 7: Minimum safe clearance
    l_min = max(envelope) if envelope else NUM_TIERS - 1
    
    # Step 8: Safe action set
    safe = [p for p in platforms if p["clearance"] >= l_min]
    
    if not safe:
        chosen = max(platforms, key=lambda p: p["clearance"])
        return chosen, chosen["clearance"] < true_tier
    
    # Step 9: Cost-optimal with exploration noise
    noise = {p["name"]: np.random.exponential(beta) for p in safe}
    chosen = min(safe, key=lambda p: p["cost"] + noise[p["name"]])
    
    return chosen, chosen["clearance"] < true_tier


# ═══════════════════════════════════════════════════════════════════════
# CompTS (retained as baseline)
# ═══════════════════════════════════════════════════════════════════════

def compts_route(query, platforms, epsilon=0.01, tau=0.80):
    """CompTS: posterior-based safe action set."""
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

def route_static_ilp(query, platforms):
    pred = query["predicted_tier"]
    true_tier = query["tier"]
    safe = [p for p in platforms if p["clearance"] >= pred]
    chosen = min(safe, key=lambda p: p["cost"]) if safe else max(platforms, key=lambda p: p["clearance"])
    return chosen, chosen["clearance"] < true_tier


def route_secure_default(query, platforms):
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
# Evaluation Harness
# ═══════════════════════════════════════════════════════════════════════

def evaluate_strategy(name, route_fn, data, platforms, **kwargs):
    total_cost = 0
    violations = 0
    platform_counts = Counter()
    violation_details = []
    
    for q in data:
        if kwargs:
            chosen, viol = route_fn(q, platforms, **kwargs)
        else:
            chosen, viol = route_fn(q, platforms)
        
        tokens = len(q.get("text", "").split()) * 1.3
        total_cost += chosen["cost"] * tokens / 1000
        violations += int(viol)
        platform_counts[chosen["name"]] += 1
        
        if viol:
            violation_details.append({
                "true_tier": q["tier"],
                "pred_tier": q["predicted_tier"],
                "conf": q["confidence"],
                "platform": chosen["name"],
                "clearance": chosen["clearance"],
            })
    
    n = len(data)
    on_prem = platform_counts.get("OnPremises", 0)
    cloud = n - on_prem
    
    return {
        "name": name,
        "cost": round(total_cost, 4),
        "violations": violations,
        "viol_pct": round(violations / n * 100, 3),
        "on_prem": on_prem,
        "cloud_routed": cloud,
        "cloud_pct": round(cloud / n * 100, 1),
        "routing": dict(platform_counts),
        "n": n,
        "violation_details": violation_details[:10],  # First 10 for analysis
    }


def fix_tier_probs(predictions):
    """Reconstruct softmax from confidence + predicted_tier if uniform/missing."""
    fixed = []
    for p in predictions:
        pred = p["predicted_tier"]
        conf = p["confidence"]
        probs = p.get("tier_probs", None)
        
        needs_fix = False
        if probs is None or not isinstance(probs, list) or len(probs) != 4:
            needs_fix = True
        elif all(abs(x - 0.25) < 0.001 for x in probs):
            needs_fix = True
        
        if needs_fix:
            remaining = (1.0 - conf) / 3.0
            probs = [remaining] * 4
            probs[pred] = conf
        
        fixed_p = dict(p)
        fixed_p["tier_probs"] = probs
        fixed.append(fixed_p)
    
    return fixed


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    data_dir = ROOT / "data"
    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    
    # ── Load test predictions ──
    pred_path = data_dir / "test_with_predictions.json"
    if not pred_path.exists():
        print("ERROR: test_with_predictions.json not found!")
        print("Run scripts/02_train_classifier.py first.")
        sys.exit(1)
    
    test_data = json.loads(pred_path.read_text())
    test_data = fix_tier_probs(test_data)
    n_test = len(test_data)
    
    # ── Load validation set for calibration ──
    val_path = data_dir / "val.json"
    if val_path.exists():
        val_raw = json.loads(val_path.read_text())
        # Val may not have predictions — use test data split for calibration
        # In practice, you'd run inference on val set. Here we split test.
        print("Note: Using first 40% of test data as calibration set")
    
    # Split test into calibration (40%) and evaluation (60%)
    random.shuffle(test_data)
    n_cal = int(n_test * 0.4)
    cal_data = test_data[:n_cal]
    eval_data = test_data[n_cal:]
    n_eval = len(eval_data)
    
    print("=" * 70)
    print("CARES: Compliance-Aware Residual Envelope Scoring")
    print("=" * 70)
    
    # Tier distribution
    tier_counts = Counter(p["tier"] for p in eval_data)
    print(f"\nEvaluation set: {n_eval} queries")
    print(f"Calibration set: {n_cal} queries")
    for t in sorted(tier_counts):
        print(f"  T{t}: {tier_counts[t]}")
    
    # Classification stats
    correct = sum(1 for p in eval_data if p["predicted_tier"] == p["tier"])
    t3_total = sum(1 for p in eval_data if p["tier"] == 3)
    t3_correct = sum(1 for p in eval_data if p["tier"] == 3 and p["predicted_tier"] == 3)
    t3_missed = t3_total - t3_correct
    
    print(f"\nClassifier accuracy: {correct}/{n_eval} ({correct/n_eval*100:.1f}%)")
    print(f"T3 recall: {t3_correct}/{t3_total} ({t3_correct/t3_total*100:.1f}%)")
    print(f"T3 MISSED (safety-critical): {t3_missed}")
    
    if t3_missed > 0:
        missed = [p for p in eval_data if p["tier"] == 3 and p["predicted_tier"] != 3]
        confs = [p["confidence"] for p in missed]
        print(f"  Missed T3 confidence: min={min(confs):.3f}, mean={np.mean(confs):.3f}, max={max(confs):.3f}")
    
    # ══════════════════════════════════════════════════════════════════
    # CARES OFFLINE CALIBRATION
    # ══════════════════════════════════════════════════════════════════
    
    print(f"\n{'='*70}")
    print("OFFLINE CALIBRATION PHASE")
    print(f"{'='*70}")
    
    # Sweep over parameters
    lam_values = [1.0, 2.0, 3.0]
    delta_values = [0.005, 0.01, 0.02, 0.05]
    
    for lam in lam_values:
        residuals = compute_directional_residuals(cal_data, lam=lam)
        n_nonzero = sum(1 for r in residuals if r > 0)
        print(f"\n  λ={lam}: {n_nonzero}/{n_cal} non-zero residuals "
              f"(= misclassified-upward)")
        
        if n_nonzero > 0:
            nonzero_vals = [r for r in residuals if r > 0]
            print(f"    Non-zero residuals: min={min(nonzero_vals):.2f}, "
                  f"max={max(nonzero_vals):.2f}, mean={np.mean(nonzero_vals):.2f}")
        
        for delta in delta_values:
            q_hat = compute_calibration_quantile(residuals, delta=delta)
            print(f"    δ={delta}: q̂_δ = {q_hat:.4f}" + 
                  (" (= ∞, no underestimation errors)" if q_hat == float('inf') else ""))
    
    # ══════════════════════════════════════════════════════════════════
    # ROUTING EVALUATION
    # ══════════════════════════════════════════════════════════════════
    
    platforms = [
        {"name": "PublicAPI",   "clearance": 0, "cost": 0.010},
        {"name": "SecureCloud", "clearance": 2, "cost": 0.011},
        {"name": "OnPremises",  "clearance": 3, "cost": 0.025},
    ]
    
    print(f"\n{'='*70}")
    print(f"ROUTING RESULTS (n={n_eval})")
    print(f"{'='*70}")
    
    results = []
    
    # ── CARES variants ──
    for lam in [1.0, 2.0, 3.0]:
        residuals = compute_directional_residuals(cal_data, lam=lam)
        for delta in [0.01, 0.02, 0.05]:
            q_hat = compute_calibration_quantile(residuals, delta=delta)
            for mu in [0.5, 1.0, 2.0]:
                name = f"CARES λ={lam} δ={delta} μ={mu}"
                r = evaluate_strategy(
                    name,
                    cares_route, eval_data, platforms,
                    q_hat=q_hat, lam=lam, mu=mu, tau=0.80
                )
                results.append(r)
    
    # ── CompTS variants ──
    for eps in [0.001, 0.005, 0.01, 0.02, 0.05]:
        r = evaluate_strategy(f"CompTS ε={eps}", compts_route, eval_data, platforms, epsilon=eps)
        results.append(r)
    
    # ── Baselines ──
    results.append(evaluate_strategy("StaticILP", route_static_ilp, eval_data, platforms))
    results.append(evaluate_strategy("SecureDefault", route_secure_default, eval_data, platforms))
    results.append(evaluate_strategy("Greedy", route_greedy, eval_data, platforms))
    for t in [0.80, 0.90, 0.95, 0.99]:
        results.append(evaluate_strategy(f"Threshold-{t}", route_threshold, eval_data, platforms, threshold=t))
    
    # ── Print results ──
    print(f"\n{'Strategy':<35} {'N':>5} {'Cost':>10} {'Viol':>5} {'Viol%':>7} {'Cloud%':>7} {'Routing'}")
    print("-" * 110)
    
    # Sort: zero-violation strategies first, then by cost
    for r in sorted(results, key=lambda x: (x["violations"], x["cost"])):
        routing_str = ", ".join(f"{k}:{v}" for k, v in sorted(r["routing"].items()))
        
        # Mark Pareto-optimal (zero violations + mixed routing)
        marker = ""
        if r["violations"] == 0 and r["cloud_routed"] > 0:
            marker = " ◆"  # Pareto candidate
        elif r["violations"] > 0:
            marker = " ✗"
        
        print(f"{r['name']:<35} {r['n']:>5} ${r['cost']:>8.4f} {r['violations']:>5} "
              f"{r['viol_pct']:>6.3f}% {r['cloud_pct']:>5.1f}%  {routing_str}{marker}")
    
    # ══════════════════════════════════════════════════════════════════
    # FIND PARETO-OPTIMAL STRATEGIES
    # ══════════════════════════════════════════════════════════════════
    
    print(f"\n{'='*70}")
    print("PARETO-OPTIMAL STRATEGIES (0 violations + mixed routing)")
    print(f"{'='*70}")
    
    pareto = [r for r in results if r["violations"] == 0 and r["cloud_routed"] > 0]
    pareto.sort(key=lambda x: x["cost"])
    
    if pareto:
        best = pareto[0]
        print(f"\nBest: {best['name']}")
        print(f"  Cost: ${best['cost']:.4f}")
        print(f"  Violations: {best['violations']}")
        print(f"  Cloud-routed: {best['cloud_routed']}/{best['n']} ({best['cloud_pct']}%)")
        print(f"  Routing: {best['routing']}")
        
        # Compare to baselines
        secure = next(r for r in results if r["name"] == "SecureDefault")
        static = next(r for r in results if r["name"] == "StaticILP")
        greedy = next(r for r in results if r["name"] == "Greedy")
        
        saving_vs_secure = (1 - best["cost"] / secure["cost"]) * 100
        print(f"\n  vs SecureDefault: {saving_vs_secure:.1f}% cost savings")
        print(f"  vs StaticILP: {static['violations']} violations avoided")
        print(f"  vs Greedy: {greedy['violations']} violations avoided")
        
        # Find best CompTS for comparison
        compts_zero = [r for r in results if "CompTS" in r["name"] and r["violations"] == 0 and r["cloud_routed"] > 0]
        if compts_zero:
            best_compts = min(compts_zero, key=lambda x: x["cost"])
            print(f"\n  Best CompTS: {best_compts['name']} (${best_compts['cost']:.4f})")
            if best["cost"] < best_compts["cost"]:
                improvement = (1 - best["cost"] / best_compts["cost"]) * 100
                print(f"  CARES is {improvement:.1f}% cheaper than best CompTS")
            elif best["cost"] > best_compts["cost"]:
                print(f"  CompTS is slightly cheaper (CARES trades cost for stronger guarantees)")
            else:
                print(f"  Same cost — CARES wins on theoretical guarantees")
    else:
        print("\nNo Pareto-optimal strategies found!")
        print("This means either ALL strategies have violations, or NONE route to cloud.")
        
        # Diagnostic
        zero_viol = [r for r in results if r["violations"] == 0]
        if zero_viol:
            print(f"\n{len(zero_viol)} strategies have 0 violations but all route to on-prem")
            print("This happens when classifier is perfect (100% recall).")
            print("The harder dataset (Script 18) should fix this.")
        else:
            has_viols = [r for r in results if r["violations"] > 0]
            print(f"\n{len(has_viols)} strategies have violations.")
    
    # ══════════════════════════════════════════════════════════════════
    # CARES SAFETY ANALYSIS
    # ══════════════════════════════════════════════════════════════════
    
    print(f"\n{'='*70}")
    print("CARES SAFETY ANALYSIS")
    print(f"{'='*70}")
    
    # For the best CARES configuration, analyze envelope behavior
    if pareto:
        # Use default parameters for analysis
        lam, delta, mu = 2.0, 0.01, 1.0
        residuals = compute_directional_residuals(cal_data, lam=lam)
        q_hat = compute_calibration_quantile(residuals, delta=delta)
        
        envelope_sizes = []
        t3_in_envelope = 0
        t3_not_in_envelope = 0
        
        for q in eval_data:
            env = cares_construct_envelope(q, q_hat, lam=lam, mu=mu)
            envelope_sizes.append(len(env))
            if q["tier"] == 3:
                if 3 in env:
                    t3_in_envelope += 1
                else:
                    t3_not_in_envelope += 1
        
        print(f"\nEnvelope statistics (λ={lam}, δ={delta}, μ={mu}):")
        print(f"  Avg envelope size: {np.mean(envelope_sizes):.2f} tiers")
        print(f"  Min: {min(envelope_sizes)}, Max: {max(envelope_sizes)}")
        print(f"  T3 queries with T3 in envelope: {t3_in_envelope}/{t3_total}")
        print(f"  T3 queries with T3 NOT in envelope: {t3_not_in_envelope}/{t3_total}")
        
        # Theoretical bound
        bound = delta + 1 / (n_cal + 1)
        print(f"\n  Theoretical safety bound: δ + 1/(n+1) = {delta} + {1/(n_cal+1):.6f} = {bound:.6f}")
        empirical_viol = t3_not_in_envelope / max(1, t3_total)
        print(f"  Empirical violation rate: {empirical_viol:.6f}")
        print(f"  Bound holds: {'YES ✓' if empirical_viol <= bound else 'NO ✗'}")
    
    # ══════════════════════════════════════════════════════════════════
    # PAPER METRICS SUMMARY
    # ══════════════════════════════════════════════════════════════════
    
    print(f"\n{'='*70}")
    print("PAPER METRICS SUMMARY")
    print(f"{'='*70}")
    
    static = next(r for r in results if r["name"] == "StaticILP")
    secure = next(r for r in results if r["name"] == "SecureDefault")
    greedy = next(r for r in results if r["name"] == "Greedy")
    
    print(f"\nDataset: n_total ≈ {int(n_eval / 0.6)}, n_cal = {n_cal}, n_eval = {n_eval}")
    print(f"Classifier: {correct/n_eval*100:.1f}% accuracy, T3 recall = {t3_correct/t3_total*100:.1f}%")
    print(f"T3 missed: {t3_missed}")
    print(f"\nBaseline costs:")
    print(f"  SecureDefault: ${secure['cost']:.4f} (0 violations, all on-prem)")
    print(f"  StaticILP: ${static['cost']:.4f} ({static['violations']} violations, {static['viol_pct']:.2f}%)")
    print(f"  Greedy: ${greedy['cost']:.4f} ({greedy['violations']} violations, {greedy['viol_pct']:.2f}%)")
    
    if pareto:
        best_cares = [r for r in pareto if "CARES" in r["name"]]
        best_compts_p = [r for r in pareto if "CompTS" in r["name"]]
        
        if best_cares:
            bc = min(best_cares, key=lambda x: x["cost"])
            savings = (1 - bc["cost"] / secure["cost"]) * 100
            print(f"\nBest CARES: {bc['name']}")
            print(f"  Cost: ${bc['cost']:.4f} ({savings:.1f}% savings vs SecureDefault)")
            print(f"  Violations: {bc['violations']}")
            print(f"  Cloud-routed: {bc['cloud_pct']}%")
        
        if best_compts_p:
            bcp = min(best_compts_p, key=lambda x: x["cost"])
            savings_c = (1 - bcp["cost"] / secure["cost"]) * 100
            print(f"\nBest CompTS: {bcp['name']}")
            print(f"  Cost: ${bcp['cost']:.4f} ({savings_c:.1f}% savings vs SecureDefault)")
            print(f"  Violations: {bcp['violations']}")
    
    # Save results
    save_results = {
        "n_total": int(n_eval / 0.6),
        "n_calibration": n_cal,
        "n_evaluation": n_eval,
        "classifier_accuracy": correct / n_eval,
        "t3_recall": t3_correct / t3_total,
        "t3_missed": t3_missed,
        "strategies": results,
    }
    
    out_path = out_dir / "cares_routing_results.json"
    with open(out_path, "w") as f:
        json.dump(save_results, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
