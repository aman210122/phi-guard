#!/usr/bin/env python3
"""
12_distribution_shift.py — Online Evaluation with Distribution Shift

PURPOSE:
  Prove the Dirichlet posterior MATTERS by simulating a deployment where
  the query distribution shifts mid-stream. CompTS should adapt, 
  CompTS-NoPosterior should not.

SCENARIO:
  Phase 1 (queries 1-2000): 70% T0/T1, 30% T2/T3 (normal operations)
  Phase 2 (queries 2001-5000): 30% T0/T1, 70% T2/T3 (flu season, surge in PHI)
  
  CompTS should detect the shift via posterior updates and increase conservatism.
  CompTS-NoPosterior uses fixed priors and can't adapt.

USAGE:
  python scripts/12_distribution_shift.py
"""

import json
import random
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent


def load_predictions(data_dir):
    pred_path = data_dir / "test_with_predictions.json"
    if not pred_path.exists():
        print(f"ERROR: {pred_path} not found. Run 02_train_classifier.py first.")
        sys.exit(1)
    return json.loads(pred_path.read_text())


def simulate_routing(query, alpha, epsilon=0.01, tau=0.80):
    """Route a single query using CompTS logic."""
    # Platform costs per token
    platforms = [
        {"name": "public", "clearance": 0, "cost": 0.010},
        {"name": "cloud", "clearance": 2, "cost": 0.011},
        {"name": "onprem", "clearance": 3, "cost": 0.025},
    ]
    
    pred_tier = query["predicted_tier"]
    confidence = query["confidence"]
    true_tier = query["tier"]
    probs = query.get("tier_probs", [0.25, 0.25, 0.25, 0.25])
    
    # Fallback check
    if confidence < tau:
        chosen = platforms[-1]  # on-prem
        violation = chosen["clearance"] < true_tier
        return chosen, violation, "fallback"
    
    # Posterior violation check for each platform
    safe_platforms = []
    for p in platforms:
        # Probability that true tier exceeds platform clearance
        viol_prob = sum(probs[t] * alpha[t] / sum(alpha)
                        for t in range(4) if t > p["clearance"])
        if viol_prob <= epsilon:
            safe_platforms.append(p)
    
    if not safe_platforms:
        chosen = platforms[-1]
        violation = chosen["clearance"] < true_tier
        return chosen, violation, "no_safe"
    
    # Pick cheapest safe platform
    chosen = min(safe_platforms, key=lambda p: p["cost"])
    violation = chosen["clearance"] < true_tier
    return chosen, violation, "normal"


def run_online_simulation(predictions, n_total=5000, shift_point=2000, seed=42):
    """Simulate online deployment with distribution shift."""
    random.seed(seed)
    np.random.seed(seed)
    
    by_tier = defaultdict(list)
    for p in predictions:
        by_tier[p["tier"]].append(p)
    
    for t in by_tier:
        random.shuffle(by_tier[t])
    
    # Generate query stream with distribution shift
    stream = []
    tier_indices = {t: 0 for t in range(4)}
    
    for i in range(n_total):
        if i < shift_point:
            # Phase 1: mostly low-sensitivity
            tier_probs = [0.35, 0.35, 0.15, 0.15]
        else:
            # Phase 2: surge in PHI queries
            tier_probs = [0.10, 0.10, 0.30, 0.50]
        
        tier = np.random.choice(4, p=tier_probs)
        pool = by_tier[tier]
        idx = tier_indices[tier] % len(pool)
        tier_indices[tier] += 1
        stream.append(pool[idx])
    
    # Strategy 1: CompTS with posterior updates
    alpha_adaptive = np.ones(4)  # Dirichlet prior
    compts_results = {"costs": [], "violations": [], "phases": []}
    
    for i, query in enumerate(stream):
        phase = 1 if i < shift_point else 2
        chosen, violation, reason = simulate_routing(
            query, alpha_adaptive, epsilon=0.01
        )
        compts_results["costs"].append(chosen["cost"])
        compts_results["violations"].append(int(violation))
        compts_results["phases"].append(phase)
        
        # Update posterior with observed tier
        pred = query["predicted_tier"]
        alpha_adaptive[pred] += 1
    
    # Strategy 2: CompTS WITHOUT posterior updates (fixed prior)
    alpha_fixed = np.ones(4)  # Never updated
    noposterior_results = {"costs": [], "violations": [], "phases": []}
    
    for i, query in enumerate(stream):
        phase = 1 if i < shift_point else 2
        chosen, violation, reason = simulate_routing(
            query, alpha_fixed, epsilon=0.01
        )
        noposterior_results["costs"].append(chosen["cost"])
        noposterior_results["violations"].append(int(violation))
        noposterior_results["phases"].append(phase)
    
    # Strategy 3: StaticILP (no posterior, no adaptation)
    static_results = {"costs": [], "violations": [], "phases": []}
    platforms = [
        {"name": "public", "clearance": 0, "cost": 0.010},
        {"name": "cloud", "clearance": 2, "cost": 0.011},
        {"name": "onprem", "clearance": 3, "cost": 0.025},
    ]
    
    for i, query in enumerate(stream):
        phase = 1 if i < shift_point else 2
        pred = query["predicted_tier"]
        # Route to cheapest platform that covers predicted tier
        safe = [p for p in platforms if p["clearance"] >= pred]
        chosen = min(safe, key=lambda p: p["cost"]) if safe else platforms[-1]
        violation = chosen["clearance"] < query["tier"]
        static_results["costs"].append(chosen["cost"])
        static_results["violations"].append(int(violation))
        static_results["phases"].append(phase)
    
    return stream, compts_results, noposterior_results, static_results


def analyze_results(name, results, shift_point):
    phase1 = [i for i in range(len(results["costs"])) if i < shift_point]
    phase2 = [i for i in range(len(results["costs"])) if i >= shift_point]
    
    p1_cost = sum(results["costs"][i] for i in phase1) / len(phase1)
    p2_cost = sum(results["costs"][i] for i in phase2) / len(phase2)
    p1_viol = sum(results["violations"][i] for i in phase1) / len(phase1)
    p2_viol = sum(results["violations"][i] for i in phase2) / len(phase2)
    total_viol = sum(results["violations"]) / len(results["violations"])
    
    return {
        "name": name,
        "phase1_avg_cost": round(p1_cost, 5),
        "phase2_avg_cost": round(p2_cost, 5),
        "phase1_viol_rate": round(p1_viol, 4),
        "phase2_viol_rate": round(p2_viol, 4),
        "total_viol_rate": round(total_viol, 4),
        "total_cost": round(sum(results["costs"]), 4),
        "cost_change": round((p2_cost - p1_cost) / max(p1_cost, 0.0001) * 100, 1),
    }


def main():
    data_dir = ROOT / "data"
    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("Distribution Shift Evaluation")
    print("=" * 60)
    
    predictions = load_predictions(data_dir)
    print(f"Loaded {len(predictions)} test predictions")
    
    n_total = 5000
    shift_point = 2000
    
    print(f"\nSimulating {n_total} queries, shift at query {shift_point}")
    print(f"  Phase 1 (1-{shift_point}): 70% T0/T1, 30% T2/T3")
    print(f"  Phase 2 ({shift_point+1}-{n_total}): 20% T0/T1, 80% T2/T3")
    
    stream, compts, noposterior, static = run_online_simulation(
        predictions, n_total, shift_point
    )
    
    # Analyze
    strategies = [
        ("CompTS (adaptive)", compts),
        ("CompTS-NoPosterior", noposterior),
        ("StaticILP", static),
    ]
    
    print(f"\n{'Strategy':<25} {'P1 Cost':>10} {'P2 Cost':>10} {'Change':>8} {'P1 Viol%':>10} {'P2 Viol%':>10} {'Total Viol%':>12}")
    print("-" * 87)
    
    summaries = []
    for name, results in strategies:
        s = analyze_results(name, results, shift_point)
        summaries.append(s)
        print(f"{name:<25} ${s['phase1_avg_cost']:.4f}   ${s['phase2_avg_cost']:.4f}   "
              f"{s['cost_change']:>+6.1f}%   {s['phase1_viol_rate']*100:>8.2f}%   "
              f"{s['phase2_viol_rate']*100:>8.2f}%   {s['total_viol_rate']*100:>10.2f}%")
    
    # Key finding
    compts_s = summaries[0]
    noposterior_s = summaries[1]
    static_s = summaries[2]
    
    print(f"\n{'='*60}")
    print("KEY FINDING")
    print(f"{'='*60}")
    
    if compts_s["phase2_viol_rate"] < noposterior_s["phase2_viol_rate"]:
        print(f"✓ CompTS adapts: Phase 2 violations {compts_s['phase2_viol_rate']*100:.2f}% "
              f"vs NoPosterior {noposterior_s['phase2_viol_rate']*100:.2f}%")
    elif compts_s["phase2_viol_rate"] == noposterior_s["phase2_viol_rate"]:
        print(f"= Both achieve same violations in Phase 2, but CompTS cost adapts:")
        print(f"  CompTS cost change: {compts_s['cost_change']:+.1f}%")
        print(f"  NoPosterior cost change: {noposterior_s['cost_change']:+.1f}%")
    
    print(f"\nStaticILP Phase 2 violations: {static_s['phase2_viol_rate']*100:.2f}% "
          f"(no safety mechanism)")
    
    # Window analysis: violations in sliding windows
    window = 200
    print(f"\nSliding window violation rate (window={window}):")
    for name, results in strategies:
        viols = results["violations"]
        windows = []
        for start in range(0, len(viols) - window, window // 2):
            w = viols[start:start + window]
            windows.append(sum(w) / len(w))
        max_window = max(windows) if windows else 0
        print(f"  {name:<25}: max window viol = {max_window*100:.2f}%")
    
    # Save
    output = {
        "n_total": n_total,
        "shift_point": shift_point,
        "strategies": summaries,
    }
    with open(out_dir / "distribution_shift_results.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nSaved: {out_dir / 'distribution_shift_results.json'}")


if __name__ == "__main__":
    main()
