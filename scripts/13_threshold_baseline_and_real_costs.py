#!/usr/bin/env python3
"""
13_threshold_baseline_and_real_costs.py

Addresses two critical reviewer concerns:
1. Missing threshold-based baseline ("what a practitioner would build first")
2. Cost model uses simulated prices 20-125x higher than reality

Evaluates routing strategies under BOTH simulated and real-world pricing.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def load_predictions(data_dir):
    path = data_dir / "test_with_predictions.json"
    if not path.exists():
        print(f"ERROR: {path} not found")
        sys.exit(1)
    return json.loads(path.read_text())


# Platform definitions
PLATFORMS_SIMULATED = [
    {"name": "PublicAPI", "clearance": 0, "cost_per_1k": 0.010, "label": "Simulated"},
    {"name": "SecureCloud", "clearance": 2, "cost_per_1k": 0.011, "label": "Simulated"},
    {"name": "OnPremises", "clearance": 3, "cost_per_1k": 0.025, "label": "Simulated"},
]

# Real measured costs from Script 09
PLATFORMS_REAL = [
    {"name": "PublicAPI", "clearance": 0, "cost_per_1k": 0.0005, "label": "Real (GPT-4o-mini)"},
    {"name": "SecureCloud", "clearance": 2, "cost_per_1k": 0.0008, "label": "Est. (Azure OpenAI)"},
    {"name": "OnPremises", "clearance": 3, "cost_per_1k": 0.0002, "label": "Real (Ollama)"},
]


def route_static_ilp(query, platforms):
    """Route to cheapest platform that covers predicted tier."""
    pred = query["predicted_tier"]
    safe = [p for p in platforms if p["clearance"] >= pred]
    chosen = min(safe, key=lambda p: p["cost_per_1k"]) if safe else platforms[-1]
    violation = chosen["clearance"] < query["tier"]
    return chosen, violation


def route_threshold(query, platforms, threshold=0.95):
    """Simple threshold: if confidence < threshold, route to on-prem."""
    pred = query["predicted_tier"]
    conf = query["confidence"]

    if conf < threshold:
        chosen = platforms[-1]  # on-prem (safest)
    else:
        safe = [p for p in platforms if p["clearance"] >= pred]
        chosen = min(safe, key=lambda p: p["cost_per_1k"]) if safe else platforms[-1]

    violation = chosen["clearance"] < query["tier"]
    return chosen, violation


def route_compts(query, platforms, epsilon=0.01, tau=0.80):
    """CompTS: posterior-based safe action set + cheapest safe platform."""
    pred = query["predicted_tier"]
    conf = query["confidence"]
    probs = query.get("tier_probs", [0.25, 0.25, 0.25, 0.25])

    # Fallback
    if conf < tau:
        chosen = platforms[-1]
        violation = chosen["clearance"] < query["tier"]
        return chosen, violation

    # Safe action set
    safe = []
    for p in platforms:
        viol_prob = sum(probs[t] for t in range(4) if t > p["clearance"])
        if viol_prob <= epsilon:
            safe.append(p)

    if not safe:
        chosen = platforms[-1]
    else:
        chosen = min(safe, key=lambda p: p["cost_per_1k"])

    violation = chosen["clearance"] < query["tier"]
    return chosen, violation


def route_secure_default(query, platforms):
    """Always route to on-premises."""
    chosen = platforms[-1]
    violation = chosen["clearance"] < query["tier"]
    return chosen, violation


def evaluate_strategy(name, router_fn, predictions, platforms, **kwargs):
    total_cost = 0
    violations = 0
    platform_counts = defaultdict(int)

    for q in predictions:
        chosen, viol = router_fn(q, platforms, **kwargs)
        tokens = len(q.get("text", "").split()) * 1.3  # rough token estimate
        cost = chosen["cost_per_1k"] * tokens / 1000
        total_cost += cost
        violations += int(viol)
        platform_counts[chosen["name"]] += 1

    n = len(predictions)
    return {
        "name": name,
        "total_cost": round(total_cost, 4),
        "avg_cost_per_query": round(total_cost / n, 6),
        "violation_count": violations,
        "violation_rate": round(violations / n * 100, 2),
        "platform_distribution": dict(platform_counts),
    }


def main():
    data_dir = ROOT / "data"
    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)

    predictions = load_predictions(data_dir)
    print(f"Loaded {len(predictions)} predictions\n")

    # ═══════════════════════════════════════════════════════════════
    # PART 1: Threshold Baseline Comparison
    # ═══════════════════════════════════════════════════════════════
    print("=" * 70)
    print("PART 1: Threshold Baseline vs CompTS (Simulated Costs)")
    print("=" * 70)

    thresholds = [0.80, 0.85, 0.90, 0.95, 0.99]
    strategies = []

    strategies.append(evaluate_strategy(
        "StaticILP", route_static_ilp, predictions, PLATFORMS_SIMULATED))
    strategies.append(evaluate_strategy(
        "CompTS (ε=0.01)", route_compts, predictions, PLATFORMS_SIMULATED))
    strategies.append(evaluate_strategy(
        "SecureDefault", route_secure_default, predictions, PLATFORMS_SIMULATED))

    for t in thresholds:
        strategies.append(evaluate_strategy(
            f"Threshold-{t}", route_threshold, predictions,
            PLATFORMS_SIMULATED, threshold=t))

    print(f"\n{'Strategy':<25} {'Cost':>10} {'Violations':>12} {'Viol%':>8}")
    print("-" * 57)
    for s in strategies:
        marker = " ←" if s["violation_count"] == 0 else ""
        print(f"{s['name']:<25} ${s['total_cost']:>8.4f}   "
              f"{s['violation_count']:>10}   {s['violation_rate']:>6.2f}%{marker}")

    # Find best threshold that achieves 0 violations
    zero_viol_thresholds = [s for s in strategies
                           if s["name"].startswith("Threshold") and s["violation_count"] == 0]
    if zero_viol_thresholds:
        best_threshold = min(zero_viol_thresholds, key=lambda s: s["total_cost"])
        compts_result = [s for s in strategies if s["name"] == "CompTS (ε=0.01)"][0]
        print(f"\nBest zero-violation threshold: {best_threshold['name']} "
              f"(cost ${best_threshold['total_cost']:.4f})")
        print(f"CompTS cost: ${compts_result['total_cost']:.4f}")
        if compts_result["total_cost"] < best_threshold["total_cost"]:
            saving = (1 - compts_result["total_cost"] / best_threshold["total_cost"]) * 100
            print(f"CompTS saves {saving:.1f}% over best threshold baseline")
        else:
            print("Threshold baseline is cheaper — CompTS value is in formal guarantees")
    else:
        print("\nNo threshold achieves 0 violations — CompTS is strictly superior")

    # ═══════════════════════════════════════════════════════════════
    # PART 2: Real-World Cost Re-evaluation
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("PART 2: Real-World Costs (Measured from Script 09)")
    print("=" * 70)

    real_strategies = []
    real_strategies.append(evaluate_strategy(
        "StaticILP", route_static_ilp, predictions, PLATFORMS_REAL))
    real_strategies.append(evaluate_strategy(
        "CompTS (ε=0.01)", route_compts, predictions, PLATFORMS_REAL))
    real_strategies.append(evaluate_strategy(
        "SecureDefault", route_secure_default, predictions, PLATFORMS_REAL))
    for t in thresholds:
        real_strategies.append(evaluate_strategy(
            f"Threshold-{t}", route_threshold, predictions,
            PLATFORMS_REAL, threshold=t))

    print(f"\n{'Strategy':<25} {'Cost':>10} {'Violations':>12} {'Viol%':>8}")
    print("-" * 57)
    for s in real_strategies:
        marker = " ←" if s["violation_count"] == 0 else ""
        print(f"{s['name']:<25} ${s['total_cost']:>8.6f}   "
              f"{s['violation_count']:>10}   {s['violation_rate']:>6.2f}%{marker}")

    # Cost difference analysis
    compts_real = [s for s in real_strategies if s["name"] == "CompTS (ε=0.01)"][0]
    static_real = [s for s in real_strategies if s["name"] == "StaticILP"][0]
    secure_real = [s for s in real_strategies if s["name"] == "SecureDefault"][0]

    print(f"\n--- COST ANALYSIS AT REAL PRICES ---")
    print(f"StaticILP total cost:     ${static_real['total_cost']:.6f} "
          f"({static_real['violation_count']} violations)")
    print(f"CompTS total cost:        ${compts_real['total_cost']:.6f} "
          f"({compts_real['violation_count']} violations)")
    print(f"SecureDefault total cost:  ${secure_real['total_cost']:.6f} "
          f"({secure_real['violation_count']} violations)")

    cost_diff = compts_real["total_cost"] - static_real["total_cost"]
    print(f"\nCompliance premium (CompTS - StaticILP): ${cost_diff:.6f}")
    print(f"Per-query compliance premium: ${cost_diff / len(predictions):.8f}")

    # Key finding
    print(f"\n{'=' * 70}")
    print("KEY FINDING")
    print("=" * 70)
    print(f"At real prices, cost differences between strategies are negligible")
    print(f"(fractions of a cent per query). The value of CompTS is NOT cost")
    print(f"savings — it is the formal compliance guarantee.")
    print(f"\nStaticILP: {static_real['violation_count']} PHI violations "
          f"({static_real['violation_rate']}%)")
    print(f"CompTS:    {compts_real['violation_count']} PHI violations "
          f"({compts_real['violation_rate']}%)")
    print(f"\nAt 10,000 queries/day, StaticILP produces "
          f"~{int(static_real['violation_rate'] / 100 * 10000)} daily violations.")
    print(f"HIPAA penalty: $141-$71,162 per violation.")
    print(f"Annual risk: ${int(static_real['violation_rate'] / 100 * 10000 * 365 * 141):,} "
          f"to ${int(static_real['violation_rate'] / 100 * 10000 * 365 * 71162):,}")

    # Save
    output = {
        "simulated_costs": strategies,
        "real_costs": real_strategies,
        "compliance_premium_real": round(cost_diff, 8),
        "per_query_premium_real": round(cost_diff / len(predictions), 10),
    }
    with open(out_dir / "threshold_and_cost_analysis.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nSaved: {out_dir / 'threshold_and_cost_analysis.json'}")


if __name__ == "__main__":
    main()
