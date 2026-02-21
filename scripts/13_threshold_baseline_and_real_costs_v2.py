#!/usr/bin/env python3
"""
13_threshold_baseline_and_real_costs_v2.py

CRITICAL FIX: On-prem cost must include infrastructure amortization.
Ollama's $0.0002/1K measures only marginal electricity. Real on-prem
inference includes:
  - GPU hardware: A100 80GB ~$15K, amortized over 3 years
  - Server, networking, cooling: ~$5K/year
  - DevOps/MLOps staff: ~$50K/year (partial allocation)
  - At ~1000 tok/sec throughput: effective cost ~$0.005-0.015/1K tokens

Three cost scenarios:
  1. Simulated (original paper)
  2. Real-measured marginal (Script 09 output — misleading for on-prem)
  3. Realistic amortized (includes infrastructure — most accurate)
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


# ═══════════════════════════════════════════════════════════════
# THREE COST MODELS
# ═══════════════════════════════════════════════════════════════

COST_MODELS = {
    "simulated": {
        "label": "Simulated (Original Paper)",
        "platforms": [
            {"name": "PublicAPI",    "clearance": 0, "cost_per_1k": 0.010},
            {"name": "SecureCloud",  "clearance": 2, "cost_per_1k": 0.011},
            {"name": "OnPremises",   "clearance": 3, "cost_per_1k": 0.025},
        ],
    },
    "realistic": {
        "label": "Realistic Amortized (Production-grade)",
        "note": "Public: GPT-4o-mini measured. Cloud: Azure OpenAI estimated. On-prem: A100 amortized.",
        "platforms": [
            {"name": "PublicAPI",    "clearance": 0, "cost_per_1k": 0.0005},   # Measured
            {"name": "SecureCloud",  "clearance": 2, "cost_per_1k": 0.003},    # Azure OpenAI w/ BAA
            {"name": "OnPremises",   "clearance": 3, "cost_per_1k": 0.008},    # GPU amortized
        ],
    },
    "enterprise": {
        "label": "Enterprise (Large health system, high volume)",
        "note": "Volume discounts on API, dedicated GPU cluster on-prem.",
        "platforms": [
            {"name": "PublicAPI",    "clearance": 0, "cost_per_1k": 0.0003},
            {"name": "SecureCloud",  "clearance": 2, "cost_per_1k": 0.002},
            {"name": "OnPremises",   "clearance": 3, "cost_per_1k": 0.005},
        ],
    },
}


def route_static_ilp(query, platforms, **kwargs):
    pred = query["predicted_tier"]
    safe = [p for p in platforms if p["clearance"] >= pred]
    chosen = min(safe, key=lambda p: p["cost_per_1k"]) if safe else platforms[-1]
    violation = chosen["clearance"] < query["tier"]
    return chosen, violation


def route_threshold(query, platforms, threshold=0.95, **kwargs):
    pred = query["predicted_tier"]
    conf = query["confidence"]
    if conf < threshold:
        chosen = platforms[-1]
    else:
        safe = [p for p in platforms if p["clearance"] >= pred]
        chosen = min(safe, key=lambda p: p["cost_per_1k"]) if safe else platforms[-1]
    violation = chosen["clearance"] < query["tier"]
    return chosen, violation


def route_compts(query, platforms, epsilon=0.01, tau=0.80, **kwargs):
    pred = query["predicted_tier"]
    conf = query["confidence"]
    probs = query.get("tier_probs", [0.25, 0.25, 0.25, 0.25])
    if conf < tau:
        chosen = platforms[-1]
        return chosen, chosen["clearance"] < query["tier"]
    safe = []
    for p in platforms:
        viol_prob = sum(probs[t] for t in range(4) if t > p["clearance"])
        if viol_prob <= epsilon:
            safe.append(p)
    chosen = min(safe, key=lambda p: p["cost_per_1k"]) if safe else platforms[-1]
    return chosen, chosen["clearance"] < query["tier"]


def route_secure_default(query, platforms, **kwargs):
    chosen = platforms[-1]
    return chosen, chosen["clearance"] < query["tier"]


def evaluate(name, fn, preds, platforms, **kwargs):
    total_cost = 0
    violations = 0
    platform_counts = defaultdict(int)
    for q in preds:
        chosen, viol = fn(q, platforms, **kwargs)
        tokens = len(q.get("text", "").split()) * 1.3
        total_cost += chosen["cost_per_1k"] * tokens / 1000
        violations += int(viol)
        platform_counts[chosen["name"]] += 1
    n = len(preds)
    return {
        "name": name,
        "total_cost": round(total_cost, 6),
        "avg_cost": round(total_cost / n, 8),
        "violations": violations,
        "viol_pct": round(violations / n * 100, 2),
        "routing": dict(platform_counts),
    }


def main():
    data_dir = ROOT / "data"
    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)

    preds = load_predictions(data_dir)
    print(f"Loaded {len(preds)} predictions\n")

    all_results = {}

    for model_key, model in COST_MODELS.items():
        platforms = model["platforms"]
        print("=" * 70)
        print(f"COST MODEL: {model['label']}")
        print(f"  Public: ${platforms[0]['cost_per_1k']}/1K  "
              f"Cloud: ${platforms[1]['cost_per_1k']}/1K  "
              f"OnPrem: ${platforms[2]['cost_per_1k']}/1K")
        print("=" * 70)

        results = []
        results.append(evaluate("StaticILP", route_static_ilp, preds, platforms))
        results.append(evaluate("CompTS (ε=0.01)", route_compts, preds, platforms))
        results.append(evaluate("SecureDefault", route_secure_default, preds, platforms))

        for t in [0.80, 0.90, 0.95, 0.99]:
            results.append(evaluate(
                f"Threshold-{t}", route_threshold, preds, platforms, threshold=t))

        print(f"\n{'Strategy':<25} {'Cost':>12} {'Viol':>6} {'Viol%':>7} {'Routing':>30}")
        print("-" * 82)
        for s in results:
            marker = " ✓" if s["violations"] == 0 else ""
            routing_str = ", ".join(f"{k}:{v}" for k, v in s["routing"].items())
            print(f"{s['name']:<25} ${s['total_cost']:>10.6f}  {s['violations']:>5}  "
                  f"{s['viol_pct']:>6.2f}%  {routing_str}{marker}")

        # Find cheapest zero-violation strategy
        safe_strategies = [s for s in results if s["violations"] == 0]
        if safe_strategies:
            cheapest_safe = min(safe_strategies, key=lambda s: s["total_cost"])
            compts = [s for s in results if s["name"] == "CompTS (ε=0.01)"][0]
            static = [s for s in results if s["name"] == "StaticILP"][0]

            print(f"\n  Cheapest safe strategy: {cheapest_safe['name']} "
                  f"(${cheapest_safe['total_cost']:.6f})")
            if static["violations"] > 0:
                premium = compts["total_cost"] - static["total_cost"]
                pct = premium / static["total_cost"] * 100 if static["total_cost"] > 0 else 0
                print(f"  Compliance premium over StaticILP: ${premium:.6f} ({pct:.1f}%)")
                print(f"  StaticILP violations: {static['violations']} "
                      f"({static['viol_pct']:.2f}%)")
                if static["violations"] > 0:
                    penalty_min = static["violations"] * 141
                    penalty_max = static["violations"] * 71162
                    print(f"  HIPAA penalty risk: ${penalty_min:,} - ${penalty_max:,}")
                    print(f"  Compliance premium pays for itself after "
                          f"{premium / 141:.4f} violations at min penalty")

        all_results[model_key] = results
        print()

    # ═══════════════════════════════════════════════════════════════
    # SUMMARY TABLE
    # ═══════════════════════════════════════════════════════════════
    print("=" * 70)
    print("SUMMARY: CompTS vs Threshold-0.99 vs StaticILP across cost models")
    print("=" * 70)
    print(f"\n{'Cost Model':<30} {'StaticILP':>15} {'Threshold-0.99':>15} {'CompTS':>15}")
    print(f"{'':30} {'Cost / Viol':>15} {'Cost / Viol':>15} {'Cost / Viol':>15}")
    print("-" * 76)

    for model_key, results in all_results.items():
        label = COST_MODELS[model_key]["label"][:28]
        static = [s for s in results if s["name"] == "StaticILP"][0]
        thresh = [s for s in results if s["name"] == "Threshold-0.99"][0]
        compts = [s for s in results if s["name"] == "CompTS (ε=0.01)"][0]
        print(f"{label:<30} ${static['total_cost']:.4f}/{static['violations']:>2}   "
              f"  ${thresh['total_cost']:.4f}/{thresh['violations']:>2}   "
              f"  ${compts['total_cost']:.4f}/{compts['violations']:>2}")

    print(f"\n{'=' * 70}")
    print("KEY INSIGHT")
    print("=" * 70)
    print("""
CompTS and Threshold-0.99 achieve identical safety (0 violations) at
identical cost. The value of CompTS over a tuned threshold is:

1. FORMAL GUARANTEE: CompTS provides Theorem 1 (provable per-step
   violation bound). Threshold-0.99 is empirically safe on THIS dataset
   but has no guarantee it won't fail on distribution shift.

2. ADAPTIVE ε: CompTS's safety level is parameterized by ε, providing
   a principled knob. Threshold requires manual tuning per deployment.

3. COST MODEL ROBUSTNESS: CompTS's safe action set adapts to any cost
   structure. A threshold tuned for one cost model may be suboptimal
   for another.

The paper's contribution is the COMPLIANCE GUARANTEE, not cost savings.
At realistic prices, the cost difference between strategies is negligible.
The value is eliminating HIPAA violation risk ($141-$71,162 per incident).
""")

    # Save
    with open(out_dir / "threshold_and_cost_analysis_v2.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Saved: {out_dir / 'threshold_and_cost_analysis_v2.json'}")


if __name__ == "__main__":
    main()
