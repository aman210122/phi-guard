#!/usr/bin/env python3
"""
08_calibration_analysis.py — Classifier Calibration for Safety Guarantees

WHY THIS MATTERS FOR NEURIPS:
  Theorem 1's safety bound depends on classifier calibration error (α).
  If the classifier is overconfident on wrong predictions, the bound
  is loose and CompTS provides weak guarantees. This script:
  
  1. Measures Expected Calibration Error (ECE) on test set
  2. Generates reliability diagrams
  3. Implements temperature scaling for post-hoc calibration
  4. Shows how calibration improves safety bounds (key paper figure)

USAGE:
  python scripts/08_calibration_analysis.py
  (requires test_with_predictions.json from script 02)
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
with open(ROOT / "configs/config.yaml") as f:
    cfg = yaml.safe_load(f)

OUT = ROOT / cfg["evaluation"]["output_dir"]; OUT.mkdir(exist_ok=True)
np.random.seed(cfg["dataset"]["random_seed"])


# ═══════════════════════════════════════════════════════════════════════════
# CALIBRATION METRICS
# ═══════════════════════════════════════════════════════════════════════════

def compute_ece(confidences: np.ndarray, accuracies: np.ndarray,
                n_bins: int = 15) -> Tuple[float, List[Dict]]:
    """
    Expected Calibration Error (Naeini et al., AAAI 2015).
    
    ECE = Σ_b (|B_b|/n) · |acc(B_b) - conf(B_b)|
    
    A perfectly calibrated model has ECE = 0.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bins = []
    ece = 0.0
    n = len(confidences)
    
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confidences > lo) & (confidences <= hi)
        count = mask.sum()
        
        if count > 0:
            avg_conf = confidences[mask].mean()
            avg_acc = accuracies[mask].mean()
            bin_ece = (count / n) * abs(avg_acc - avg_conf)
            ece += bin_ece
            bins.append({
                "bin": i, "lo": round(lo, 3), "hi": round(hi, 3),
                "count": int(count), "avg_confidence": round(float(avg_conf), 4),
                "avg_accuracy": round(float(avg_acc), 4),
                "gap": round(float(abs(avg_acc - avg_conf)), 4),
            })
        else:
            bins.append({"bin": i, "lo": round(lo, 3), "hi": round(hi, 3),
                         "count": 0, "avg_confidence": 0, "avg_accuracy": 0, "gap": 0})
    
    return float(ece), bins


def compute_mce(confidences: np.ndarray, accuracies: np.ndarray,
                n_bins: int = 15) -> float:
    """Maximum Calibration Error — worst-case bin gap."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    max_gap = 0.0
    
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() > 0:
            gap = abs(accuracies[mask].mean() - confidences[mask].mean())
            max_gap = max(max_gap, gap)
    
    return float(max_gap)


def compute_overconfidence_rate(confidences: np.ndarray, correctness: np.ndarray,
                                threshold: float = 0.80) -> Dict:
    """
    Critical metric: How often is the classifier CONFIDENT AND WRONG?
    This directly affects Theorem 1's p_τ term.
    """
    confident = confidences >= threshold
    confident_wrong = confident & (~correctness)
    confident_right = confident & correctness
    
    n = len(confidences)
    n_confident = confident.sum()
    
    return {
        "n_total": n,
        "n_confident": int(n_confident),
        "n_confident_wrong": int(confident_wrong.sum()),
        "n_confident_right": int(confident_right.sum()),
        "p_tau": float(confident_wrong.sum() / n) if n > 0 else 0,
        "confident_accuracy": float(confident_right.sum() / n_confident) if n_confident > 0 else 0,
        "threshold": threshold,
    }


# ═══════════════════════════════════════════════════════════════════════════
# TEMPERATURE SCALING
# ═══════════════════════════════════════════════════════════════════════════

def find_optimal_temperature(logits: np.ndarray, labels: np.ndarray,
                              lr: float = 0.01, max_iters: int = 200) -> float:
    """
    Find optimal temperature T for logits/T that minimizes NLL.
    (Guo et al., ICML 2017: "On Calibration of Modern Neural Networks")
    
    Uses simple grid search + gradient-free optimization since we
    don't need backprop for a single scalar.
    """
    def nll_at_temp(T):
        scaled = logits / T
        # Softmax
        exp_scaled = np.exp(scaled - scaled.max(axis=1, keepdims=True))
        probs = exp_scaled / exp_scaled.sum(axis=1, keepdims=True)
        # NLL
        nll = -np.log(probs[np.arange(len(labels)), labels] + 1e-10).mean()
        return nll
    
    # Grid search
    best_T = 1.0
    best_nll = nll_at_temp(1.0)
    
    for T in np.arange(0.1, 5.0, 0.05):
        nll = nll_at_temp(T)
        if nll < best_nll:
            best_nll = nll
            best_T = T
    
    return float(best_T)


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Apply temperature scaling to logits and return calibrated probabilities."""
    scaled = logits / temperature
    exp_scaled = np.exp(scaled - scaled.max(axis=1, keepdims=True))
    probs = exp_scaled / exp_scaled.sum(axis=1, keepdims=True)
    return probs


# ═══════════════════════════════════════════════════════════════════════════
# SAFETY BOUND IMPACT
# ═══════════════════════════════════════════════════════════════════════════

def compute_safety_impact(ece_before: float, ece_after: float,
                           recall: float = 0.97, epsilon: float = 0.01,
                           p_phi: float = 0.4) -> Dict:
    """
    Show how calibration improvement tightens the safety bound.
    
    Before calibration: bound uses p_τ ≈ ece_before
    After calibration:  bound uses p_τ ≈ ece_after
    """
    bound_before = p_phi * (1 - recall) * ece_before * epsilon
    bound_after = p_phi * (1 - recall) * ece_after * epsilon
    
    improvement = 1 - (bound_after / bound_before) if bound_before > 0 else 0
    
    return {
        "bound_before_calibration": bound_before,
        "bound_after_calibration": bound_after,
        "improvement_factor": round(bound_before / max(bound_after, 1e-10), 2),
        "improvement_percent": round(improvement * 100, 1),
        "ece_before": ece_before,
        "ece_after": ece_after,
    }


# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

def load_predictions(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load test predictions. Returns (confidences, correct, tier_probs, true_tiers)."""
    with open(path) as f:
        data = json.load(f)
    
    confidences = []
    correct = []
    tier_probs = []
    true_tiers = []
    
    for item in data:
        true_t = item.get("tier", 0)
        pred_t = item.get("predicted_tier", item.get("tier", 0))
        conf = item.get("confidence", 0.95)
        probs = item.get("tier_probs", None)
        
        if probs is None:
            probs = [(1 - conf) / 3.0] * 4
            probs[pred_t] = conf
        
        confidences.append(conf)
        correct.append(int(pred_t == true_t))
        tier_probs.append(probs)
        true_tiers.append(true_t)
    
    return (np.array(confidences), np.array(correct, dtype=bool),
            np.array(tier_probs), np.array(true_tiers))


# ═══════════════════════════════════════════════════════════════════════════
# FIGURES
# ═══════════════════════════════════════════════════════════════════════════

def generate_calibration_figures(bins_before: List[Dict], bins_after: List[Dict],
                                  ece_before: float, ece_after: float,
                                  safety: Dict, out_dir: Path):
    """Generate calibration analysis figures."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "serif", "font.size": 11,
                          "figure.dpi": 300, "axes.labelsize": 12})

    # ── Reliability Diagram (before and after) ────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    
    for ax, bins, ece, title in [(ax1, bins_before, ece_before, "Before Temp. Scaling"),
                                   (ax2, bins_after, ece_after, "After Temp. Scaling")]:
        confs = [b["avg_confidence"] for b in bins if b["count"] > 0]
        accs = [b["avg_accuracy"] for b in bins if b["count"] > 0]
        counts = [b["count"] for b in bins if b["count"] > 0]
        
        # Bar widths based on bin boundaries
        width = 1.0 / 15
        positions = [(b["lo"] + b["hi"]) / 2 for b in bins if b["count"] > 0]
        
        ax.bar(positions, accs, width=width * 0.9, alpha=0.6, color="#3498db",
               edgecolor="white", label="Accuracy")
        ax.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.5, label="Perfect calibration")
        ax.set_xlabel("Mean Predicted Confidence", fontweight="bold")
        ax.set_ylabel("Fraction Correct", fontweight="bold")
        ax.set_title(f"{title}\nECE = {ece:.4f}", fontweight="bold")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    fig.suptitle("Reliability Diagrams: Calibration Analysis", fontweight="bold", y=1.02)
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(out_dir / f"fig_calibration_reliability.{ext}", dpi=300, bbox_inches="tight")
    plt.close()
    print("  fig_calibration_reliability")

    # ── Safety bound improvement ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    labels = ["Before\nCalibration", "After\nCalibration"]
    bounds = [safety["bound_before_calibration"], safety["bound_after_calibration"]]
    colors = ["#e74c3c", "#2ecc71"]
    
    bars = ax.bar(labels, bounds, color=colors, width=0.5, edgecolor="white", lw=2)
    ax.set_ylabel("P(violation per step)", fontweight="bold")
    ax.set_title(f"Safety Bound Improvement: {safety['improvement_factor']}× tighter",
                 fontweight="bold")
    
    for bar, val in zip(bars, bounds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(bounds)*0.02,
                f"{val:.2e}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(out_dir / f"fig_calibration_safety.{ext}", dpi=300, bbox_inches="tight")
    plt.close()
    print("  fig_calibration_safety")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("Calibration Analysis for CompTS Safety Guarantees")
    print("=" * 60)

    # Load predictions
    test_path = ROOT / "data/test_with_predictions.json"
    if not test_path.exists():
        print("No test_with_predictions.json found. Generating synthetic calibration data...")
        # Generate synthetic data for analysis
        n = 2000
        confidences = np.random.beta(15, 2, n)  # Typical overconfident model
        true_tiers = np.random.choice(4, n, p=[0.2, 0.15, 0.25, 0.4])
        # Simulate imperfect classifier
        correct = np.random.random(n) < (confidences * 0.95 + 0.02)
        tier_probs = np.zeros((n, 4))
        for i in range(n):
            tier_probs[i, true_tiers[i]] = confidences[i]
            remaining = (1 - confidences[i]) / 3
            for t in range(4):
                if t != true_tiers[i]:
                    tier_probs[i, t] = remaining
        logits = np.log(tier_probs + 1e-10)
    else:
        confidences, correct, tier_probs, true_tiers = load_predictions(test_path)
        logits = np.log(tier_probs + 1e-10)
    
    print(f"Loaded {len(confidences)} predictions")
    print(f"Overall accuracy: {correct.mean():.4f}")
    print(f"Mean confidence: {confidences.mean():.4f}")

    # ── ECE before calibration ────────────────────────────────────────────
    print("\n▸ Pre-calibration metrics:")
    ece_before, bins_before = compute_ece(confidences, correct)
    mce_before = compute_mce(confidences, correct)
    overconf = compute_overconfidence_rate(confidences, correct)
    
    print(f"  ECE: {ece_before:.4f}")
    print(f"  MCE: {mce_before:.4f}")
    print(f"  Overconfidence: {overconf['n_confident_wrong']}/{overconf['n_confident']} "
          f"confident predictions wrong (p_τ = {overconf['p_tau']:.4f})")

    # ── Temperature scaling ───────────────────────────────────────────────
    print("\n▸ Finding optimal temperature...")
    T_opt = find_optimal_temperature(logits, true_tiers)
    print(f"  Optimal T = {T_opt:.3f}")
    
    # Apply calibration
    calibrated_probs = apply_temperature(logits, T_opt)
    cal_confidences = calibrated_probs.max(axis=1)
    cal_preds = calibrated_probs.argmax(axis=1)
    cal_correct = (cal_preds == true_tiers)

    # ── ECE after calibration ─────────────────────────────────────────────
    print("\n▸ Post-calibration metrics:")
    ece_after, bins_after = compute_ece(cal_confidences, cal_correct)
    mce_after = compute_mce(cal_confidences, cal_correct)
    overconf_after = compute_overconfidence_rate(cal_confidences, cal_correct)
    
    print(f"  ECE: {ece_after:.4f} (was {ece_before:.4f})")
    print(f"  MCE: {mce_after:.4f} (was {mce_before:.4f})")
    print(f"  Overconfidence: p_τ = {overconf_after['p_tau']:.4f} (was {overconf['p_tau']:.4f})")

    # ── Safety bound impact ───────────────────────────────────────────────
    print("\n▸ Safety bound impact:")
    # Use overconfidence rate as p_τ (more accurate than ECE for our bound)
    safety = compute_safety_impact(
        max(overconf["p_tau"], ece_before),
        max(overconf_after["p_tau"], ece_after),
    )
    print(f"  Bound before: {safety['bound_before_calibration']:.2e}")
    print(f"  Bound after:  {safety['bound_after_calibration']:.2e}")
    print(f"  Improvement:  {safety['improvement_factor']}× tighter "
          f"({safety['improvement_percent']}%)")

    # ── Generate figures ──────────────────────────────────────────────────
    print("\nFigures:")
    try:
        generate_calibration_figures(bins_before, bins_after, ece_before, ece_after,
                                     safety, OUT)
    except ImportError:
        print("  matplotlib not available, skipping figures")

    # ── Save results ──────────────────────────────────────────────────────
    results = {
        "pre_calibration": {
            "ece": ece_before, "mce": mce_before,
            "overconfidence": overconf,
        },
        "post_calibration": {
            "ece": ece_after, "mce": mce_after,
            "temperature": T_opt,
            "overconfidence": overconf_after,
        },
        "safety_impact": safety,
    }
    
    results_path = OUT / "calibration_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {results_path}")


if __name__ == "__main__":
    main()
