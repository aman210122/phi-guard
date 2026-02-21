#!/usr/bin/env python3
"""
15_diagnose_routing.py — WHY does CompTS route everything to on-prem?

Inspects the posterior violation probability for each platform on every query.
Shows exactly which queries COULD be routed to cheaper platforms and why they aren't.
"""

import json
import sys
from pathlib import Path
from collections import Counter

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def main():
    data_dir = ROOT / "data"
    
    # Try hybrid first, fall back to original
    pred_path = data_dir / "test_with_predictions_hybrid.json"
    if not pred_path.exists():
        pred_path = data_dir / "test_with_predictions.json"
    
    predictions = json.loads(pred_path.read_text())
    print(f"Loaded {len(predictions)} predictions from {pred_path.name}\n")
    
    platforms = [
        {"name": "PublicAPI",   "clearance": 0},
        {"name": "SecureCloud", "clearance": 2},
        {"name": "OnPremises",  "clearance": 3},
    ]
    
    tau = 0.80
    
    print("=" * 70)
    print("DIAGNOSIS: Per-query violation probabilities")
    print("=" * 70)
    
    # For each query, compute violation prob for each platform
    platform_viable = {p["name"]: {"viable": 0, "blocked": 0, "reasons": []} 
                       for p in platforms}
    
    low_conf_count = 0
    
    # Collect violation probs
    public_viol_probs = []
    cloud_viol_probs = []
    
    for q in predictions:
        conf = q["confidence"]
        probs = q.get("tier_probs", [0.25, 0.25, 0.25, 0.25])
        true_tier = q["tier"]
        pred_tier = q["predicted_tier"]
        
        if conf < tau:
            low_conf_count += 1
            continue
        
        # Violation prob for PublicAPI (clearance=0): P(tier > 0)
        pub_viol = sum(probs[t] for t in range(1, 4))
        public_viol_probs.append(pub_viol)
        
        # Violation prob for SecureCloud (clearance=2): P(tier > 2)  
        cloud_viol = sum(probs[t] for t in range(3, 4))  # just probs[3]
        cloud_viol_probs.append(cloud_viol)
        
    public_viol_probs = np.array(public_viol_probs)
    cloud_viol_probs = np.array(cloud_viol_probs)
    
    print(f"\nLow confidence (< τ={tau}): {low_conf_count} queries → fallback to on-prem")
    print(f"High confidence: {len(public_viol_probs)} queries analyzed")
    
    print(f"\n--- PublicAPI (clearance=0) ---")
    print(f"Violation prob = P(true_tier > 0) from softmax posteriors")
    print(f"  Min:    {public_viol_probs.min():.6f}")
    print(f"  Median: {np.median(public_viol_probs):.6f}")
    print(f"  Mean:   {public_viol_probs.mean():.6f}")
    print(f"  Max:    {public_viol_probs.max():.6f}")
    for eps in [0.01, 0.05, 0.10, 0.20, 0.30, 0.50]:
        n_viable = (public_viol_probs <= eps).sum()
        print(f"  Viable at ε={eps}: {n_viable}/{len(public_viol_probs)} "
              f"({n_viable/len(public_viol_probs)*100:.1f}%)")
    
    print(f"\n--- SecureCloud (clearance=2) ---")
    print(f"Violation prob = P(true_tier > 2) = P(T3) from softmax posteriors")
    print(f"  Min:    {cloud_viol_probs.min():.6f}")
    print(f"  Median: {np.median(cloud_viol_probs):.6f}")
    print(f"  Mean:   {cloud_viol_probs.mean():.6f}")
    print(f"  Max:    {cloud_viol_probs.max():.6f}")
    for eps in [0.01, 0.05, 0.10, 0.20, 0.30, 0.50]:
        n_viable = (cloud_viol_probs <= eps).sum()
        print(f"  Viable at ε={eps}: {n_viable}/{len(cloud_viol_probs)} "
              f"({n_viable/len(cloud_viol_probs)*100:.1f}%)")
    
    # Break down by TRUE tier
    print(f"\n--- SecureCloud viability by TRUE tier ---")
    for tier in range(4):
        tier_mask = [q["tier"] == tier for q in predictions if q["confidence"] >= tau]
        tier_probs = cloud_viol_probs[tier_mask]
        if len(tier_probs) == 0:
            continue
        labels = ["T0_Public", "T1_Internal", "T2_Limited", "T3_Restricted"]
        viable_01 = (tier_probs <= 0.01).sum()
        viable_10 = (tier_probs <= 0.10).sum()
        print(f"  {labels[tier]}: n={len(tier_probs)}, "
              f"mean_viol_prob={tier_probs.mean():.4f}, "
              f"viable@ε=0.01: {viable_01} ({viable_01/len(tier_probs)*100:.0f}%), "
              f"viable@ε=0.10: {viable_10} ({viable_10/len(tier_probs)*100:.0f}%)")
    
    # Show the actual softmax distributions for a few T0 queries
    print(f"\n--- Sample T0 queries: softmax posteriors ---")
    t0_queries = [q for q in predictions if q["tier"] == 0 and q["confidence"] >= tau][:5]
    for i, q in enumerate(t0_queries):
        probs = q.get("tier_probs", [])
        pred = q["predicted_tier"]
        conf = q["confidence"]
        print(f"  T0 query {i}: pred=T{pred}, conf={conf:.4f}, "
              f"probs=[{probs[0]:.4f}, {probs[1]:.4f}, {probs[2]:.4f}, {probs[3]:.4f}]")
        cloud_viol = probs[3]
        print(f"    → SecureCloud viol_prob = {cloud_viol:.4f} "
              f"({'SAFE' if cloud_viol <= 0.01 else 'BLOCKED'} at ε=0.01)")
    
    print(f"\n--- Sample T3 queries that BERT misclassified ---")
    missed = [q for q in predictions 
              if q["tier"] == 3 and q.get("original_bert_pred", q["predicted_tier"]) != 3][:5]
    for i, q in enumerate(missed):
        probs = q.get("tier_probs", [])
        pred = q["predicted_tier"]
        conf = q["confidence"]
        orig = q.get("original_bert_pred", "?")
        print(f"  Missed T3 query {i}: bert_pred=T{orig}, hybrid_pred=T{pred}, "
              f"conf={conf:.4f}")
        print(f"    probs=[{probs[0]:.4f}, {probs[1]:.4f}, {probs[2]:.4f}, {probs[3]:.4f}]")


if __name__ == "__main__":
    main()
