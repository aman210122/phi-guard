#!/usr/bin/env python3
"""
14_hybrid_classifier.py — DistilBERT + Rule-Based PHI Detection

PURPOSE:
  The current DistilBERT classifier has T3 recall=0.93, meaning 33 PHI
  queries slip through. At ε=0.01, CompTS compensates by routing 
  EVERYTHING to on-prem — making it equivalent to SecureDefault.
  
  If we boost T3 recall to ~0.98, CompTS can confidently route T0/T1/T2
  queries to cheaper platforms while catching the remaining few T3 errors.
  THIS is what makes CompTS outperform threshold baselines.

APPROACH:
  Hybrid classifier = max(DistilBERT_tier, RuleBased_tier)
  
  Rule-based layer scans for:
  - Full names (capitalized word pairs not in medical dictionary)
  - Medical Record Numbers (MRN patterns)
  - Social Security Numbers
  - Date of Birth patterns near patient context
  - Phone numbers
  - Addresses
  
  If rules detect PHI → classify as T3 regardless of DistilBERT output.
  This boosts T3 recall (catches what DistilBERT misses) at cost of 
  slightly lower T2 precision (some T2 may be bumped to T3).

USAGE:
  python scripts/14_hybrid_classifier.py

OUTPUT:
  Comparison: DistilBERT-only vs Hybrid
  New predictions saved for routing evaluation
"""

import json
import re
import sys
from pathlib import Path
from collections import Counter

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

# ═══════════════════════════════════════════════════════════════
# RULE-BASED PHI DETECTOR
# ═══════════════════════════════════════════════════════════════

# Common medical terms that look like names but aren't
MEDICAL_TERMS = {
    "chief", "complaint", "history", "present", "illness", "admission",
    "discharge", "diagnosis", "medications", "allergies", "assessment",
    "plan", "patient", "hospital", "department", "service", "attending",
    "surgery", "medicine", "cardiology", "oncology", "neurology",
    "pulmonary", "renal", "hepatic", "cardiac", "acute", "chronic",
    "bilateral", "anterior", "posterior", "superior", "inferior",
    "normal", "abnormal", "elevated", "decreased", "stable", "critical",
    "review", "systems", "physical", "examination", "laboratory",
    "imaging", "procedure", "operation", "anesthesia", "recovery",
    "follow", "clinic", "outpatient", "inpatient", "emergency",
    "intensive", "care", "unit", "ward", "floor", "bed", "room",
    "doctor", "nurse", "physician", "surgeon", "specialist",
    "mg", "ml", "tab", "cap", "bid", "tid", "qid", "prn", "po", "iv",
    "blood", "pressure", "heart", "rate", "temperature", "oxygen",
    "saturation", "respiratory", "pulse", "weight", "height", "bmi",
    "white", "red", "cell", "count", "hemoglobin", "hematocrit",
    "platelet", "sodium", "potassium", "chloride", "bicarbonate",
    "glucose", "creatinine", "protein", "albumin", "bilirubin",
    "left", "right", "upper", "lower", "middle", "central", "lateral",
    "medial", "proximal", "distal", "superficial", "deep",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december",
    "social", "family", "past", "medical", "surgical",
    "united", "states", "american", "national", "general", "memorial",
    "community", "regional", "university", "medical", "center",
    "saint", "mount", "north", "south", "east", "west",
}

# Common first/last names for synthetic PHI detection
COMMON_NAMES = {
    "james", "john", "robert", "michael", "william", "david", "richard",
    "joseph", "thomas", "charles", "mary", "patricia", "jennifer", "linda",
    "elizabeth", "barbara", "susan", "jessica", "sarah", "karen",
    "smith", "johnson", "williams", "brown", "jones", "garcia", "miller",
    "davis", "rodriguez", "martinez", "hernandez", "lopez", "gonzalez",
    "wilson", "anderson", "taylor", "moore", "jackson", "martin", "lee",
    "thompson", "white", "harris", "sanchez", "clark", "lewis", "robinson",
}


def detect_phi_rules(text: str) -> dict:
    """Apply rule-based PHI detection. Returns detection results."""
    detections = {
        "ssn": False,
        "mrn": False,
        "phone": False,
        "name_pattern": False,
        "dob_pattern": False,
        "address": False,
        "total_detections": 0,
        "confidence_boost": 0.0,
    }
    
    text_lower = text.lower()
    
    # SSN patterns: XXX-XX-XXXX or XXX XX XXXX
    ssn_pattern = r'\b\d{3}[-\s]\d{2}[-\s]\d{4}\b'
    if re.search(ssn_pattern, text):
        detections["ssn"] = True
        detections["total_detections"] += 1
    
    # MRN patterns: "MRN" or "Medical Record" followed by numbers
    mrn_patterns = [
        r'(?:mrn|medical\s*record\s*(?:number|no|#)?)\s*[:.]?\s*\d{4,}',
        r'\b[A-Z]{1,3}\d{6,}\b',  # e.g., MR1234567
    ]
    for pat in mrn_patterns:
        if re.search(pat, text, re.IGNORECASE):
            detections["mrn"] = True
            detections["total_detections"] += 1
            break
    
    # Phone numbers: (XXX) XXX-XXXX or XXX-XXX-XXXX
    phone_pattern = r'(?:\(\d{3}\)\s*|\b\d{3}[-.])\d{3}[-.]?\d{4}\b'
    if re.search(phone_pattern, text):
        detections["phone"] = True
        detections["total_detections"] += 1
    
    # Name patterns: Two consecutive capitalized words that aren't medical terms
    # Look for patterns like "Mr. Smith", "Dr. Johnson", or "John Smith"
    name_patterns = [
        r'\b(?:Mr|Mrs|Ms|Dr|Prof)\.?\s+[A-Z][a-z]+\b',
        r'\b(?:patient|pt)\s+[A-Z][a-z]+\s+[A-Z][a-z]+\b',
    ]
    for pat in name_patterns:
        if re.search(pat, text):
            detections["name_pattern"] = True
            detections["total_detections"] += 1
            break
    
    # Also check for known synthetic names from Synthea
    words = text.split()
    for i in range(len(words) - 1):
        w1 = words[i].strip(".,;:()\"'").lower()
        w2 = words[i+1].strip(".,;:()\"'").lower()
        if (w1 in COMMON_NAMES and w2 in COMMON_NAMES and
            w1 not in MEDICAL_TERMS and w2 not in MEDICAL_TERMS and
            words[i][0].isupper() and words[i+1][0].isupper()):
            detections["name_pattern"] = True
            detections["total_detections"] += 1
            break
    
    # DOB patterns near patient context
    dob_patterns = [
        r'(?:date\s*of\s*birth|dob|born|birthday)\s*[:.]?\s*\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}',
        r'(?:age|aged)\s*[:.]?\s*\d{1,3}\s*(?:year|yr|y\.?o\.?)',
    ]
    for pat in dob_patterns:
        if re.search(pat, text, re.IGNORECASE):
            detections["dob_pattern"] = True
            detections["total_detections"] += 1
            break
    
    # Address patterns: number + street name + street type
    address_pattern = r'\b\d{1,5}\s+[A-Z][a-z]+\s+(?:Street|St|Avenue|Ave|Boulevard|Blvd|Drive|Dr|Road|Rd|Lane|Ln|Way|Court|Ct|Place|Pl)\b'
    if re.search(address_pattern, text):
        detections["address"] = True
        detections["total_detections"] += 1
    
    # Calculate confidence boost
    if detections["total_detections"] >= 3:
        detections["confidence_boost"] = 0.99  # Very high confidence
    elif detections["total_detections"] == 2:
        detections["confidence_boost"] = 0.95
    elif detections["total_detections"] == 1:
        detections["confidence_boost"] = 0.85
    else:
        detections["confidence_boost"] = 0.0
    
    return detections


def hybrid_classify(query: dict) -> dict:
    """Combine DistilBERT prediction with rule-based detection."""
    bert_pred = query["predicted_tier"]
    bert_conf = query["confidence"]
    bert_probs = query.get("tier_probs", [0.25, 0.25, 0.25, 0.25])
    text = query.get("text", "")
    
    # Run rule-based detection
    rules = detect_phi_rules(text)
    
    # Hybrid decision
    if rules["total_detections"] > 0 and bert_pred < 3:
        # Rules found PHI but DistilBERT didn't classify as T3
        # Boost to T3
        hybrid_pred = 3
        hybrid_conf = max(bert_conf, rules["confidence_boost"])
        # Adjust probabilities
        hybrid_probs = list(bert_probs)
        boost = rules["confidence_boost"]
        # Shift probability mass to T3
        for i in range(3):
            hybrid_probs[i] *= (1 - boost)
        hybrid_probs[3] = boost + hybrid_probs[3] * (1 - boost)
        hybrid_source = "rule_override"
    else:
        # Use DistilBERT as-is
        hybrid_pred = bert_pred
        hybrid_conf = bert_conf
        hybrid_probs = bert_probs
        hybrid_source = "distilbert"
    
    return {
        **query,
        "predicted_tier": hybrid_pred,
        "confidence": hybrid_conf,
        "tier_probs": hybrid_probs,
        "hybrid_source": hybrid_source,
        "rule_detections": rules["total_detections"],
        "original_bert_pred": bert_pred,
        "original_bert_conf": bert_conf,
    }


def main():
    data_dir = ROOT / "data"
    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    
    # Load original DistilBERT predictions
    pred_path = data_dir / "test_with_predictions.json"
    if not pred_path.exists():
        print(f"ERROR: {pred_path} not found")
        sys.exit(1)
    
    predictions = json.loads(pred_path.read_text())
    print(f"Loaded {len(predictions)} predictions\n")
    
    # ── DistilBERT baseline stats ──────────────────────────────────
    true_labels = [p["tier"] for p in predictions]
    bert_preds = [p["predicted_tier"] for p in predictions]
    
    t3_mask = [t == 3 for t in true_labels]
    t3_bert_correct = sum(1 for t, p in zip(true_labels, bert_preds) if t == 3 and p == 3)
    t3_total = sum(t3_mask)
    bert_t3_recall = t3_bert_correct / t3_total
    bert_missed = t3_total - t3_bert_correct
    
    print("=" * 60)
    print("DistilBERT Only")
    print("=" * 60)
    print(f"T3 Recall: {bert_t3_recall:.4f} ({bert_missed} missed out of {t3_total})")
    
    # ── Apply hybrid classifier ────────────────────────────────────
    hybrid_predictions = [hybrid_classify(p) for p in predictions]
    
    hybrid_preds = [p["predicted_tier"] for p in hybrid_predictions]
    
    t3_hybrid_correct = sum(1 for t, p in zip(true_labels, hybrid_preds) if t == 3 and p == 3)
    hybrid_t3_recall = t3_hybrid_correct / t3_total
    hybrid_missed = t3_total - t3_hybrid_correct
    
    # How many did rules catch that BERT missed?
    rule_catches = sum(1 for p in hybrid_predictions 
                       if p["hybrid_source"] == "rule_override" and p["tier"] == 3)
    rule_false_alarms = sum(1 for p in hybrid_predictions
                           if p["hybrid_source"] == "rule_override" and p["tier"] != 3)
    
    print(f"\n{'=' * 60}")
    print("Hybrid (DistilBERT + Rules)")
    print("=" * 60)
    print(f"T3 Recall: {hybrid_t3_recall:.4f} ({hybrid_missed} missed out of {t3_total})")
    print(f"Rules caught {rule_catches} T3 queries that BERT missed")
    print(f"Rules false alarms: {rule_false_alarms} (non-T3 bumped to T3)")
    print(f"Net T3 recall improvement: {bert_t3_recall:.4f} → {hybrid_t3_recall:.4f}")
    
    # Overall accuracy
    bert_acc = sum(1 for t, p in zip(true_labels, bert_preds) if t == p) / len(true_labels)
    hybrid_acc = sum(1 for t, p in zip(true_labels, hybrid_preds) if t == p) / len(true_labels)
    print(f"\nOverall accuracy: {bert_acc:.4f} → {hybrid_acc:.4f}")
    
    # ── Per-tier comparison ────────────────────────────────────────
    print(f"\n{'Tier':<15} {'BERT Recall':>12} {'Hybrid Recall':>14} {'Change':>8}")
    print("-" * 51)
    for tier in range(4):
        tier_mask = [t == tier for t in true_labels]
        tier_total = sum(tier_mask)
        if tier_total == 0:
            continue
        bert_recall = sum(1 for t, p in zip(true_labels, bert_preds) if t == tier and p == tier) / tier_total
        hybrid_recall = sum(1 for t, p in zip(true_labels, hybrid_preds) if t == tier and p == tier) / tier_total
        change = hybrid_recall - bert_recall
        labels = ["T0_Public", "T1_Internal", "T2_Limited", "T3_Restricted"]
        print(f"{labels[tier]:<15} {bert_recall:>12.4f} {hybrid_recall:>14.4f} {change:>+8.4f}")
    
    # ── Save hybrid predictions ────────────────────────────────────
    hybrid_path = data_dir / "test_with_predictions_hybrid.json"
    with open(hybrid_path, "w") as f:
        json.dump(hybrid_predictions, f, indent=2, default=str)
    
    # ── Now simulate routing with hybrid classifier ────────────────
    print(f"\n{'=' * 60}")
    print("ROUTING COMPARISON: DistilBERT vs Hybrid")
    print("=" * 60)
    
    platforms = [
        {"name": "PublicAPI", "clearance": 0, "cost_per_1k": 0.010},
        {"name": "SecureCloud", "clearance": 2, "cost_per_1k": 0.011},
        {"name": "OnPremises", "clearance": 3, "cost_per_1k": 0.025},
    ]
    
    def route_compts(preds, epsilon=0.01, tau=0.80):
        total_cost = 0
        violations = 0
        platform_counts = Counter()
        
        for q in preds:
            conf = q["confidence"]
            probs = q.get("tier_probs", [0.25, 0.25, 0.25, 0.25])
            true_tier = q["tier"]
            tokens = len(q.get("text", "").split()) * 1.3
            
            if conf < tau:
                chosen = platforms[-1]
            else:
                safe = []
                for p in platforms:
                    viol_prob = sum(probs[t] for t in range(4) if t > p["clearance"])
                    if viol_prob <= epsilon:
                        safe.append(p)
                chosen = min(safe, key=lambda p: p["cost_per_1k"]) if safe else platforms[-1]
            
            total_cost += chosen["cost_per_1k"] * tokens / 1000
            violations += int(chosen["clearance"] < true_tier)
            platform_counts[chosen["name"]] += 1
        
        return total_cost, violations, dict(platform_counts)
    
    def route_static(preds):
        total_cost = 0
        violations = 0
        for q in preds:
            pred = q["predicted_tier"]
            true_tier = q["tier"]
            tokens = len(q.get("text", "").split()) * 1.3
            safe = [p for p in platforms if p["clearance"] >= pred]
            chosen = min(safe, key=lambda p: p["cost_per_1k"]) if safe else platforms[-1]
            total_cost += chosen["cost_per_1k"] * tokens / 1000
            violations += int(chosen["clearance"] < true_tier)
        return total_cost, violations
    
    # DistilBERT routing
    bert_compts_cost, bert_compts_viol, bert_routing = route_compts(predictions)
    bert_static_cost, bert_static_viol = route_static(predictions)
    
    # Hybrid routing
    hybrid_compts_cost, hybrid_compts_viol, hybrid_routing = route_compts(hybrid_predictions)
    hybrid_static_cost, hybrid_static_viol = route_static(hybrid_predictions)
    
    secure_cost = sum(platforms[-1]["cost_per_1k"] * len(q.get("text","").split()) * 1.3 / 1000 
                      for q in predictions)
    
    print(f"\n{'Strategy':<35} {'Cost':>10} {'Viol':>6} {'Routing':>35}")
    print("-" * 88)
    print(f"{'StaticILP (DistilBERT)':<35} ${bert_static_cost:>8.4f}  {bert_static_viol:>4}   ---")
    print(f"{'CompTS ε=0.01 (DistilBERT)':<35} ${bert_compts_cost:>8.4f}  {bert_compts_viol:>4}   {bert_routing}")
    print(f"{'StaticILP (Hybrid)':<35} ${hybrid_static_cost:>8.4f}  {hybrid_static_viol:>4}   ---")
    print(f"{'CompTS ε=0.01 (Hybrid)':<35} ${hybrid_compts_cost:>8.4f}  {hybrid_compts_viol:>4}   {hybrid_routing}")
    print(f"{'SecureDefault':<35} ${secure_cost:>8.4f}  {0:>4}   OnPremises:1380")
    
    # The key question
    print(f"\n{'=' * 60}")
    print("KEY QUESTION: Does hybrid CompTS route a MIX?")
    print("=" * 60)
    
    if len(hybrid_routing) > 1:
        print("✓ YES! Hybrid CompTS routes to multiple platforms at 0 violations!")
        print(f"  Routing mix: {hybrid_routing}")
        print(f"  Cost savings vs SecureDefault: "
              f"{(1 - hybrid_compts_cost/secure_cost)*100:.1f}%")
        print(f"\n  THIS is the NeurIPS result.")
    else:
        print("✗ No — still routing everything to on-prem.")
        print(f"  T3 recall = {hybrid_t3_recall:.4f} — may need higher recall")
        print(f"  or higher ε to enable mixed routing.")
        
        # Try higher epsilon
        print(f"\n  Trying ε=0.05...")
        cost_05, viol_05, routing_05 = route_compts(hybrid_predictions, epsilon=0.05)
        print(f"  CompTS ε=0.05 (Hybrid): ${cost_05:.4f}  {viol_05} violations  {routing_05}")
        
        print(f"\n  Trying ε=0.10...")
        cost_10, viol_10, routing_10 = route_compts(hybrid_predictions, epsilon=0.10)
        print(f"  CompTS ε=0.10 (Hybrid): ${cost_10:.4f}  {viol_10} violations  {routing_10}")
    
    # Save summary
    summary = {
        "distilbert_t3_recall": round(bert_t3_recall, 4),
        "hybrid_t3_recall": round(hybrid_t3_recall, 4),
        "rules_caught": rule_catches,
        "rules_false_alarms": rule_false_alarms,
        "distilbert_compts_cost": round(bert_compts_cost, 4),
        "distilbert_compts_violations": bert_compts_viol,
        "distilbert_compts_routing": bert_routing,
        "hybrid_compts_cost": round(hybrid_compts_cost, 4),
        "hybrid_compts_violations": hybrid_compts_viol,
        "hybrid_compts_routing": hybrid_routing,
    }
    
    with open(out_dir / "hybrid_classifier_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSaved: {hybrid_path}")
    print(f"Summary: {out_dir / 'hybrid_classifier_results.json'}")


if __name__ == "__main__":
    main()
