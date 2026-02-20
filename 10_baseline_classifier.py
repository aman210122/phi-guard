#!/usr/bin/env python3
"""
10_baseline_classifier.py — TF-IDF + Logistic Regression Baseline

PURPOSE:
  Train a simple non-neural classifier to prove CompTS safety works
  across classifier quality levels. If CompTS achieves 0% violations
  with BOTH DistilBERT (F1=0.97) and TF-IDF+LR (expected F1~0.85-0.92),
  that proves the safety theorem generalizes.

USAGE:
  python scripts/10_baseline_classifier.py
  
  Then re-run: python scripts/05_bandit_routing.py  (uses new predictions)
"""

import json
import sys
from pathlib import Path
from collections import Counter

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def main():
    print("=" * 60)
    print("Baseline Classifier: TF-IDF + Logistic Regression")
    print("=" * 60)
    
    # Load data
    data_dir = ROOT / "data"
    train = json.loads((data_dir / "train.json").read_text())
    val = json.loads((data_dir / "val.json").read_text())
    test = json.loads((data_dir / "test.json").read_text())
    
    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.calibration import CalibratedClassifierCV
    
    train_texts = [d["text"] for d in train]
    train_labels = [d["tier"] for d in train]
    val_texts = [d["text"] for d in val]
    val_labels = [d["tier"] for d in val]
    test_texts = [d["text"] for d in test]
    test_labels = [d["tier"] for d in test]
    
    # TF-IDF features
    print("\nFitting TF-IDF (max_features=10000, ngram_range=(1,2))...")
    tfidf = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    X_train = tfidf.fit_transform(train_texts)
    X_val = tfidf.transform(val_texts)
    X_test = tfidf.transform(test_texts)
    
    print(f"  Feature matrix: {X_train.shape}")
    
    # Logistic Regression with calibration
    print("\nTraining Logistic Regression...")
    base_lr = LogisticRegression(
        C=1.0,
        max_iter=1000,
        solver="lbfgs",
        random_state=42,
    )
    
    # Calibrate for reliable confidence scores
    lr = CalibratedClassifierCV(base_lr, cv=3, method="isotonic")
    lr.fit(X_train, train_labels)
    
    # Evaluate
    val_pred = lr.predict(X_val)
    val_proba = lr.predict_proba(X_val)
    test_pred = lr.predict(X_test)
    test_proba = lr.predict_proba(X_test)
    
    labels = ["T0_Public", "T1_Internal", "T2_Limited", "T3_Restricted"]
    
    print("\n" + "=" * 60)
    print("Test Evaluation (TF-IDF + Logistic Regression)")
    print("=" * 60)
    print(classification_report(test_labels, test_pred, target_names=labels))
    
    # Safety metrics
    t3_mask = np.array(test_labels) == 3
    t3_pred = np.array(test_pred)[t3_mask]
    t3_correct = (t3_pred == 3).sum()
    t3_total = t3_mask.sum()
    t3_recall = t3_correct / max(t3_total, 1)
    t3_misclassified = t3_total - t3_correct
    
    print(f"\n=== SAFETY ===")
    print(f"T3 Recall: {t3_recall:.4f} ({t3_misclassified}/{t3_total} misclassified)")
    
    # Confidence on errors
    t3_proba = test_proba[t3_mask]
    t3_conf = np.max(t3_proba, axis=1)
    wrong_mask = t3_pred != 3
    if wrong_mask.any():
        wrong_conf = t3_conf[wrong_mask]
        print(f"  Misclassified confidence: mean={wrong_conf.mean():.3f} max={wrong_conf.max():.3f}")
        tau = 0.80
        caught = (wrong_conf < tau).sum()
        print(f"  Caught by τ={tau}: {caught}/{wrong_mask.sum()}")
    
    # Overall confidence
    all_conf = np.max(test_proba, axis=1)
    correct_mask = np.array(test_pred) == np.array(test_labels)
    print(f"Confidence: correct={all_conf[correct_mask].mean():.3f} "
          f"incorrect={all_conf[~correct_mask].mean():.3f}")
    
    # Save predictions in same format as Script 02
    print("\nSaving predictions...")
    predictions = []
    for i, item in enumerate(test):
        proba = test_proba[i].tolist()
        pred = int(test_pred[i])
        conf = float(max(proba))
        
        predictions.append({
            **item,
            "predicted_tier": pred,
            "confidence": conf,
            "tier_probs": proba,
            "correct": pred == item["tier"],
            "classifier": "tfidf_logreg",
        })
    
    # Save as separate file (don't overwrite DistilBERT predictions)
    baseline_pred_path = data_dir / "test_with_predictions_baseline.json"
    with open(baseline_pred_path, "w") as f:
        json.dump(predictions, f, indent=2, default=str)
    
    # Also save a comparison summary
    summary = {
        "classifier": "TF-IDF + Logistic Regression",
        "features": "TF-IDF (10K features, unigrams+bigrams)",
        "n_test": len(test),
        "accuracy": float(correct_mask.mean()),
        "t3_recall": float(t3_recall),
        "t3_misclassified": int(t3_misclassified),
        "t3_total": int(t3_total),
        "mean_confidence_correct": float(all_conf[correct_mask].mean()),
        "mean_confidence_incorrect": float(all_conf[~correct_mask].mean()) if (~correct_mask).any() else None,
    }
    
    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "baseline_classifier_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nPredictions: {baseline_pred_path}")
    print(f"Summary: {out_dir / 'baseline_classifier_results.json'}")
    
    # Generate LaTeX comparison table
    print("\n" + "=" * 60)
    print("COMPARISON: DistilBERT vs TF-IDF+LR")
    print("=" * 60)
    
    # Load DistilBERT results if available
    distilbert_path = ROOT / "models" / "test_results.json"
    if distilbert_path.exists():
        db_results = json.loads(distilbert_path.read_text())
        print(f"\n{'Metric':<30} {'DistilBERT':>12} {'TF-IDF+LR':>12}")
        print("-" * 56)
        print(f"{'Accuracy':<30} {db_results.get('accuracy', 'N/A'):>12} {summary['accuracy']:>12.4f}")
        print(f"{'T3 Recall':<30} {db_results.get('t3_recall', 'N/A'):>12} {summary['t3_recall']:>12.4f}")
        print(f"{'T3 Misclassified':<30} {db_results.get('t3_misclassified', 'N/A'):>12} {summary['t3_misclassified']:>12}")
    
    print("\nNext: Run scripts/05_bandit_routing.py with --predictions baseline")
    print("  to verify CompTS achieves 0% violations with this weaker classifier too.")


if __name__ == "__main__":
    main()
