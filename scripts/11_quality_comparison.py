#!/usr/bin/env python3
"""
11_quality_comparison.py — Cross-Platform Response Quality Evaluation

PURPOSE:
  Send SAME queries to BOTH endpoints and measure response quality difference.
  Proves that routing non-PHI queries to cheaper platforms doesn't degrade answers.

REQUIRES:
  - OpenAI API key (set OPENAI_API_KEY)
  - Ollama running (ollama serve)
  - pip install openai bert-score

USAGE:
  python scripts/11_quality_comparison.py --n-queries 100
"""

import argparse
import json
import os
import sys
import time
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def create_prompt(text: str) -> str:
    text = text[:500] if len(text) > 500 else text
    return (
        f"You are a clinical assistant. Based on the following text, "
        f"provide a brief clinical summary or answer in 2-3 sentences.\n\n"
        f"Text: {text}\n\nResponse:"
    )


def query_openai(prompt, max_tokens=150):
    from openai import OpenAI
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
    start = time.time()
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens, temperature=0.3,
    )
    return resp.choices[0].message.content, time.time() - start


def query_ollama(prompt, model="llama3.1:8b", max_tokens=150):
    import requests
    start = time.time()
    r = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt,
              "options": {"num_predict": max_tokens, "temperature": 0.3},
              "stream": False},
        timeout=120,
    )
    return r.json().get("response", ""), time.time() - start


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-queries", type=int, default=100)
    parser.add_argument("--data-dir", type=str, default="data")
    args = parser.parse_args()

    data_dir = ROOT / args.data_dir
    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("Cross-Platform Response Quality Comparison")
    print("=" * 60)

    # Check endpoints
    has_openai = bool(os.environ.get("OPENAI_API_KEY"))
    has_ollama = False
    try:
        import requests
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        has_ollama = r.status_code == 200
    except Exception:
        pass

    if not (has_openai and has_ollama):
        print("ERROR: Need BOTH OpenAI and Ollama running.")
        if not has_openai:
            print("  Missing: set OPENAI_API_KEY")
        if not has_ollama:
            print("  Missing: ollama serve")
        sys.exit(1)

    print("  ✓ OpenAI ready")
    print("  ✓ Ollama ready")

    # Load test queries balanced by tier
    test = json.loads((data_dir / "test.json").read_text())
    by_tier = {}
    for item in test:
        by_tier.setdefault(item.get("tier", 0), []).append(item)

    random.seed(42)
    per_tier = args.n_queries // 4
    queries = []
    for tier in sorted(by_tier.keys()):
        pool = by_tier[tier]
        random.shuffle(pool)
        queries.extend(pool[:per_tier])
    random.shuffle(queries)
    queries = queries[:args.n_queries]

    print(f"\nSending {len(queries)} queries to BOTH endpoints...\n")

    results = []
    for i, q in enumerate(queries):
        prompt = create_prompt(q["text"])
        try:
            openai_resp, openai_time = query_openai(prompt)
        except Exception as e:
            openai_resp, openai_time = f"ERROR: {e}", 0

        try:
            ollama_resp, ollama_time = query_ollama(prompt)
        except Exception as e:
            ollama_resp, ollama_time = f"ERROR: {e}", 0

        results.append({
            "idx": i, "tier": q.get("tier", 0),
            "openai_response": openai_resp,
            "ollama_response": ollama_resp,
            "openai_latency": openai_time,
            "ollama_latency": ollama_time,
        })

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(queries)}] done")

    # Filter valid pairs
    valid = [r for r in results
             if not r["openai_response"].startswith("ERROR")
             and not r["ollama_response"].startswith("ERROR")
             and len(r["openai_response"]) > 10
             and len(r["ollama_response"]) > 10]

    print(f"\nValid pairs: {len(valid)}/{len(results)}")

    # Compute BERTScore
    openai_texts = [r["openai_response"] for r in valid]
    ollama_texts = [r["ollama_response"] for r in valid]

    try:
        from bert_score import score as bert_score_fn
        print("\nComputing BERTScore...")
        P, R, F1 = bert_score_fn(ollama_texts, openai_texts, lang="en", verbose=False)
        bs_p = P.mean().item()
        bs_r = R.mean().item()
        bs_f1 = F1.mean().item()
        method = "bert-score"

        # Per-tier breakdown
        tier_scores = {}
        for r, f1_val in zip(valid, F1.tolist()):
            t = r["tier"]
            tier_scores.setdefault(t, []).append(f1_val)
    except ImportError:
        print("\nWARNING: bert-score not installed. Using word overlap fallback.")
        scores = []
        tier_scores = {}
        for r in valid:
            o_words = set(r["openai_response"].lower().split())
            l_words = set(r["ollama_response"].lower().split())
            overlap = len(o_words & l_words) / max(len(o_words | l_words), 1)
            scores.append(overlap)
            tier_scores.setdefault(r["tier"], []).append(overlap)
        bs_p = bs_r = bs_f1 = sum(scores) / len(scores)
        method = "word_overlap"

    # Quality preservation rate
    threshold = 0.80
    if method == "bert-score":
        above = sum(1 for f in F1.tolist() if f >= threshold)
    else:
        above = sum(1 for s in scores if s >= threshold)
    preservation_rate = above / max(len(valid), 1)

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\nBERTScore (Ollama vs OpenAI as reference):")
    print(f"  Precision: {bs_p:.4f}")
    print(f"  Recall:    {bs_r:.4f}")
    print(f"  F1:        {bs_f1:.4f}")
    print(f"  Method:    {method}")
    print(f"\nQuality Preservation Rate (F1 ≥ {threshold}): {preservation_rate:.1%} ({above}/{len(valid)})")

    labels = {0: "T0_Public", 1: "T1_Internal", 2: "T2_Limited", 3: "T3_Restricted"}
    print(f"\nPer-tier BERTScore F1:")
    for t in sorted(tier_scores.keys()):
        vals = tier_scores[t]
        avg = sum(vals) / len(vals)
        print(f"  {labels.get(t, f'T{t}')}: {avg:.4f} (n={len(vals)})")

    print(f"\nLatency comparison:")
    openai_avg = sum(r["openai_latency"] for r in valid) / len(valid)
    ollama_avg = sum(r["ollama_latency"] for r in valid) / len(valid)
    print(f"  OpenAI avg: {openai_avg*1000:.0f}ms")
    print(f"  Ollama avg: {ollama_avg*1000:.0f}ms")
    print(f"  Ratio: {ollama_avg/max(openai_avg, 0.001):.1f}x slower on-prem")

    # Save
    summary = {
        "n_pairs": len(valid),
        "bertscore_precision": round(bs_p, 4),
        "bertscore_recall": round(bs_r, 4),
        "bertscore_f1": round(bs_f1, 4),
        "quality_preservation_rate": round(preservation_rate, 4),
        "quality_threshold": threshold,
        "method": method,
        "per_tier_f1": {labels.get(t, f"T{t}"): round(sum(v)/len(v), 4)
                        for t, v in tier_scores.items()},
        "openai_avg_latency_ms": round(openai_avg * 1000, 1),
        "ollama_avg_latency_ms": round(ollama_avg * 1000, 1),
    }

    with open(out_dir / "quality_comparison.json", "w") as f:
        json.dump(summary, f, indent=2)

    # LaTeX table
    latex = """\\begin{table}[t]
\\centering
\\caption{Response quality comparison: on-premises Llama-3.1-8B vs.\\ public GPT-4o-mini.}
\\label{tab:quality}
\\begin{tabular}{lcccc}
\\toprule
\\textbf{Tier} & \\textbf{n} & \\textbf{BERTScore F1} & \\textbf{Preservation} & \\textbf{Latency Ratio} \\\\
\\midrule
"""
    for t in sorted(tier_scores.keys()):
        vals = tier_scores[t]
        avg = sum(vals) / len(vals)
        above_t = sum(1 for v in vals if v >= threshold)
        pres = above_t / len(vals)
        latex += f"{labels.get(t, f'T{t}')} & {len(vals)} & {avg:.3f} & {pres:.0%} & --- \\\\\n"

    latex += f"""\\midrule
Overall & {len(valid)} & {bs_f1:.3f} & {preservation_rate:.0%} & {ollama_avg/max(openai_avg,0.001):.1f}$\\times$ \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
    with open(out_dir / "quality_table.tex", "w") as f:
        f.write(latex)

    print(f"\nSaved: {out_dir / 'quality_comparison.json'}")
    print(f"LaTeX: {out_dir / 'quality_table.tex'}")


if __name__ == "__main__":
    main()
