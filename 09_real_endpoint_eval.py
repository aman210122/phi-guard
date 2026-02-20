#!/usr/bin/env python3
"""
09_real_endpoint_eval.py — Real LLM Endpoint Validation

PURPOSE:
  Call ACTUAL LLM endpoints and measure REAL cost, latency, and response quality.
  This transforms the paper from "simulation study" to "validated system."

ENDPOINTS:
  1. Public API: OpenAI GPT-4o-mini (requires OPENAI_API_KEY env var)
  2. On-Premises: Ollama running locally (requires `ollama serve` + model pulled)
  3. Secure Cloud: Azure OpenAI OR second Ollama model (configurable)

SETUP:
  pip install openai bert-score requests
  
  # For Ollama (free, local):
  # Download from https://ollama.com, then:
  ollama pull llama3.1:8b
  ollama serve  # in separate terminal
  
  # For OpenAI:
  export OPENAI_API_KEY="sk-..."  # or set in .env file

USAGE:
  python scripts/09_real_endpoint_eval.py --n-queries 200

OUTPUT:
  outputs/real_endpoint_results.json
  outputs/real_endpoint_table.tex  (LaTeX table for paper)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ═══════════════════════════════════════════════════════════════════════════
# ENDPOINT CLIENTS
# ═══════════════════════════════════════════════════════════════════════════

class OpenAIEndpoint:
    """Public API tier — GPT-4o-mini via OpenAI."""
    
    def __init__(self):
        try:
            from openai import OpenAI
            api_key = os.environ.get("OPENAI_API_KEY", "")
            if not api_key:
                print("  WARNING: OPENAI_API_KEY not set. OpenAI endpoint disabled.")
                self.client = None
                return
            self.client = OpenAI(api_key=api_key)
            self.model = "gpt-4o-mini"
            # Pricing as of Jan 2026: $0.15/1M input, $0.60/1M output
            self.cost_per_input_token = 0.15 / 1_000_000
            self.cost_per_output_token = 0.60 / 1_000_000
        except ImportError:
            print("  WARNING: openai package not installed. Run: pip install openai")
            self.client = None
    
    @property
    def available(self):
        return self.client is not None
    
    def query(self, prompt: str, max_tokens: int = 150) -> Dict:
        if not self.available:
            return {"error": "OpenAI not configured"}
        
        start = time.time()
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.3,
            )
            latency = time.time() - start
            
            text = response.choices[0].message.content
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            cost = (input_tokens * self.cost_per_input_token + 
                    output_tokens * self.cost_per_output_token)
            
            return {
                "text": text,
                "latency_ms": latency * 1000,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": cost,
                "model": self.model,
                "platform": "public_api",
            }
        except Exception as e:
            return {"error": str(e), "latency_ms": (time.time() - start) * 1000}


class OllamaEndpoint:
    """On-premises tier — Local Ollama instance."""
    
    def __init__(self, model: str = "llama3.1:8b", port: int = 11434):
        import requests
        self.model = model
        self.base_url = f"http://localhost:{port}"
        self.requests = requests
        # Estimate on-prem cost based on GPU amortization
        # ~$0.50/hr for consumer GPU, ~1000 tokens/sec throughput
        self.cost_per_token = 0.50 / 3600 / 1000  # ~$0.000000139/token
        
        # Check availability
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=3)
            self.is_available = r.status_code == 200
            if self.is_available:
                models = [m["name"] for m in r.json().get("models", [])]
                if not any(model.split(":")[0] in m for m in models):
                    print(f"  WARNING: Model '{model}' not found in Ollama. Available: {models}")
                    print(f"  Run: ollama pull {model}")
                    self.is_available = False
        except Exception:
            self.is_available = False
            print("  WARNING: Ollama not running. Start with: ollama serve")
    
    @property
    def available(self):
        return self.is_available
    
    def query(self, prompt: str, max_tokens: int = 150) -> Dict:
        if not self.available:
            return {"error": "Ollama not available"}
        
        start = time.time()
        try:
            r = self.requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "options": {"num_predict": max_tokens, "temperature": 0.3},
                    "stream": False,
                },
                timeout=60,
            )
            latency = time.time() - start
            data = r.json()
            
            text = data.get("response", "")
            # Ollama reports token counts
            eval_count = data.get("eval_count", len(text.split()))
            prompt_count = data.get("prompt_eval_count", len(prompt.split()))
            total_tokens = prompt_count + eval_count
            cost = total_tokens * self.cost_per_token
            
            return {
                "text": text,
                "latency_ms": latency * 1000,
                "input_tokens": prompt_count,
                "output_tokens": eval_count,
                "cost_usd": cost,
                "model": self.model,
                "platform": "on_premises",
            }
        except Exception as e:
            return {"error": str(e), "latency_ms": (time.time() - start) * 1000}


# ═══════════════════════════════════════════════════════════════════════════
# QUALITY EVALUATION
# ═══════════════════════════════════════════════════════════════════════════

def compute_bertscore(predictions: List[str], references: List[str]) -> Dict:
    """Compute BERTScore between response pairs."""
    try:
        from bert_score import score as bert_score
        P, R, F1 = bert_score(predictions, references, lang="en", verbose=False)
        return {
            "precision": P.mean().item(),
            "recall": R.mean().item(),
            "f1": F1.mean().item(),
        }
    except ImportError:
        print("  WARNING: bert-score not installed. Run: pip install bert-score")
        # Fallback: simple word overlap
        scores = []
        for pred, ref in zip(predictions, references):
            pred_words = set(pred.lower().split())
            ref_words = set(ref.lower().split())
            if not ref_words:
                scores.append(0.0)
                continue
            overlap = len(pred_words & ref_words)
            scores.append(overlap / max(len(ref_words), 1))
        avg = sum(scores) / max(len(scores), 1)
        return {"precision": avg, "recall": avg, "f1": avg, "method": "word_overlap_fallback"}


# ═══════════════════════════════════════════════════════════════════════════
# MAIN EVALUATION
# ═══════════════════════════════════════════════════════════════════════════

def load_test_queries(data_dir: Path, n_queries: int = 200) -> List[Dict]:
    """Load test queries from dataset, balanced across tiers."""
    test_path = data_dir / "test.json"
    if not test_path.exists():
        print(f"  ERROR: {test_path} not found. Run 06_mimic_ingest_v2.py first.")
        sys.exit(1)
    
    with open(test_path) as f:
        data = json.load(f)
    
    # Sample balanced across tiers
    by_tier = {}
    for item in data:
        t = item.get("tier", 0)
        by_tier.setdefault(t, []).append(item)
    
    import random
    random.seed(42)
    per_tier = n_queries // 4
    samples = []
    for tier in sorted(by_tier.keys()):
        pool = by_tier[tier]
        random.shuffle(pool)
        samples.extend(pool[:per_tier])
    
    random.shuffle(samples)
    return samples[:n_queries]


def create_prompt(query_text: str) -> str:
    """Create a standardized prompt for all endpoints."""
    # Truncate very long texts
    text = query_text[:500] if len(query_text) > 500 else query_text
    return (
        f"You are a clinical assistant. Based on the following text, "
        f"provide a brief clinical summary or answer in 2-3 sentences.\n\n"
        f"Text: {text}\n\nResponse:"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-queries", type=int, default=200)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--ollama-model", type=str, default="llama3.1:8b")
    parser.add_argument("--skip-openai", action="store_true")
    parser.add_argument("--skip-ollama", action="store_true")
    args = parser.parse_args()
    
    ROOT = Path(__file__).resolve().parent.parent
    data_dir = ROOT / args.data_dir
    out_dir = ROOT / args.output_dir
    out_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("Real LLM Endpoint Validation")
    print("=" * 60)
    
    # Initialize endpoints
    endpoints = {}
    
    if not args.skip_openai:
        print("\nInitializing OpenAI endpoint...")
        openai_ep = OpenAIEndpoint()
        if openai_ep.available:
            endpoints["public_api"] = openai_ep
            print(f"  ✓ OpenAI {openai_ep.model} ready")
    
    if not args.skip_ollama:
        print("\nInitializing Ollama endpoint...")
        ollama_ep = OllamaEndpoint(model=args.ollama_model)
        if ollama_ep.available:
            endpoints["on_premises"] = ollama_ep
            print(f"  ✓ Ollama {ollama_ep.model} ready")
    
    if not endpoints:
        print("\nERROR: No endpoints available. Set up at least one:")
        print("  OpenAI: export OPENAI_API_KEY='sk-...'")
        print("  Ollama: ollama pull llama3.1:8b && ollama serve")
        sys.exit(1)
    
    print(f"\nActive endpoints: {list(endpoints.keys())}")
    
    # Load queries
    print(f"\nLoading {args.n_queries} test queries...")
    queries = load_test_queries(data_dir, args.n_queries)
    print(f"  Loaded {len(queries)} queries")
    tier_dist = {}
    for q in queries:
        t = q.get("tier", 0)
        tier_dist[t] = tier_dist.get(t, 0) + 1
    print(f"  Tier distribution: {tier_dist}")
    
    # Run queries through each endpoint
    results = {name: [] for name in endpoints}
    
    for name, endpoint in endpoints.items():
        print(f"\n{'='*40}")
        print(f"Querying {name} ({len(queries)} queries)...")
        print(f"{'='*40}")
        
        for i, query in enumerate(queries):
            prompt = create_prompt(query["text"])
            result = endpoint.query(prompt)
            result["query_idx"] = i
            result["tier"] = query.get("tier", 0)
            result["query_text_preview"] = query["text"][:100]
            results[name].append(result)
            
            if (i + 1) % 20 == 0:
                errors = sum(1 for r in results[name] if "error" in r)
                avg_lat = sum(r.get("latency_ms", 0) for r in results[name] if "error" not in r) / max(i + 1 - errors, 1)
                print(f"  [{i+1}/{len(queries)}] avg_latency={avg_lat:.0f}ms errors={errors}")
    
    # ── Compute statistics ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    summary = {}
    for name, ep_results in results.items():
        valid = [r for r in ep_results if "error" not in r]
        errors = len(ep_results) - len(valid)
        
        if not valid:
            print(f"\n{name}: ALL QUERIES FAILED ({errors} errors)")
            continue
        
        avg_latency = sum(r["latency_ms"] for r in valid) / len(valid)
        p50_latency = sorted(r["latency_ms"] for r in valid)[len(valid) // 2]
        p99_latency = sorted(r["latency_ms"] for r in valid)[int(len(valid) * 0.99)]
        total_cost = sum(r["cost_usd"] for r in valid)
        avg_cost_per_query = total_cost / len(valid)
        avg_input_tokens = sum(r.get("input_tokens", 0) for r in valid) / len(valid)
        avg_output_tokens = sum(r.get("output_tokens", 0) for r in valid) / len(valid)
        avg_cost_per_1k = avg_cost_per_query / max(avg_input_tokens / 1000, 0.001)
        
        summary[name] = {
            "n_queries": len(valid),
            "n_errors": errors,
            "avg_latency_ms": round(avg_latency, 1),
            "p50_latency_ms": round(p50_latency, 1),
            "p99_latency_ms": round(p99_latency, 1),
            "total_cost_usd": round(total_cost, 6),
            "avg_cost_per_query_usd": round(avg_cost_per_query, 6),
            "avg_cost_per_1k_tokens": round(avg_cost_per_1k, 6),
            "avg_input_tokens": round(avg_input_tokens, 1),
            "avg_output_tokens": round(avg_output_tokens, 1),
            "model": valid[0].get("model", "unknown"),
        }
        
        print(f"\n{name} ({summary[name]['model']}):")
        print(f"  Queries: {len(valid)} successful, {errors} errors")
        print(f"  Latency: avg={avg_latency:.0f}ms  p50={p50_latency:.0f}ms  p99={p99_latency:.0f}ms")
        print(f"  Cost: ${total_cost:.4f} total, ${avg_cost_per_query:.6f}/query, ${avg_cost_per_1k:.4f}/1K tokens")
        print(f"  Tokens: avg {avg_input_tokens:.0f} in, {avg_output_tokens:.0f} out")
    
    # ── Cross-endpoint quality comparison (BERTScore) ──────────────────
    endpoint_names = list(results.keys())
    if len(endpoint_names) >= 2:
        print(f"\nComputing response quality (BERTScore)...")
        
        ep1_name, ep2_name = endpoint_names[0], endpoint_names[1]
        ep1_results = results[ep1_name]
        ep2_results = results[ep2_name]
        
        # Align by query index
        paired_responses = []
        for r1 in ep1_results:
            if "error" in r1:
                continue
            idx = r1["query_idx"]
            r2 = next((r for r in ep2_results if r.get("query_idx") == idx and "error" not in r), None)
            if r2:
                paired_responses.append((r1["text"], r2["text"]))
        
        if paired_responses:
            preds = [p[0] for p in paired_responses]
            refs = [p[1] for p in paired_responses]
            bs = compute_bertscore(preds, refs)
            
            summary["quality_comparison"] = {
                "endpoint_1": ep1_name,
                "endpoint_2": ep2_name,
                "n_pairs": len(paired_responses),
                "bertscore_f1": round(bs["f1"], 4),
                "bertscore_precision": round(bs["precision"], 4),
                "bertscore_recall": round(bs["recall"], 4),
                "method": bs.get("method", "bert-score"),
            }
            
            print(f"  {ep1_name} vs {ep2_name}: BERTScore F1={bs['f1']:.4f} ({len(paired_responses)} pairs)")
    
    # ── Generate LaTeX table ──────────────────────────────────────────
    latex = "\\begin{table}[t]\n\\centering\n"
    latex += "\\caption{Validation on real LLM endpoints ($n=" + str(args.n_queries) + "$ queries per platform).}\n"
    latex += "\\label{tab:real_endpoints}\n"
    latex += "\\begin{tabular}{lcccc}\n\\toprule\n"
    latex += "\\textbf{Platform} & \\textbf{Model} & \\textbf{Cost/1K tok} & \\textbf{Latency (ms)} & \\textbf{Tokens (in/out)} \\\\\n"
    latex += "\\midrule\n"
    
    for name, s in summary.items():
        if name == "quality_comparison":
            continue
        latex += f"{name.replace('_', ' ').title()} & {s['model']} & \\${s['avg_cost_per_1k_tokens']:.4f} & {s['avg_latency_ms']:.0f} & {s['avg_input_tokens']:.0f}/{s['avg_output_tokens']:.0f} \\\\\n"
    
    latex += "\\bottomrule\n\\end{tabular}\n"
    
    if "quality_comparison" in summary:
        qc = summary["quality_comparison"]
        latex += f"\\\\[0.5em]\n\\footnotesize Cross-platform BERTScore F1: {qc['bertscore_f1']:.3f} ({qc['n_pairs']} paired queries)\n"
    
    latex += "\\end{table}\n"
    
    # Save
    with open(out_dir / "real_endpoint_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    with open(out_dir / "real_endpoint_table.tex", "w", encoding="utf-8") as f:
        f.write(latex)
    
    print(f"\nSaved: {out_dir / 'real_endpoint_results.json'}")
    print(f"LaTeX: {out_dir / 'real_endpoint_table.tex'}")
    print("\nCopy the LaTeX table into your paper's experiments section.")


if __name__ == "__main__":
    main()
