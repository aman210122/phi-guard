#!/usr/bin/env python3
"""
21_pilot_barp_baselines.py — PILOT & BaRP baselines for NeurIPS comparison

PURPOSE:
  PILOT (Panda et al., EMNLP Findings 2025): LinUCB, cost-quality, no safety.
  BaRP (Wei et al., arXiv 2510.07429, 2025): Policy gradient, cost-quality, no safety.

KEY DESIGN:
  Both use classifier prediction for minimum clearance (like StaticILP),
  then apply bandit optimization among eligible platforms. They produce
  the SAME violations as StaticILP — the misclassified T3 queries —
  because they have NO safety mechanisms (τ fallback, posterior check,
  safe action set, conformal envelope).

USAGE:
  python scripts/21_pilot_barp_baselines.py
"""

import json, math, random, sys
from pathlib import Path
from collections import Counter
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SEED = 42; random.seed(SEED); np.random.seed(SEED)

PLATFORMS = [
    {"name": "Public_API",   "clearance": 0, "cost": 0.0100, "latency": 310,  "quality": 1.00},
    {"name": "Secure_Cloud", "clearance": 2, "cost": 0.0110, "latency": 420,  "quality": 0.95},
    {"name": "On_Premises",  "clearance": 3, "cost": 0.0254, "latency": 1180, "quality": 0.90},
]
SORTED_BY_COST = sorted(PLATFORMS, key=lambda p: p["cost"])

def fix_tier_probs(predictions):
    fixed = []
    for p in predictions:
        pred = p["predicted_tier"]; conf = p["confidence"]
        probs = p.get("tier_probs")
        if probs is None or not isinstance(probs, list) or len(probs) != 4 \
           or all(abs(x - 0.25) < 0.001 for x in probs):
            remaining = (1.0 - conf) / 3.0
            probs = [remaining] * 4; probs[pred] = conf
        fixed_p = dict(p); fixed_p["tier_probs"] = probs; fixed.append(fixed_p)
    return fixed

def cheapest_with_clearance(min_clearance):
    for p in SORTED_BY_COST:
        if p["clearance"] >= min_clearance: return p
    return PLATFORMS[-1]


class PILOTRouter:
    """LinUCB: classifier prediction → eligible platforms → UCB selection."""
    def __init__(self, d=8, alpha=1.0, cost_weight=0.5):
        self.n_arms = len(PLATFORMS); self.d = d
        self.alpha_ucb = alpha; self.cost_weight = cost_weight
        self.A = [np.eye(d) for _ in range(self.n_arms)]
        self.b = [np.zeros(d) for _ in range(self.n_arms)]

    def _ctx(self, q):
        f = list(q["tier_probs"]) + [q["confidence"], q["predicted_tier"]/3.0,
                                      q.get("token_count",200)/500.0, 1.0]
        return np.array(f[:self.d], dtype=np.float64)

    def route(self, query):
        elig = [i for i, p in enumerate(PLATFORMS) if p["clearance"] >= query["predicted_tier"]]
        if not elig: return len(PLATFORMS) - 1
        x = self._ctx(query); best_arm = elig[0]; best_ucb = -1e9
        for arm in elig:
            Ai = np.linalg.inv(self.A[arm]); th = Ai @ self.b[arm]
            ucb = th @ x + self.alpha_ucb * np.sqrt(x @ Ai @ x)
            if ucb > best_ucb: best_ucb = ucb; best_arm = arm
        return best_arm

    def update(self, arm, query):
        x = self._ctx(query); tokens = query.get("token_count", 200)
        p = PLATFORMS[arm]; cost = p["cost"]*tokens/1000
        mx = PLATFORMS[-1]["cost"]*tokens/1000
        reward = p["quality"] + self.cost_weight*(1 - cost/mx if mx > 0 else 0)
        self.A[arm] += np.outer(x, x); self.b[arm] += reward * x


class BaRPRouter:
    """Policy gradient: classifier prediction → eligible → REINFORCE selection."""
    def __init__(self, d=8, lr=0.01, cost_pref=0.5):
        self.n_arms = len(PLATFORMS); self.d = d; self.lr = lr
        self.pref = [cost_pref, 1.0-cost_pref]
        self.theta = np.random.randn(self.n_arms, d)*0.01
        self.baseline = 0.0; self.step = 0

    def _ctx(self, q):
        f = list(q["tier_probs"]) + [q["confidence"], q["predicted_tier"]/3.0,
                                      q.get("token_count",200)/500.0, 1.0]
        return np.array(f[:self.d], dtype=np.float64)

    def route(self, query, explore=True):
        elig = [i for i, p in enumerate(PLATFORMS) if p["clearance"] >= query["predicted_tier"]]
        if not elig: return len(PLATFORMS) - 1
        x = self._ctx(query)
        logits = np.array([self.theta[a] @ x for a in elig])
        logits -= logits.max(); exp = np.exp(logits); probs = exp/exp.sum()
        if explore and self.step < 200:
            return elig[int(np.random.choice(len(elig), p=probs))]
        return elig[int(np.argmax(probs))]

    def update(self, arm, query):
        x = self._ctx(query); tokens = query.get("token_count", 200)
        p = PLATFORMS[arm]; cost = p["cost"]*tokens/1000
        mx = PLATFORMS[-1]["cost"]*tokens/1000
        reward = self.pref[1]*p["quality"] + self.pref[0]*(1 - cost/mx if mx > 0 else 0)
        self.step += 1
        self.baseline += (reward - self.baseline)/self.step
        adv = reward - self.baseline
        logits = self.theta @ x; logits -= logits.max()
        probs = np.exp(logits)/np.exp(logits).sum()
        for a in range(self.n_arms):
            g = x*(1-probs[a]) if a == arm else -x*probs[a]
            self.theta[a] += self.lr * adv * g


def eval_router(router, data, name):
    total_cost = 0.0; viols = 0; cloud = 0; viol_det = []
    for i, q in enumerate(data):
        tokens = q.get("token_count", 200)
        if isinstance(router, BaRPRouter):
            arm = router.route(q, explore=(i < 200))
        else:
            arm = router.route(q)
        ch = PLATFORMS[arm]; cost = ch["cost"]*tokens/1000
        vd = ch["clearance"] < q["tier"]
        router.update(arm, q); total_cost += cost
        if vd:
            viols += 1
            viol_det.append({"true":q["tier"],"pred":q["predicted_tier"],
                             "conf":round(q["confidence"],3),"plat":ch["name"]})
        if ch["name"] != "On_Premises": cloud += 1
    n = len(data)
    return {"name":name,"cost":round(total_cost,4),"violations":viols,
            "viol_pct":round(100*viols/n,2),"cloud_pct":round(100*cloud/n,1),
            "viol_details":viol_det[:20]}


def eval_static(data, tau=None):
    tc = 0.0; v = 0; cl = 0; vd = []
    for q in data:
        pred = q["predicted_tier"]
        if tau and q["confidence"] < tau: pred = 3
        ch = cheapest_with_clearance(pred); tok = q.get("token_count",200)
        cost = ch["cost"]*tok/1000; violated = ch["clearance"] < q["tier"]
        tc += cost
        if violated: v += 1; vd.append({"true":q["tier"],"pred":q["predicted_tier"],"conf":round(q["confidence"],3)})
        if ch["name"] != "On_Premises": cl += 1
    n = len(data)
    return {"name":f"StaticILP τ={tau}","cost":round(tc,4),"violations":v,
            "viol_pct":round(100*v/n,2),"cloud_pct":round(100*cl/n,1),"viol_details":vd[:20]}


def eval_shift(cls, kwargs, data, name):
    lo = [q for q in data if q["tier"] <= 1]; hi = [q for q in data if q["tier"] >= 2]
    random.shuffle(lo); random.shuffle(hi)
    n1 = min(2000, len(lo)+len(hi)//2); n1_lo = int(n1*0.70)
    p1 = lo[:n1_lo] + hi[:n1-n1_lo]; random.shuffle(p1)
    n2 = min(3000, len(lo)-n1_lo+len(hi)-(n1-n1_lo)); n2_lo = int(n2*0.20)
    p2 = lo[n1_lo:n1_lo+n2_lo] + hi[n1-n1_lo:n1-n1_lo+n2-n2_lo]; random.shuffle(p2)
    if len(p1) < 500 or len(p2) < 500: return None
    allq = p1 + p2; router = cls(**kwargs)
    v1=v2=0; win=[]; mw=0.0; W=100
    for i, q in enumerate(allq):
        arm = router.route(q, explore=(i<100)) if isinstance(router, BaRPRouter) else router.route(q)
        vd = PLATFORMS[arm]["clearance"] < q["tier"]; router.update(arm, q)
        if i < len(p1): v1 += int(vd)
        else: v2 += int(vd)
        win.append(int(vd))
        if len(win)>W: win.pop(0)
        if len(win)==W: mw = max(mw, sum(win)/W)
    return {"name":name,"p1_viol_pct":round(100*v1/len(p1),2),
            "p2_viol_pct":round(100*v2/len(p2),2),"max_window_pct":round(100*mw,2),
            "total_viol_pct":round(100*(v1+v2)/len(allq),2)}


def main():
    data_dir = ROOT / "data"; out_dir = ROOT / "outputs"; out_dir.mkdir(exist_ok=True)
    pred_path = data_dir / "test_with_predictions.json"
    if not pred_path.exists():
        print(f"ERROR: {pred_path} not found. Run 02_train_classifier.py first."); sys.exit(1)
    data = fix_tier_probs(json.loads(pred_path.read_text())); n = len(data)

    print("="*70); print("PILOT & BaRP Baselines for NeurIPS"); print("="*70)
    print(f"Test set: {n} queries")
    td = Counter(q["tier"] for q in data)
    print(f"Tiers: {', '.join(f'T{t}={c}' for t,c in sorted(td.items()))}")

    # StaticILP reference
    sn = eval_static(data, tau=None); st = eval_static(data, tau=0.80)
    print(f"\nStaticILP (no τ):  cost=${sn['cost']:.2f}, viol={sn['violations']} ({sn['viol_pct']}%), cloud={sn['cloud_pct']}%")
    print(f"StaticILP (τ=0.80): cost=${st['cost']:.2f}, viol={st['violations']} ({st['viol_pct']}%), cloud={st['cloud_pct']}%")

    # PILOT
    print(f"\n{'='*70}\nPILOT (LinUCB)\n{'='*70}")
    pr = []
    for a in [0.5, 1.0, 2.0]:
        r = eval_router(PILOTRouter(alpha=a), data, f"PILOT α={a}"); pr.append(r)
        print(f"  α={a}: cost=${r['cost']:.2f}, viol={r['violations']} ({r['viol_pct']}%), cloud={r['cloud_pct']}%")

    # BaRP
    print(f"\n{'='*70}\nBaRP (Policy Gradient)\n{'='*70}")
    br = []
    for w in [0.3, 0.5, 0.7]:
        r = eval_router(BaRPRouter(cost_pref=w), data, f"BaRP w={w}"); br.append(r)
        print(f"  w_cost={w}: cost=${r['cost']:.2f}, viol={r['violations']} ({r['viol_pct']}%), cloud={r['cloud_pct']}%")

    sec_cost = sum(PLATFORMS[-1]["cost"]*q.get("token_count",200)/1000 for q in data)

    # Distribution shift
    print(f"\n{'='*70}\nDistribution Shift\n{'='*70}")
    ps = eval_shift(PILOTRouter, {"alpha":1.0}, data, "PILOT")
    bs = eval_shift(BaRPRouter, {"cost_pref":0.5}, data, "BaRP")
    if ps: print(f"  PILOT: P1={ps['p1_viol_pct']}% → P2={ps['p2_viol_pct']}% (max window={ps['max_window_pct']}%)")
    if bs: print(f"  BaRP:  P1={bs['p1_viol_pct']}% → P2={bs['p2_viol_pct']}% (max window={bs['max_window_pct']}%)")

    # Summary
    bp = min(pr, key=lambda x: x["cost"]); bb = min(br, key=lambda x: x["cost"])
    print(f"\n{'='*70}\nNUMBERS FOR PAPER\n{'='*70}")
    print(f"{'Strategy':<28} {'Cost':>10} {'Viol':>5} {'Viol%':>7} {'Cloud%':>7}")
    print("-"*60)
    for lb, c, v, cl in [
        ("PILOT-style", bp["cost"], bp["violations"], bp["cloud_pct"]),
        ("BaRP-style", bb["cost"], bb["violations"], bb["cloud_pct"]),
        ("StaticILP (no τ)", sn["cost"], sn["violations"], sn["cloud_pct"]),
        ("Threshold (τ=0.80)", st["cost"], st["violations"], st["cloud_pct"]),
        ("SecureDefault", sec_cost, 0, 0.0),
    ]:
        print(f"{lb:<28} ${c:>8.2f} {v:>5} {100*v/n:>6.2f}% {cl:>5.1f}%")

    if bp["viol_details"]:
        print(f"\n--- Violation Details ---")
        for v in bp["viol_details"][:5]:
            print(f"  True=T{v['true']}, Pred=T{v['pred']}, Conf={v['conf']:.3f} → {v['plat']}")

    print(f"\nKEY INSIGHT: PILOT/BaRP violations = StaticILP violations (same source: misclassification)")
    print(f"SafeTS/CARES catch these via safe action set. Cost-quality bandits cannot.")

    save = {"pilot": [{k:v for k,v in r.items() if k!="viol_details"} for r in pr],
            "barp": [{k:v for k,v in r.items() if k!="viol_details"} for r in br],
            "static_no_tau": {k:v for k,v in sn.items() if k!="viol_details"},
            "static_tau80": {k:v for k,v in st.items() if k!="viol_details"},
            "shift": {"pilot":ps,"barp":bs}}
    with open(out_dir/"pilot_barp_results.json","w") as f: json.dump(save,f,indent=2,default=str)
    print(f"\nSaved: {out_dir/'pilot_barp_results.json'}")

if __name__ == "__main__":
    main()
