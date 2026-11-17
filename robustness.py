"""Robustness sweep in MuJoCo: make one condition worse at a time, measure fall rate (Wilson 95% CI) and tracking.
Same episodes/commands as sim2sim.py --headless. None of these but obs noise were randomized in H1 training.

Usage: python robustness.py <policy.onnx> [--episodes 100] [--out results/robustness.json]
"""
import argparse
import json

from eval_stats import summary
from sim2sim import H1, evaluate

SWEEP = {  # condition: levels (first = nominal)
    "push_n": [0, 100, 200, 300, 400, 600],  # one 0.2 s horizontal shove on torso_link at t = 5 s
    "friction": [1.0, 0.5, 0.3, 0.2, 0.1],  # x sliding friction (nominal MuJoCo 1.0; Isaac training 0.8/0.6)
    "payload_kg": [0, 10, 20, 30, 40],  # extra mass on torso_link
    "delay_ms": [0, 10, 20, 40, 60, 80],  # PD target latency
    "noise": [0, 1, 2, 4, 8],  # x training obs noise
}

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("onnx")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--out", default="results/robustness.json")
    p.add_argument("--only", choices=list(SWEEP), help="run one condition")
    a = p.parse_args()
    out = {}
    for cond, levels in SWEEP.items():
        if a.only and cond != a.only:
            continue
        out[cond] = {}
        for lv in levels:
            if cond == "push_n":
                res = evaluate(H1(a.onnx), a.episodes, push_n=lv)
            else:
                res = evaluate(H1(a.onnx, **{cond: lv}), a.episodes)
            out[cond][str(lv)] = res
            print(f"{cond:>10} = {lv:<5} {summary(res)}", flush=True)
    json.dump(out, open(a.out, "w"), indent=1)
