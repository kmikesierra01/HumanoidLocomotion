"""Stair-course success per training checkpoint (MuJoCo): export each model_<it>.pt to ONNX, run the course.

Usage: python stairs_curve.py <rough run dir> [--episodes 100] [--out results/mujoco_stairs_by_checkpoint.json]
"""
import argparse
import json
import os

from export_ckpt import export
from sim2sim import H1, evaluate_stairs

ITERS = (0, 50, 100, 150, 200, 250, 300, 400, 500, 1000, 2000, 2999)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--out", default="results/mujoco_stairs_by_checkpoint.json")
    a = p.parse_args()
    os.makedirs("policies/ckpt", exist_ok=True)
    out = {}
    for it in ITERS:
        onnx = f"policies/ckpt/stairs_{it}.onnx"
        export(os.path.join(a.run, f"model_{it}.pt"), onnx)
        out[str(it)] = evaluate_stairs(H1(onnx, course_on=True), a.episodes)
        print(it, {k: sum(e["outcome"] == k for e in out[str(it)]) for k in ("success", "fall", "stuck")}, flush=True)
    json.dump(out, open(a.out, "w"), indent=1)
