"""Control-latency sweep on the stair course (MuJoCo, rough/stairs policy): PD targets reach the motors N ms late.
Same 100 course episodes as `sim2sim.py --stairs` (delay 0 = that run).

Usage: python stairs_delay.py policies/h1_stairs.onnx [--episodes 100] [--out results/mujoco_stairs_delay.json]
"""
import argparse
import json

from sim2sim import H1, evaluate_stairs

DELAYS_MS = (0, 10, 20, 30, 40, 60)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("onnx")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--out", default="results/mujoco_stairs_delay.json")
    a = p.parse_args()
    out = {}
    for d in DELAYS_MS:
        out[str(d)] = evaluate_stairs(H1(a.onnx, delay_ms=d, course_on=True), a.episodes)
        print(f"delay {d:3d} ms", {k: sum(e["outcome"] == k for e in out[str(d)]) for k in ("success", "fall", "stuck")},
              flush=True)
    json.dump(out, open(a.out, "w"), indent=1)
