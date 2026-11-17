"""Stair course (course.py) in Isaac Lab with the rough-terrain policy.onnx: same episodes as `sim2sim.py --stairs`
(one env per episode). Also records env 0's height-scan input every second for `sim2sim.py --check-scan`.

Usage: python stairs_isaac.py <policy.onnx> --episodes 100 --out results/isaac_stairs.json   (Isaac Lab venv)
       python stairs_isaac.py <policy.onnx> --gui --episodes 1 [--vx 0.65]   Isaac window on the course (screen-record it)
"""
import argparse
import json

import onnxruntime as ort  # must load before Kit starts (WinError 1114 otherwise)
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("onnx")
parser.add_argument("--episodes", type=int, default=100)
parser.add_argument("--out")
parser.add_argument("--gui", action="store_true", help="open the Isaac window (use --episodes 1: envs overlap)")
parser.add_argument("--vx", type=float, help="same forward speed for every episode instead of course.stair_command(i)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = not args.gui
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: E402,F401
from isaaclab.terrains import SubTerrainBaseCfg, TerrainGeneratorCfg  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import course as c  # noqa: E402
import h1_isaac as h  # noqa: E402

TASK = "Isaac-Velocity-Rough-H1-Play-v0"  # no obs noise, no pushes
N = args.episodes
cfg = parse_env_cfg(TASK, num_envs=N)
cfg.scene.terrain.terrain_generator = TerrainGeneratorCfg(
    size=c.TILE, num_rows=1, num_cols=1, border_width=5.0, curriculum=False, use_cache=False,
    sub_terrains={"course": SubTerrainBaseCfg(function=c.isaac_course_terrain, size=c.TILE)})
cfg.scene.terrain.max_init_terrain_level = None
cfg.curriculum.terrain_levels = None
cfg.events.reset_base.params["pose_range"] = {}  # every env starts at the origin facing +x (envs don't collide)
cmd = cfg.commands.base_velocity
cmd.heading_command, cmd.rel_heading_envs, cmd.rel_standing_envs = True, 1.0, 0.0  # hold heading 0 (course.heading_wz)
cmd.ranges.heading, cmd.resampling_time_range = (0.0, 0.0), (1e6, 1e6)
cfg.episode_length_s = c.STAIR_SECONDS + 1.0
if args.gui:  # side view of the course, no command arrows
    cmd.debug_vis = False
    cfg.viewer.eye, cfg.viewer.lookat = (2.4, -5.0, 1.8), (2.4, 0.0, 0.6)
env = gym.make(TASK, cfg=cfg)
u = env.unwrapped
robot, scanner = u.scene["robot"], u.scene["height_scanner"]
assert torch.allclose(u.scene.env_origins, torch.zeros_like(u.scene.env_origins)), u.scene.env_origins[:3]

env.reset()
vx = np.full(N, args.vx) if args.vx else np.array([c.stair_command(i) for i in range(N)])
term = u.command_manager.get_term("base_velocity")
term.vel_command_b[:] = 0.0
term.vel_command_b[:, 0] = torch.tensor(vx, dtype=torch.float, device=u.device)
obs = u.observation_manager.compute()["policy"]
assert obs.shape[1] == 69 + len(h.SCAN_GRID), obs.shape
sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])

dt = h.SIM_DT * h.DECIMATION
outcome, t_end, max_x, scan_log = [None] * N, [c.STAIR_SECONDS] * N, np.zeros(N), []
for t in range(int(c.STAIR_SECONDS / dt)):
    o = obs.cpu().numpy()
    if t % 50 == 0 and outcome[0] is None:
        scan_log.append({"t": t, "pos": scanner.data.pos_w[0].tolist(), "quat": scanner.data.quat_w[0].tolist(),
                         "scan": o[0, 69:].tolist()})
    act = np.concatenate([sess.run(None, {"obs": o[k : k + 1]})[0] for k in range(N)])  # exported ONNX is batch 1
    obs, _, terminated, _, _ = env.step(torch.from_numpy(act).to(u.device))
    obs = obs["policy"]
    p, term_np = robot.data.root_pos_w.cpu().numpy(), terminated.cpu().numpy()
    for k in range(N):
        if outcome[k] is not None:
            continue
        # terminated envs are already auto-reset, so their pose is not used; base_contact termination counts as a fall
        if term_np[k] or p[k, 2] - c.ground_z(p[k, 0], p[k, 1]) < h.FALL_Z:
            outcome[k], t_end[k] = "fall", (t + 1) * dt
        elif p[k, 0] >= c.GOAL_X:
            outcome[k], t_end[k] = "success", (t + 1) * dt
        else:
            max_x[k] = max(max_x[k], p[k, 0])
    if all(o is not None for o in outcome):
        break

res = [c.stair_result(float(vx[k]), outcome[k] or "stuck", t_end[k], max_x[k]) for k in range(N)]
print("ISAAC stairs", {k: sum(r["outcome"] == k for r in res) for k in ("success", "fall", "stuck")})
if args.out:
    json.dump({"episodes": res, "scan_log": scan_log}, open(args.out, "w"), indent=1)
env.close()
app.close()
