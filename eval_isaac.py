"""Evaluate policy.onnx in Isaac Lab with the same episodes as `sim2sim.py --headless` (one env per episode).

Usage: python eval_isaac.py <policy.onnx> --episodes 100 --out isaac.json   (Isaac Lab venv)
"""
import argparse
import json

import onnxruntime as ort  # must load before Kit starts: importing it after AppLauncher fails (WinError 1114)
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("onnx")
parser.add_argument("--episodes", type=int, default=100)
parser.add_argument("--out", required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: E402,F401
from isaaclab.utils.math import quat_apply_inverse, yaw_quat  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import h1_isaac as h  # noqa: E402

TASK = "Isaac-Velocity-Flat-H1-Play-v0"  # no obs noise, no pushes
N = args.episodes
cfg = parse_env_cfg(TASK, num_envs=N)
c = cfg.commands.base_velocity
c.heading_command, c.rel_standing_envs, c.resampling_time_range = False, 0.0, (1e6, 1e6)  # keep our fixed command
cfg.episode_length_s = h.EVAL_SECONDS + 1.0  # no time-out reset inside the window
env = gym.make(TASK, cfg=cfg)
u = env.unwrapped
robot = u.scene["robot"]
cmds = np.stack([h.eval_command(i) for i in range(N)])

env.reset()
u.command_manager.get_term("base_velocity").vel_command_b[:] = torch.from_numpy(cmds).to(u.device)
obs = u.observation_manager.compute()["policy"]  # recompute with our commands
sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])

alive = np.ones(N, bool)
fall_step = [None] * N
v_xy, wz = [[] for _ in range(N)], [[] for _ in range(N)]
for t in range(int(h.EVAL_SECONDS / (h.SIM_DT * h.DECIMATION))):
    o = obs.cpu().numpy()
    act = np.concatenate([sess.run(None, {"obs": o[k : k + 1]})[0] for k in range(N)])  # exported ONNX is batch 1
    obs, _, terminated, _, _ = env.step(torch.from_numpy(act).to(u.device))
    obs = obs["policy"]
    # terminated envs are already auto-reset, so their pose is not used; base_contact termination counts as a fall
    fell = alive & ((robot.data.root_pos_w[:, 2] < h.FALL_Z) | terminated).cpu().numpy()
    vel = quat_apply_inverse(yaw_quat(robot.data.root_quat_w), robot.data.root_lin_vel_w)[:, :2].cpu().numpy()
    yaw_rate = robot.data.root_ang_vel_w[:, 2].cpu().numpy()
    for k in np.flatnonzero(fell):
        fall_step[k] = t
    alive &= ~fell
    for k in np.flatnonzero(alive):
        v_xy[k].append(vel[k])
        wz[k].append(yaw_rate[k])

res = [h.episode_result(cmds[k], fall_step[k], v_xy[k], wz[k]) for k in range(N)]
print(f"ISAAC falls {sum(r['fell'] for r in res)}/{N}")
json.dump(res, open(args.out, "w"), indent=1)
env.close()
app.close()
