"""Check h1_isaac.py (sim2sim.py constants) against the real Isaac Lab H1 flat env. Prints "CHECK OK" at the end.

Usage: python check_isaac_cfg.py   (Isaac Lab venv)
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402

import isaaclab_tasks  # noqa: E402,F401  (registers tasks)
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import h1_isaac as s  # noqa: E402

TASK = "Isaac-Velocity-Flat-H1-Play-v0"
env = gym.make(TASK, cfg=parse_env_cfg(TASK, num_envs=1))
u = env.unwrapped
robot = u.scene["robot"]
obs, _ = env.reset()
obs = obs["policy"][0].cpu().numpy()
d = robot.data
np_ = lambda t: t[0].cpu().numpy()  # noqa: E731

print("joint_names", robot.joint_names)
print("body_names[0]", robot.body_names[0])
print("effort_limit", dict(zip(robot.joint_names, np_(d.joint_effort_limits).round(1))))
print("armature", np_(d.joint_armature).round(3), "friction", np_(d.joint_friction_coeff).round(3))
print("obs after reset", obs.round(3))

assert robot.joint_names == s.ISAAC_JOINTS, robot.joint_names
assert robot.body_names[0] == "pelvis", robot.body_names[0]
assert np.allclose(np_(d.default_joint_pos), s.DEFAULT, atol=1e-4), np_(d.default_joint_pos)
assert np.allclose(np_(d.joint_stiffness), s.KP) and np.allclose(np_(d.joint_damping), s.KD), (
    np_(d.joint_stiffness), np_(d.joint_damping))
assert abs(u.physics_dt - s.SIM_DT) < 1e-9 and u.cfg.decimation == s.DECIMATION, (u.physics_dt, u.cfg.decimation)
assert u.action_manager.get_term("joint_pos").cfg.scale == s.ACTION_SCALE
assert obs.shape == (69,), obs.shape  # 3+3+3+3+19+19+19
assert np.allclose(obs[6:9], [0, 0, -1], atol=0.05), obs[6:9]  # projected gravity, upright
assert np.allclose(obs[12:31], 0, atol=1e-3), obs[12:31]  # joint_pos_rel at default pose
print("CHECK OK")
env.close()
app.close()
