"""Isaac Lab H1 flat-env facts that sim2sim.py must reproduce. numpy only: mujoco can't be imported inside the
Isaac Sim (Kit) process (WinError 1114 DLL init failed), and check_isaac_cfg.py imports this there."""
import numpy as np

# Isaac Lab articulation order (PhysX breadth-first) = order of joint_pos/joint_vel/actions in the policy.
ISAAC_JOINTS = [
    "left_hip_yaw", "right_hip_yaw", "torso", "left_hip_roll", "right_hip_roll",
    "left_shoulder_pitch", "right_shoulder_pitch", "left_hip_pitch", "right_hip_pitch",
    "left_shoulder_roll", "right_shoulder_roll", "left_knee", "right_knee",
    "left_shoulder_yaw", "right_shoulder_yaw", "left_ankle", "right_ankle", "left_elbow", "right_elbow",
]
# joint suffix: (default pos rad, kp, kd), from isaaclab_assets/robots/unitree.py H1_CFG
CFG = {
    "hip_yaw": (0.0, 150, 5), "hip_roll": (0.0, 150, 5), "hip_pitch": (-0.28, 200, 5), "knee": (0.79, 200, 5),
    "ankle": (-0.52, 20, 4), "torso": (0.0, 200, 5), "shoulder_pitch": (0.28, 40, 10),
    "shoulder_roll": (0.0, 40, 10), "shoulder_yaw": (0.0, 40, 10), "elbow": (0.52, 40, 10),
}
DEFAULT, KP, KD = np.array([CFG[n.removeprefix("left_").removeprefix("right_")] for n in ISAAC_JOINTS]).T
ACTION_SCALE = 0.5  # velocity_env_cfg.py ActionsCfg, use_default_offset=True
SIM_DT, DECIMATION = 0.005, 4  # 200 Hz physics, 50 Hz policy
INIT_Z = 1.05  # H1_CFG init_state.pos
# Uniform obs noise half-widths used in training (velocity_env_cfg.py PolicyCfg Unoise), same 69-dim layout.
# The only randomization H1 training had: pushes, mass, CoM and friction randomization are all off for H1.
OBS_NOISE = np.concatenate([[0.1] * 3, [0.2] * 3, [0.05] * 3, [0.0] * 3, [0.01] * 19, [1.5] * 19, [0.0] * 19])

# Rough-terrain policy: obs = the 69 above + height_scan (velocity_env_cfg.py height_scanner, H1 on torso_link).
# Rays start 20 m above torso_link, follow its yaw only (ray_alignment="yaw"), point straight down, hit only the
# ground mesh. Grid: GridPatternCfg(resolution=0.1, size=[1.6, 1.0]) with torch meshgrid "xy" -> x varies fastest.
# value = torso_z - hit_z - SCAN_OFFSET, clipped to +-1.
SCAN_GRID = np.stack(np.meshgrid(np.arange(-0.8, 0.8 + 1e-9, 0.1), np.arange(-0.5, 0.5 + 1e-9, 0.1), indexing="xy"),
                     -1).reshape(-1, 2)  # (187, 2), torso-yaw frame
SCAN_OFFSET, SCAN_HEIGHT = 0.5, 20.0

# Evaluation protocol, identical in eval_isaac.py and sim2sim.py --headless
EVAL_SECONDS = 20.0
WARMUP_STEPS = 50  # first 1 s (drop + acceleration) not counted in tracking error
FALL_Z = 0.6  # pelvis height below this = fallen (standing ~1.0); Isaac also counts base_contact termination


def eval_command(i):
    """Episode i's fixed (vx, vy, wz), drawn from the H1 training ranges (rough_env_cfg.py Commands)."""
    r = np.random.default_rng(i)
    return np.array([r.uniform(0.0, 1.0), 0.0, r.uniform(-1.0, 1.0)], np.float32)


def episode_result(cmd, fall_step, v_xy, wz):
    """v_xy (T,2) yaw-frame linear velocity and wz (T,) world yaw rate, recorded while alive."""
    v_xy, wz = np.asarray(v_xy)[WARMUP_STEPS:], np.asarray(wz)[WARMUP_STEPS:]
    ok = len(wz) > 0
    return {
        "cmd": [float(c) for c in cmd],
        "fell": fall_step is not None,
        "fall_time_s": None if fall_step is None else fall_step * SIM_DT * DECIMATION,
        "err_xy": float(np.linalg.norm(v_xy - cmd[:2], axis=1).mean()) if ok else None,
        "err_wz": float(np.abs(wz - cmd[2]).mean()) if ok else None,
    }
