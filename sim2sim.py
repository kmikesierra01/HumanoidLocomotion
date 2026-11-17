"""Run an Isaac Lab H1 policy (policy.onnx) in MuJoCo (Menagerie unitree_h1) with ONNX Runtime.
Flat policy (69 obs) or rough/stairs policy (69 + 187 height scan, detected from the ONNX input size).

Usage:
  python sim2sim.py <policy.onnx> [--vx 1.0] [--wz 0.0] [--course]               viewer (--course: stairs, heading hold)
  python sim2sim.py <policy.onnx> --headless --episodes 100 --out mujoco.json     same episodes as eval_isaac.py
  python sim2sim.py <policy.onnx> --stairs --episodes 100 --out mujoco_stairs.json   same episodes as stairs_isaac.py
  python sim2sim.py <policy.onnx> --check-scan isaac_stairs.json                  height scan vs Isaac's recorded scans
  python sim2sim.py <policy.onnx> --course --vx 0.7 --record out.mp4 [--seconds 12]  mp4 instead of the viewer
"""
import argparse
import json
import os
import time
from collections import deque

import mujoco
import mujoco.viewer
import numpy as np
import onnxruntime as ort

import course
from h1_isaac import (ACTION_SCALE, DECIMATION, DEFAULT, EVAL_SECONDS, FALL_Z, INIT_Z, ISAAC_JOINTS, KD, KP,  # noqa
                      OBS_NOISE, SCAN_GRID, SCAN_HEIGHT, SCAN_OFFSET, SIM_DT, episode_result, eval_command)

XML = os.path.join(os.environ.get("MENAGERIE", r"C:\Storage\IsaacSim\mujoco_menagerie"), "unitree_h1", "scene.xml")
PUSH_START_S, PUSH_DURATION_S = 5.0, 0.2  # one horizontal shove on torso_link per episode (robustness.py)
GROUND_ONLY = np.array([1, 0, 0, 0, 0, 0], np.uint8)  # floor + course are geom group 0; H1 geoms are groups 2/3
DOWN = np.array([0.0, 0.0, -1.0])


def yaw_of(R):
    return np.arctan2(R[1, 0], R[0, 0])


class H1:
    def __init__(self, onnx_path, friction=1.0, payload_kg=0.0, delay_ms=0.0, noise=0.0, seed=0, course_on=False):
        """Perturbations for robustness.py (defaults = nominal model):
        friction   scale on every geom's sliding friction (MuJoCo uses the max of a contact pair, so all are scaled)
        payload_kg extra mass on torso_link (at its CoM)
        delay_ms   PD target reaches the motors this late (multiple of SIM_DT)
        noise      obs noise as a multiple of the training noise OBS_NOISE (first 69 obs only)
        course_on  add the stair course (course.py)"""
        spec = mujoco.MjSpec.from_file(XML)
        if course_on:
            course.add_to_spec(spec)
        spec.visual.global_.offwidth, spec.visual.global_.offheight = 1280, 720  # --record (no effect on physics)
        self.m = spec.compile()
        self.course_on = course_on
        self.m.opt.timestep = SIM_DT
        self.m.geom_friction[:, 0] *= friction
        self.torso = self.m.body("torso_link").id
        self.m.body_mass[self.torso] += payload_kg
        self.delay = round(delay_ms / 1000 / SIM_DT)
        self.noise = noise * OBS_NOISE
        self.rng = np.random.default_rng(seed)
        self.d = mujoco.MjData(self.m)
        self.qadr = np.array([self.m.joint(n).qposadr[0] for n in ISAAC_JOINTS])
        self.vadr = np.array([self.m.joint(n).dofadr[0] for n in ISAAC_JOINTS])
        self.act = np.array([self.m.actuator(n).id for n in ISAAC_JOINTS])
        self.pelvis = self.m.body("pelvis").id
        self.sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        n_obs = self.sess.get_inputs()[0].shape[1]
        assert n_obs in (69, 69 + len(SCAN_GRID)), n_obs
        self.scan = n_obs > 69
        self.reset()

    def reset(self):
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[2] = INIT_Z
        self.d.qpos[self.qadr] = DEFAULT
        mujoco.mj_forward(self.m, self.d)
        self.action = np.zeros(len(ISAAC_JOINTS), np.float32)
        self.targets = deque([DEFAULT] * (self.delay + 1), maxlen=self.delay + 1)  # [0] = target sent delay steps ago

    def base_vel(self):
        """Pelvis COM linear velocity and angular velocity, both world axes (Isaac root_com_*_vel_w)."""
        v = np.zeros(6)  # [ang, lin]
        mujoco.mj_objectVelocity(self.m, self.d, mujoco.mjtObj.mjOBJ_BODY, self.pelvis, v, 0)
        return v[3:], v[:3]

    def height_scan(self, pos, yaw):
        """Isaac height_scan obs for a sensor (torso_link) at pos with heading yaw: 187 rays straight down."""
        c, s = np.cos(yaw), np.sin(yaw)
        xy = pos[:2] + SCAN_GRID @ np.array([[c, s], [-s, c]])
        z0 = pos[2] + SCAN_HEIGHT
        gid = np.zeros(1, np.int32)
        hit = np.array([z0 - mujoco.mj_ray(self.m, self.d, np.array([x, y, z0]), DOWN, GROUND_ONLY, 1, -1, gid)
                        for x, y in xy])
        return np.clip(pos[2] - hit - SCAN_OFFSET, -1.0, 1.0)

    def obs(self, cmd):
        """Same layout as velocity_env_cfg.py PolicyCfg (no scaling): 69 proprioceptive [+ 187 height scan]."""
        R = self.d.xmat[self.pelvis].reshape(3, 3)  # body -> world
        lin, ang = self.base_vel()
        noise = self.rng.uniform(-self.noise, self.noise)
        o = noise + np.concatenate([
            R.T @ lin,  # base_lin_vel
            R.T @ ang,  # base_ang_vel
            R.T @ [0, 0, -1],  # projected_gravity
            cmd,  # velocity_commands (vx, vy, wz)
            self.d.qpos[self.qadr] - DEFAULT,  # joint_pos_rel
            self.d.qvel[self.vadr],  # joint_vel_rel (default vel = 0)
            self.action,  # last_action (raw policy output)
        ])
        if self.scan:
            o = np.concatenate([o, self.height_scan(self.d.xpos[self.torso], yaw_of(self.d.xmat[self.torso].reshape(3, 3)))])
        return o.astype(np.float32)

    def step(self, cmd):
        self.action = self.sess.run(None, {"obs": self.obs(cmd)[None]})[0][0]
        target = DEFAULT + ACTION_SCALE * self.action
        for _ in range(DECIMATION):
            self.targets.append(target)
            tau = KP * (self.targets[0] - self.d.qpos[self.qadr]) - KD * self.d.qvel[self.vadr]
            self.d.ctrl[self.act] = tau  # clipped by Menagerie ctrlrange (real H1 limits, tighter than Isaac's)
            mujoco.mj_step(self.m, self.d)

    def yaw(self):
        return yaw_of(self.d.xmat[self.pelvis].reshape(3, 3))

    def fallen(self):
        x, y, z = self.d.qpos[:3]
        return z - (course.ground_z(x, y) if self.course_on else 0.0) < FALL_Z

    def tracked_vel(self):
        """Yaw-frame (vx, vy) and world yaw rate, as in track_lin_vel_xy_yaw_frame_exp / track_ang_vel_z_world_exp."""
        w, x, y, z = self.d.qpos[3:7]  # (yaw from qpos as when results/mujoco.json + robustness.json were made)
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        c, s = np.cos(yaw), np.sin(yaw)
        lin, ang = self.base_vel()
        return np.array([c * lin[0] + s * lin[1], -s * lin[0] + c * lin[1]]), ang[2]


def view(h1, cmd):
    with mujoco.viewer.launch_passive(h1.m, h1.d) as v:
        while v.is_running():
            t0 = time.time()
            if h1.course_on:
                cmd[2] = course.heading_wz(h1.yaw())
            h1.step(cmd)
            if h1.fallen() or (h1.course_on and h1.d.qpos[0] >= course.GOAL_X + 2):
                print("fell, reset" if h1.fallen() else "reached the end, reset")
                h1.reset()
            v.cam.lookat[:] = h1.d.qpos[:3]
            v.sync()
            time.sleep(max(0.0, SIM_DT * DECIMATION - (time.time() - t0)))


def record(h1, cmd, path, seconds):
    """Offscreen render to mp4 at the policy rate (50 fps, real time), camera following the pelvis. No reset on fall."""
    import imageio

    r = mujoco.Renderer(h1.m, 720, 1280)
    cam = mujoco.MjvCamera()
    cam.azimuth, cam.elevation, cam.distance = 125, -15, 4.0
    with imageio.get_writer(path, fps=round(1 / (SIM_DT * DECIMATION)), macro_block_size=1) as w:
        for _ in range(int(seconds / (SIM_DT * DECIMATION))):
            if h1.course_on:
                cmd[2] = course.heading_wz(h1.yaw())
            h1.step(cmd)
            cam.lookat[:] = h1.d.qpos[0], h1.d.qpos[1], 0.8
            r.update_scene(h1.d, cam)
            w.append_data(r.render())


def evaluate(h1, episodes, push_n=0.0):
    """push_n: force of one PUSH_DURATION_S shove at PUSH_START_S, direction drawn per episode (seed 10000+i)."""
    dt = SIM_DT * DECIMATION
    push_steps = range(round(PUSH_START_S / dt), round((PUSH_START_S + PUSH_DURATION_S) / dt))
    results = []
    for i in range(episodes):
        cmd = eval_command(i)
        a = np.random.default_rng(10000 + i).uniform(-np.pi, np.pi)
        h1.reset()
        fall_step, v_xy, wz = None, [], []
        for t in range(int(EVAL_SECONDS / dt)):
            h1.d.xfrc_applied[h1.torso, :2] = push_n * np.array([np.cos(a), np.sin(a)]) if t in push_steps else 0.0
            h1.step(cmd)
            if h1.fallen():
                fall_step = t
                break
            vxy, w = h1.tracked_vel()
            v_xy.append(vxy)
            wz.append(w)
        results.append(episode_result(cmd, fall_step, v_xy, wz))
    return results


def evaluate_stairs(h1, episodes):
    """Course run per episode: forward speed course.stair_command(i), heading held at 0."""
    dt = SIM_DT * DECIMATION
    results = []
    for i in range(episodes):
        vx = course.stair_command(i)
        h1.reset()
        outcome, max_x = "stuck", 0.0
        for t in range(int(course.STAIR_SECONDS / dt)):
            h1.step(np.array([vx, 0.0, course.heading_wz(h1.yaw())], np.float32))
            if h1.fallen():
                outcome = "fall"
                break
            if h1.d.qpos[0] >= course.GOAL_X:
                outcome = "success"
                break
            max_x = max(max_x, h1.d.qpos[0])
        results.append(course.stair_result(vx, outcome, (t + 1) * dt, max_x))
    return results


def check_scan(h1, log):
    """Isaac's recorded (torso pos, quat wxyz, height_scan obs) vs this MuJoCo scan at the same pose."""
    worst = 0.0
    for r in log:
        w, x, y, z = r["quat"]
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        diff = np.abs(h1.height_scan(np.array(r["pos"]), yaw) - np.array(r["scan"]))
        worst = max(worst, diff.max())
        print(f"t={r['t']:4d} torso x={r['pos'][0]:.2f}: max |diff| {diff.max():.4f}, rays off by >1 cm: {(diff > 0.01).sum()}")
    return worst


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("onnx")
    p.add_argument("--vx", type=float, default=1.0)
    p.add_argument("--wz", type=float, default=0.0)
    p.add_argument("--course", action="store_true", help="viewer on the stair course")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--stairs", action="store_true", help="headless stair-course evaluation")
    p.add_argument("--check-scan")
    p.add_argument("--record", help="write an mp4 instead of opening the viewer (with --vx/--wz/--course)")
    p.add_argument("--seconds", type=float, default=12.0, help="--record length")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--out")
    a = p.parse_args()
    h1 = H1(a.onnx, course_on=a.course or a.stairs or bool(a.check_scan))
    if a.record:
        record(h1, np.array([a.vx, 0.0, a.wz], np.float32), a.record, a.seconds)
    elif a.check_scan:
        print(f"worst |diff| {check_scan(h1, json.load(open(a.check_scan))['scan_log']):.4f}")
    elif a.stairs:
        res = evaluate_stairs(h1, a.episodes)
        print("MUJOCO stairs", {k: sum(r["outcome"] == k for r in res) for k in ("success", "fall", "stuck")})
        if a.out:
            json.dump(res, open(a.out, "w"), indent=1)
    elif not a.headless:
        view(h1, np.array([a.vx, 0.0, a.wz], np.float32))
    else:
        res = evaluate(h1, a.episodes)
        print(f"falls {sum(r['fell'] for r in res)}/{len(res)}")
        if a.out:
            json.dump(res, open(a.out, "w"), indent=1)
