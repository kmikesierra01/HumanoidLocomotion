"""Stair evaluation course: flat approach -> 3 steps up -> flat platform. One box list, used by both engines
(MuJoCo: box geoms, because mesh collisions are convexified there; Isaac: one trimesh, because its height
scanner only ray-casts the ground mesh). Robot starts at the origin facing +x.

Usage: python course.py   -> renders results/course_*.png and writes results/course.obj
"""
import numpy as np

RISE, TREAD, N_STEPS = 0.15, 0.30, 3  # m; typical building stairs ~0.15-0.18 rise, 0.28-0.30 tread
APPROACH = 1.5  # flat distance from start to first step edge
PLATFORM = 5.0  # top platform length
WIDTH = 3.0
TOP_Z = RISE * N_STEPS


def boxes():
    """[(center xyz, half-size xyz)] of the solid steps and platform."""
    out = []
    for i in range(N_STEPS):
        x0, h = APPROACH + i * TREAD, (i + 1) * RISE
        out.append(((x0 + TREAD / 2, 0, h / 2), (TREAD / 2, WIDTH / 2, h / 2)))
    x0 = APPROACH + N_STEPS * TREAD
    out.append(((x0 + PLATFORM / 2, 0, TOP_Z / 2), (PLATFORM / 2, WIDTH / 2, TOP_Z / 2)))
    return out


def add_to_spec(spec):
    """Add the course as static box geoms to a MuJoCo MjSpec world body."""
    import mujoco

    for k, (c, hs) in enumerate(boxes()):
        g = spec.worldbody.add_geom()
        g.name, g.type, g.pos, g.size = f"course_{k}", mujoco.mjtGeom.mjGEOM_BOX, c, hs
        g.rgba = (0.55, 0.55, 0.6, 1) if k < N_STEPS else (0.45, 0.5, 0.55, 1)


def trimesh_course(ground_size=(20.0, 10.0)):
    """Course + ground slab (centered on the start) as one trimesh for Isaac Lab's terrain importer."""
    import trimesh

    gx, gy = ground_size
    parts = [trimesh.creation.box((gx, gy, 0.02), trimesh.transformations.translation_matrix((0, 0, -0.01)))]
    parts += [trimesh.creation.box(np.multiply(hs, 2), trimesh.transformations.translation_matrix(c)) for c, hs in boxes()]
    return trimesh.util.concatenate(parts)


TILE = (20.0, 10.0)


def isaac_course_terrain(difficulty, cfg):
    """SubTerrainBaseCfg.function for a 1x1 TerrainGenerator of size TILE. The generator shifts the tile by -TILE/2,
    so returning the course shifted by +TILE/2 puts world frame = course frame (start at the origin)."""
    m = trimesh_course(TILE)
    m.apply_translation((TILE[0] / 2, TILE[1] / 2, 0))
    return [m], np.array([TILE[0] / 2, TILE[1] / 2, 0.0])


# Stair evaluation protocol, identical in stairs_isaac.py and sim2sim.py --stairs
GOAL_X = APPROACH + N_STEPS * TREAD + 2.0  # success = pelvis 2 m onto the top platform
STAIR_SECONDS = 20.0  # not there by then (and not fallen) = stuck


def ground_z(x, y=0.0):
    """Course surface height under (x, y)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    k = np.clip(np.floor((x - APPROACH) / TREAD) + 1, 0, N_STEPS)
    on = (x >= APPROACH) & (x < APPROACH + N_STEPS * TREAD + PLATFORM) & (np.abs(y) <= WIDTH / 2)
    return np.where(on, k * RISE, 0.0)


def stair_command(i):
    """Episode i's forward speed command (m/s), inside the H1 training range (0, 1)."""
    return float(np.random.default_rng(i).uniform(0.4, 1.0))


def heading_wz(yaw, target=0.0):
    """Yaw-rate command that holds heading `target`, as Isaac Lab's UniformVelocityCommand heading control
    (stiffness 0.5, clipped to the ang_vel_z range +-1)."""
    err = (target - yaw + np.pi) % (2 * np.pi) - np.pi
    return float(np.clip(0.5 * err, -1.0, 1.0))


def stair_result(vx, outcome, t_s, max_x):
    return {"vx": vx, "outcome": outcome, "time_s": t_s, "max_x": float(max_x)}


if __name__ == "__main__":
    import mujoco
    from PIL import Image

    from sim2sim import XML
    from h1_isaac import DEFAULT, INIT_Z, ISAAC_JOINTS

    spec = mujoco.MjSpec.from_file(XML)
    add_to_spec(spec)
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 900, 600
    m = spec.compile()
    d = mujoco.MjData(m)
    d.qpos[2] = INIT_Z
    for n, q in zip(ISAAC_JOINTS, DEFAULT):
        d.qpos[m.joint(n).qposadr[0]] = q
    mujoco.mj_forward(m, d)

    # step edges / top surface check: ray straight down at each tread center hits the expected height
    for i in range(N_STEPS):
        x = APPROACH + (i + 0.5) * TREAD
        gid = np.zeros(1, np.int32)
        dist = mujoco.mj_ray(m, d, np.array([x, 0, 2.0]), np.array([0, 0, -1.0]), None, 1, -1, gid)
        assert abs((2.0 - dist) - (i + 1) * RISE) < 1e-6, (i, 2.0 - dist)

    r = mujoco.Renderer(m, 600, 900)
    cam = mujoco.MjvCamera()
    for name, (az, el, dist, look) in {"side": (90, -10, 4.5, (1.8, 0, 0.6)), "persp": (135, -25, 5.0, (1.8, 0, 0.4))}.items():
        cam.azimuth, cam.elevation, cam.distance, cam.lookat[:] = az, el, dist, look
        r.update_scene(d, cam)
        Image.fromarray(r.render()).save(f"results/course_{name}.png")
    tm = trimesh_course()
    tm.export("results/course.obj")
    print("course OK: steps", [round((i + 1) * RISE, 2) for i in range(N_STEPS)], "top", TOP_Z,
          "| trimesh bounds", tm.bounds.round(2).tolist(), "faces", len(tm.faces))
