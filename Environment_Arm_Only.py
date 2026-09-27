"""
Environment_Arm_Only.py
=======================
Minimal UR5e (+ Robotiq 2F-85) workcell for the CVAE config-generation work.

Scene layout
------------
    floor      plane at z = 0
    pedestal   cylinder, radius 0.12, spans z = 0.00 -> 0.80
    robot base (0, 0, 0.80)

WHY THE PEDESTAL STAYS
----------------------
The table went away, but the robot base height did not. Keeping base_z = 0.80
means every kinematic constant measured by the old scene (and therefore your
analytic IK, your existing dataset, and your trained checkpoints) still lines
up. A pedestal that now runs floor -> base is also honest collision geometry:
without it the arm could sweep through the space its own mount occupies and the
static filter would happily pass configurations that a real cell would not.

Set BASE_Z = 0.0 and PEDESTAL = False if you would rather bolt the arm straight
to the floor, but be aware that changes the workspace and invalidates the old
data.

WHY THE GRIPPER STAYS
---------------------
The pinch site is the TCP your IK solves to. Drop the gripper and there is no
pinch site, the flange->TCP offset changes, and the collision geometry that
actually matters at the EE (the fingers) disappears. Set WITH_GRIPPER = False
if you genuinely want a bare arm.

Run directly to get a build report plus a viewer. Import build_model() from
anywhere else.
"""

from pathlib import Path

import mujoco
import numpy as np

# ----------------------------------------------------------------------------
# Paths -- edit MENAGERIE if your checkout lives elsewhere
# ----------------------------------------------------------------------------
MENAGERIE = Path(r"D:\University\Thesis\Code\mujoco_menagerie")
ARM_XML = MENAGERIE / "universal_robots_ur5e" / "ur5e.xml"
GRIPPER_XML = MENAGERIE / "robotiq_2f85" / "2f85.xml"

# ----------------------------------------------------------------------------
# Scene constants
# ----------------------------------------------------------------------------
BASE_Z = 0.80  # world height of the robot base
PEDESTAL = True  # draw a mount cylinder from the floor up to BASE_Z
PEDESTAL_RADIUS = 0.12
FLOOR = True  # a plane the arm is not allowed to hit
WITH_GRIPPER = True

ARM_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

# UR5e "home" pose from ur5e.xml's keyframe (arm joints only).
HOME_ARM_Q = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])

ARM_BODY_CHAIN = [
    "base",
    "shoulder_link",
    "upper_arm_link",
    "forearm_link",
    "wrist_1_link",
    "wrist_2_link",
    "wrist_3_link",
]


# ============================================================================
# Build
# ============================================================================
def build_model():
    """Merge arm (+ gripper), add floor and pedestal, compile in memory.

    Returns (model, data, meta). meta carries the arm qpos/dof index arrays so
    callers never have to assume qpos[:6] is the arm.
    """
    if not ARM_XML.exists():
        raise FileNotFoundError(f"Arm XML not found: {ARM_XML}")

    spec = mujoco.MjSpec.from_file(str(ARM_XML))

    if WITH_GRIPPER:
        if not GRIPPER_XML.exists():
            raise FileNotFoundError(f"Gripper XML not found: {GRIPPER_XML}")
        gripper_spec = mujoco.MjSpec.from_file(str(GRIPPER_XML))

        try:
            attach_site = spec.site("attachment_site")
        except (KeyError, ValueError) as exc:
            names = [s.name for s in spec.sites]
            raise RuntimeError(
                f"Could not find 'attachment_site' in the arm spec. Sites: {names}"
            ) from exc

        # The gripper XML declares cone="elliptic" impratio="10"; the arm does
        # not. On merge the parent's options win and MuJoCo warns. Adopt the
        # gripper's values explicitly rather than silently inheriting the arm's.
        spec.option.cone = gripper_spec.option.cone
        spec.option.impratio = gripper_spec.option.impratio

        spec.attach(gripper_spec, site=attach_site)

    # --- Lift the robot onto the base height ---------------------------------
    # The arm's root body carries its own orientation (quat 0 0 0 -1, i.e. 180
    # deg about z). Move it without touching that rotation; downstream code
    # measures the base frame rather than assuming identity.
    arm_root = spec.worldbody.bodies[0]
    original_pos = np.array(arm_root.pos, dtype=float).copy()
    original_quat = np.array(arm_root.quat, dtype=float).copy()
    arm_root.pos = [0.0, 0.0, BASE_Z]

    world = spec.worldbody

    # --- Lighting (cosmetic; the headlight works regardless) -----------------
    try:
        world.add_light(
            pos=[0, 0, 3.0],
            dir=[0, 0, -1],
            type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        )
    except (TypeError, AttributeError):
        try:
            world.add_light(pos=[0, 0, 3.0], dir=[0, 0, -1], directional=True)
        except TypeError:
            world.add_light(pos=[0, 0, 3.0], dir=[0, 0, -1])

    if FLOOR:
        world.add_geom(
            name="floor",
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=[3.0, 3.0, 0.05],
            pos=[0, 0, 0],
            rgba=[0.35, 0.36, 0.38, 1.0],
        )

    if PEDESTAL and BASE_Z > 0.0:
        half = BASE_Z / 2.0
        world.add_geom(
            name="pedestal",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[PEDESTAL_RADIUS, half, 0.0],
            pos=[0, 0, half],
            rgba=[0.25, 0.25, 0.28, 1.0],
        )

    model = spec.compile()
    data = mujoco.MjData(model)

    qadr, dadr = arm_indices(model)
    meta = {
        "arm_root_name": arm_root.name,
        "arm_root_original_pos": original_pos,
        "arm_root_quat": original_quat,
        "arm_qpos_adr": qadr,
        "arm_dof_adr": dadr,
        "base_z": BASE_Z,
        "with_gripper": WITH_GRIPPER,
    }
    return model, data, meta


def arm_indices(model):
    """qpos and dof addresses of the six arm joints, in canonical order."""
    qadr, dadr = [], []
    for n in ARM_JOINT_NAMES:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
        if jid < 0:
            raise RuntimeError(f"Arm joint '{n}' not found after merge.")
        qadr.append(model.jnt_qposadr[jid])
        dadr.append(model.jnt_dofadr[jid])
    return np.array(qadr), np.array(dadr)


def reset_scene(model, data, arm_q=None):
    """Reset to a known state and set the arm.

    Uses mj_resetData (which restores model.qpos0), not mj_resetDataKeyframe.
    The 'home' keyframe in ur5e.xml declares only 6 qpos values while the merged
    model has more, and MuJoCo pads the remainder with zeros.
    """
    mujoco.mj_resetData(model, data)
    set_arm(model, data, HOME_ARM_Q if arm_q is None else arm_q)


def set_arm(model, data, q, forward=True):
    """Write a 6-vector of joint angles into the arm's qpos slots."""
    q = np.asarray(q, dtype=float).reshape(6)
    data.qpos[arm_indices(model)[0]] = q
    if forward:
        mujoco.mj_forward(model, data)


# ============================================================================
# Validity helpers -- the two things the static filter needs
# ============================================================================
def within_limits(model, q, margin=0.0):
    """True if every arm joint is inside its declared range (plus margin)."""
    q = np.asarray(q, dtype=float).reshape(6)
    for i, name in enumerate(ARM_JOINT_NAMES):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if not model.jnt_limited[jid]:
            continue
        lo, hi = model.jnt_range[jid]
        if q[i] < lo + margin or q[i] > hi - margin:
            return False
    return True


def geom_label(model, gid):
    """Readable name for a geom. Most UR5e mesh geoms are unnamed in the XML,
    so fall back to the parent body, which is what you actually want to see in
    a collision report."""
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
    if name:
        return name
    body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[gid])
    return f"{body or '?'}#{gid}"


def baseline_contacts(model, data, arm_q=None):
    """Geom pairs that are in contact in a known-good pose.

    Reject on NEW pairs, not on ncon != 0 -- the gripper pads touch each other
    at rest and that is not a collision.
    """
    reset_scene(model, data, arm_q)
    mujoco.mj_forward(model, data)
    return {
        tuple(sorted((int(data.contact[i].geom1), int(data.contact[i].geom2))))
        for i in range(data.ncon)
    }


def contact_pairs(model, data, q):
    """Geom pairs in contact at configuration q (no dynamics, pose only)."""
    set_arm(model, data, q)
    return {
        tuple(sorted((int(data.contact[i].geom1), int(data.contact[i].geom2))))
        for i in range(data.ncon)
    }


def is_collision_free(model, data, q, baseline):
    """True if q introduces no contact pair beyond the baseline set."""
    return not (contact_pairs(model, data, q) - baseline)


# ============================================================================
# Report
# ============================================================================
def rule(title=""):
    print("\n" + "=" * 78)
    if title:
        print(title)
        print("=" * 78)


def report(model, data, meta):
    rule("BUILD SUMMARY")
    print(f"arm root body            : '{meta['arm_root_name']}'")
    print(f"  original pos           : {meta['arm_root_original_pos']}")
    print(f"  original quat (w x y z): {meta['arm_root_quat']}   <-- NOT identity")
    print(f"  moved to               : [0, 0, {BASE_Z}]")
    print(f"gripper attached         : {WITH_GRIPPER}")
    print(f"floor / pedestal         : {FLOOR} / {PEDESTAL}")
    print(f"nq={model.nq}  nv={model.nv}  nu={model.nu}  ngeom={model.ngeom}")

    rule("JOINT / QPOS LAYOUT")
    print(f"{'id':>3}  {'name':<28} {'type':<9} {'qposadr':>7} {'dofadr':>6}  range")
    print("-" * 78)
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or "<unnamed>"
        jtype = mujoco.mjtJoint(model.jnt_type[j]).name.replace("mjJNT_", "").lower()
        lo, hi = model.jnt_range[j]
        rng_s = f"[{lo: .4f}, {hi: .4f}]" if model.jnt_limited[j] else "unlimited"
        print(
            f"{j:>3}  {name:<28} {jtype:<9} {model.jnt_qposadr[j]:>7} "
            f"{model.jnt_dofadr[j]:>6}  {rng_s}"
        )

    qadr = meta["arm_qpos_adr"]
    print(f"\narm qpos addresses : {qadr.tolist()}")
    print(f"arm dof addresses  : {meta['arm_dof_adr'].tolist()}")
    if np.array_equal(qadr, np.arange(6)):
        print("  -> qpos[:6] IS the six arm joints. `data.qpos[:6] = q` is valid.")
    else:
        print("  -> qpos[:6] is NOT the arm. Use meta['arm_qpos_adr'].")

    rule("MEASURED KINEMATIC PARAMETERS (vs published UR5e DH)")

    def bpos(name):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        return model.body_pos[bid].copy()

    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    measured = dict(
        d1=bpos("shoulder_link")[2],
        a2=bpos("forearm_link")[2],
        a3=bpos("wrist_1_link")[2],
        d4=bpos("upper_arm_link")[1]
        + bpos("forearm_link")[1]
        + bpos("wrist_2_link")[1],
        d5=bpos("wrist_3_link")[2],
        d6=model.site_pos[sid][1] if sid >= 0 else float("nan"),
    )
    published = dict(d1=0.1625, a2=0.4250, a3=0.3922, d4=0.1333, d5=0.0997, d6=0.0996)

    print(f"{'param':<8} {'measured':>12} {'published':>12} {'delta (mm)':>12}")
    print("-" * 78)
    for k in ["d1", "a2", "a3", "d4", "d5", "d6"]:
        print(
            f"{k:<8} {measured[k]:>12.6f} {published[k]:>12.6f} "
            f"{(measured[k] - published[k]) * 1000.0:>12.3f}"
        )
    print("\n  Use the MEASURED column in the analytic IK, not the published DH.")

    rule("TCP OFFSET AND WORKSPACE")
    reset_scene(model, data, np.zeros(6))
    site_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i)
        for i in range(model.nsite)
    ]
    pinch = [n for n in site_names if n and n.endswith("pinch")]
    if pinch:
        pid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, pinch[0])
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
        T_a = np.eye(4)
        T_a[:3, :3] = data.site_xmat[aid].reshape(3, 3)
        T_a[:3, 3] = data.site_xpos[aid]
        T_p = np.eye(4)
        T_p[:3, :3] = data.site_xmat[pid].reshape(3, 3)
        T_p[:3, 3] = data.site_xpos[pid]
        T = np.linalg.inv(T_a) @ T_p
        print(f"pinch site               : '{pinch[0]}'")
        print("T_attachment_site -> pinch:")
        for row in T:
            print("    [" + "  ".join(f"{v: .6f}" for v in row) + "]")
        tcp_len = float(np.linalg.norm(T[:3, 3]))
        print(f"\ngripper length (flange -> pinch) = {tcp_len:.6f} m")
    else:
        tcp_len = 0.0
        print("no pinch site (bare arm); TCP = attachment_site")

    shoulder_z = BASE_Z + measured["d1"]
    reach = measured["a2"] + measured["a3"]
    print(f"shoulder height (world z)  : {shoulder_z:.4f}")
    print(f"planar reach |a2| + |a3|   : {reach:.4f}")
    print(f"floor clearance below base : {BASE_Z:.4f} m")

    rule("BASELINE CONTACTS AT THE HOME POSE")
    base = baseline_contacts(model, data)
    print(f"data.ncon = {data.ncon}")
    for g1, g2 in sorted(base):
        print(f"  {geom_label(model, g1):<34} <-> {geom_label(model, g2)}")
    print(
        "\n  These pairs are 'normal' (gripper pads touching) and must NOT count"
        "\n  as collisions. The filter rejects on NEW pairs, not on ncon != 0."
    )
    return base


# ============================================================================
# Main
# ============================================================================
def main():
    model, data, meta = build_model()
    baseline = report(model, data, meta)

    # Smoke test: home pose should be valid, a folded-in pose should not.
    rule("VALIDITY SMOKE TEST")
    for label, q in [
        ("home", HOME_ARM_Q),
        ("zeros", np.zeros(6)),
        ("folded (elbow slammed shut)", np.array([0.0, -0.2, 2.9, 0.0, 0.0, 0.0])),
    ]:
        lim = within_limits(model, q)
        free = is_collision_free(model, data, q, baseline)
        new = contact_pairs(model, data, q) - baseline
        names = ", ".join(
            f"{geom_label(model, a)}<->{geom_label(model, b)}" for a, b in sorted(new)
        )
        print(
            f"  {label:<30} limits={'ok' if lim else 'VIOLATED':<9} "
            f"collision={'free' if free else 'HIT'}   {names}"
        )

    rule("LAUNCHING VIEWER  (close the window to exit)")
    reset_scene(model, data)
    import mujoco.viewer

    with mujoco.viewer.launch_passive(
        model, data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        while viewer.is_running():
            mujoco.mj_forward(model, data)  # pose only, no dynamics
            viewer.sync()


if __name__ == "__main__":
    main()
