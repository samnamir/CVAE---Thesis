"""
Analytic_IK.py
==============
Closed-form FK/IK for the UR5e, calibrated against the compiled MuJoCo model.

Orientation is free. No task convention is baked in here: a target is a full
4x4 world pose of the TCP, and `ik_tcp` returns every candidate config that
reaches it (up to 8: shoulder left/right x wrist flip x elbow up/down).

Frames
------
    T_world_dhbase   model base frame rotated 180 deg about z; the DH chain
                     starts here, so the model's own base quat is respected
                     rather than assumed to be identity.
    T_flange_tcp     constant flange -> pinch-site offset (the gripper).

Link lengths are MEASURED from the compiled model, never taken from published
DH tables -- the mesh geometry disagrees with the datasheet by up to ~0.7 mm,
which is far above the 1e-6 tolerance the solver is held to.

Build targets with `pose(p, R)`; `random_rotation(rng)` gives a uniform R.

Terminology (used consistently across every file in this pipeline)
-----------------------------------------------------------------
    target            an EE pose: the 4x4 world pose of the TCP (the pinch
                      site between the fingers). What the CVAE conditions on.
    candidate config  a 6-vector of joint angles returned by IK for a target,
                      before any collision filtering.
    valid config      a candidate config that survives the collision filter.
    mode              a distinct valid config for one target. Same object as a
                      valid config; the word is used only when COUNTING the
                      diversity of valid configs for a target, which is
                      the quantity this thesis reports.
"""

import mujoco
import numpy as np
from Environment_Arm_Only import ARM_JOINT_NAMES, build_model, reset_scene

TOL_MATH = 1e-9    # isolated analytic round trip
TOL_MUJOCO = 1e-6  # anything passing through the compiled model


# ============================================================================
# Small helpers
# ============================================================================
def wrap_to_pi(x):
    return (np.asarray(x, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.0]])


def pose(p, R=None):
    """4x4 from a position and optional rotation."""
    T = np.eye(4)
    T[:3, 3] = p
    if R is not None:
        T[:3, :3] = R
    return T


def random_rotation(rng):
    """Uniformly distributed rotation matrix (normalised Gaussian quaternion)."""
    q = rng.normal(size=4)
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def dh(theta, d, a, alpha):
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st * ca, st * sa, a * ct],
        [st, ct * ca, -ct * sa, a * st],
        [0.0, sa, ca, d],
        [0.0, 0.0, 0.0, 1.0],
    ])


# ============================================================================
# Kinematics
# ============================================================================
class UR5eKinematics:
    def __init__(self, model, data, verify=True):
        self.model, self.data = model, data

        jids = []
        for name in ARM_JOINT_NAMES:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise RuntimeError(f"joint '{name}' missing from model")
            jids.append(jid)
        self.arm_jids = jids
        self.arm_qadr = model.jnt_qposadr[jids].copy()
        self.jnt_lower = model.jnt_range[jids, 0].copy()
        self.jnt_upper = model.jnt_range[jids, 1].copy()

        self._measure()
        if verify:
            self._verify()

    # ---------------- calibration ----------------
    def _site(self, sid):
        return pose(self.data.site_xpos[sid], self.data.site_xmat[sid].reshape(3, 3))

    def _measure(self):
        m, d = self.model, self.data

        def bpos(name):
            bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid < 0:
                raise RuntimeError(f"body '{name}' missing from model")
            return m.body_pos[bid].copy()

        names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, i) for i in range(m.nsite)]
        pinch = [s for s in names if s and s.endswith("pinch")]
        if not pinch:
            raise RuntimeError(f"no pinch site found; sites = {names}")
        self.pinch_site_name = pinch[0]
        self.pinch_site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, pinch[0])
        self.attach_site_id = mujoco.mj_name2id(
            m, mujoco.mjtObj.mjOBJ_SITE, "attachment_site"
        )
        if self.attach_site_id < 0:
            raise RuntimeError("site 'attachment_site' missing from model")

        self.d1 = float(bpos("shoulder_link")[2])
        self.a2 = float(bpos("forearm_link")[2])
        self.a3 = float(bpos("wrist_1_link")[2])
        self.d4 = float(
            bpos("upper_arm_link")[1] + bpos("forearm_link")[1] + bpos("wrist_2_link")[1]
        )
        self.d5 = float(bpos("wrist_3_link")[2])
        self.d6 = float(m.site_pos[self.attach_site_id][1])

        self._a = [0.0, -self.a2, -self.a3, 0.0, 0.0, 0.0]
        self._d = [self.d1, 0.0, 0.0, self.d4, self.d5, self.d6]
        self._al = [np.pi / 2, 0.0, 0.0, np.pi / 2, -np.pi / 2, 0.0]

        reset_scene(m, d, arm_q=np.zeros(6))
        base_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base")
        T_w_base = pose(d.xpos[base_id], d.xmat[base_id].reshape(3, 3))

        self.T_world_dhbase = T_w_base @ rot_z(np.pi)
        self.T_dhbase_world = np.linalg.inv(self.T_world_dhbase)
        self.T_flange_tcp = (
            np.linalg.inv(self._site(self.attach_site_id)) @ self._site(self.pinch_site_id)
        )
        self.T_tcp_flange = np.linalg.inv(self.T_flange_tcp)

    def _verify(self, n=200, seed=12345):
        rng = np.random.default_rng(seed)
        wf = wt = 0.0
        for _ in range(n):
            q = rng.uniform(-np.pi, np.pi, 6)
            reset_scene(self.model, self.data, arm_q=q)
            wf = max(wf, np.abs(self.fk_flange(q) - self._site(self.attach_site_id)).max())
            wt = max(wt, np.abs(self.fk_tcp(q) - self._site(self.pinch_site_id)).max())
        self.calib_error_flange, self.calib_error_tcp = wf, wt
        if max(wf, wt) > TOL_MUJOCO:
            raise RuntimeError(
                f"Calibration failed (flange {wf:.3e}, tcp {wt:.3e}). The model's "
                "frame convention is not what this solver assumes."
            )

    # ---------------- forward ----------------
    def dh_fk(self, q):
        T = np.eye(4)
        for i in range(6):
            T = T @ dh(q[i], self._d[i], self._a[i], self._al[i])
        return T

    def fk_flange(self, q):
        return self.T_world_dhbase @ self.dh_fk(q)

    def fk_tcp(self, q):
        return self.fk_flange(q) @ self.T_flange_tcp

    # ---------------- inverse ----------------
    def ik_flange(self, T_world_flange):
        """Every candidate config for a world FLANGE pose, wrapped to [-pi, pi)."""
        T = self.T_dhbase_world @ np.asarray(T_world_flange, dtype=float)
        d4, d6, a2, a3 = self.d4, self.d6, self.a2, self.a3
        px, py = T[0, 3], T[1, 3]
        sols = []

        # theta1 -- the joint-5 origin sits at constant offset d4 in frame 1
        P5 = T @ np.array([0.0, 0.0, -d6, 1.0])
        R = np.hypot(P5[0], P5[1])
        if R < abs(d4) - 1e-12:
            return sols  # inside the shoulder cylinder: unreachable
        phi = np.arctan2(P5[1], P5[0])
        asr = np.arcsin(np.clip(d4 / R, -1.0, 1.0))

        for t1 in (phi + asr, phi + np.pi - asr):
            s1, c1 = np.sin(t1), np.cos(t1)
            A1 = dh(t1, self._d[0], self._a[0], self._al[0])

            # theta5
            arg = (px * s1 - py * c1 - d4) / d6
            if abs(arg) > 1.0 + 1e-9:
                continue
            ac5 = np.arccos(np.clip(arg, -1.0, 1.0))

            for t5 in (ac5, -ac5):
                s5 = np.sin(t5)
                # theta6 -- undefined at the wrist singularity, pin it to 0
                if abs(s5) < 1e-8:
                    t6 = 0.0
                else:
                    T61 = np.linalg.inv(T) @ A1
                    t6 = np.arctan2(-T61[1, 2] / s5, T61[0, 2] / s5)

                A5 = dh(t5, self._d[4], self._a[4], self._al[4])
                A6 = dh(t6, self._d[5], self._a[5], self._al[5])
                T14 = np.linalg.inv(A1) @ T @ np.linalg.inv(A6) @ np.linalg.inv(A5)
                p13 = (T14 @ np.array([0.0, -d4, 0.0, 1.0]))[:3]
                L = np.linalg.norm(p13)
                if L < 1e-12:
                    continue

                # theta3 -- elbow up / down
                c3 = (L * L - a2 * a2 - a3 * a3) / (2.0 * a2 * a3)
                if abs(c3) > 1.0 + 1e-9:
                    continue
                ac3 = np.arccos(np.clip(c3, -1.0, 1.0))

                for t3 in (ac3, -ac3):
                    t2 = np.arctan2(-p13[1], -p13[0]) - np.arcsin(
                        np.clip(a3 * np.sin(t3) / L, -1.0, 1.0)
                    )
                    A2 = dh(t2, self._d[1], self._a[1], self._al[1])
                    A3 = dh(t3, self._d[2], self._a[2], self._al[2])
                    T34 = np.linalg.inv(A2 @ A3) @ T14
                    t4 = np.arctan2(T34[1, 0], T34[0, 0])
                    sols.append(wrap_to_pi([t1, t2, t3, t4, t5, t6]))
        return sols

    def ik_tcp(self, T_world_tcp):
        """Every candidate config reaching the target (a world TCP pose)."""
        return self.ik_flange(np.asarray(T_world_tcp, dtype=float) @ self.T_tcp_flange)

    # ---------------- selection ----------------
    def within_limits(self, q, margin=0.0):
        q = np.asarray(q, dtype=float)
        return bool(
            np.all(q >= self.jnt_lower + margin) and np.all(q <= self.jnt_upper - margin)
        )

    def configs_in_limits(self, T_world_tcp, margin=0.0):
        return [q for q in self.ik_tcp(T_world_tcp) if self.within_limits(q, margin)]

    def nearest_config(self, configs, q_ref, weights=None):
        """Closest config to q_ref, considering +-2pi rewinds.

        Wrapped configs are geometrically correct but may wind a joint the long
        way round from where the arm currently is. Motion-planning stage only.
        """
        if not configs:
            return None
        q_ref = np.asarray(q_ref, dtype=float)
        w = np.ones(6) if weights is None else np.asarray(weights, dtype=float)
        best, best_cost = None, np.inf
        for q in configs:
            cand = np.array(q, dtype=float)
            for j in range(6):
                opts = [
                    cand[j] + k * 2.0 * np.pi
                    for k in (-1, 0, 1)
                    if self.jnt_lower[j] <= cand[j] + k * 2.0 * np.pi <= self.jnt_upper[j]
                ]
                if opts:
                    cand[j] = min(opts, key=lambda o: abs(o - q_ref[j]))
            if not self.within_limits(cand):
                continue
            cost = float(np.sum(w * (cand - q_ref) ** 2))
            if cost < best_cost:
                best, best_cost = cand, cost
        return best

    def set_arm_qpos(self, q):
        self.data.qpos[self.arm_qadr] = np.asarray(q, dtype=float)


# ============================================================================
# Tests
# ============================================================================
def rule(t=""):
    print("\n" + "=" * 74)
    if t:
        print(t + "\n" + "=" * 74)


def test_math(kin, n=10000, seed=0):
    """FK -> IK -> FK must close on itself for arbitrary configurations."""
    rule(f"TEST A -- analytic round trip ({n} configs, tol {TOL_MATH:g})")
    rng = np.random.default_rng(seed)
    counts, worst, misses = np.zeros(9, int), 0.0, 0
    for _ in range(n):
        q = rng.uniform(-np.pi, np.pi, 6)
        T = kin.fk_flange(q)
        sols = kin.ik_flange(T)
        counts[len(sols)] += 1
        if not sols:
            misses += 1
            continue
        worst = max(worst, min(np.abs(kin.fk_flange(s) - T).max() for s in sols))
    print(f"worst round-trip error : {worst:.3e}")
    print(f"targets with 0 candidate configs: {misses}")
    print("candidate configs per target: "
          + ", ".join(f"{k}->{counts[k]}" for k in range(9) if counts[k]))
    ok = worst < TOL_MATH and misses == 0
    print(f"\n  RESULT: {'PASS' if ok else 'FAIL'}")
    return ok


def test_fk(kin, n=1000, seed=1):
    """Analytic FK must match what MuJoCo actually renders."""
    rule(f"TEST B -- FK vs MuJoCo ({n} configs, tol {TOL_MUJOCO:g})")
    print(f"calibration residual flange/tcp : "
          f"{kin.calib_error_flange:.3e} / {kin.calib_error_tcp:.3e}")
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(n):
        q = rng.uniform(kin.jnt_lower, kin.jnt_upper)
        reset_scene(kin.model, kin.data, arm_q=q)
        worst = max(worst, np.abs(kin.fk_tcp(q) - kin._site(kin.pinch_site_id)).max())
    print(f"worst TCP pose error   : {worst:.3e}")
    ok = worst < TOL_MUJOCO
    print(f"\n  RESULT: {'PASS' if ok else 'FAIL'}")
    return ok


def test_ik(kin, n=500, seed=2):
    """Every candidate config must land the real TCP on its target."""
    rule(f"TEST C -- IK replayed through MuJoCo ({n} targets, tol {TOL_MUJOCO:g})")
    rng = np.random.default_rng(seed)
    wp = wr = 0.0
    checked = 0
    for _ in range(n):
        T = kin.fk_tcp(rng.uniform(-np.pi, np.pi, 6))
        for s in kin.ik_tcp(T):
            reset_scene(kin.model, kin.data, arm_q=s)
            wp = max(wp, np.abs(kin.data.site_xpos[kin.pinch_site_id] - T[:3, 3]).max())
            wr = max(wr, np.abs(
                kin.data.site_xmat[kin.pinch_site_id].reshape(3, 3) - T[:3, :3]
            ).max())
            checked += 1
    print(f"candidate configs verified : {checked}")
    print(f"worst position / rotation error : {wp:.3e} m / {wr:.3e}")
    ok = max(wp, wr) < TOL_MUJOCO
    print(f"\n  RESULT: {'PASS' if ok else 'FAIL'}")
    return ok


def test_coverage(kin, n=3000, seed=3, r=(0.11, 0.98), z=(0.20, 1.60)):
    """How many modes a target has, for the two ways of generating targets.

    FK-first  : draw a configuration, take its pose. Reachable by construction.
    Pose-first: draw a position in the shell and a uniform random orientation.
                Honest about how the CVAE will be queried, but some poses have
                no valid config at all.
    """
    rule("TEST D -- mode coverage, free orientation")
    rng = np.random.default_rng(seed)

    for label, gen in [
        ("FK-first", lambda: kin.fk_tcp(rng.uniform(-np.pi, np.pi, 6))),
        ("pose-first", lambda: pose(
            [
                (rr := np.sqrt(rng.uniform(r[0] ** 2, r[1] ** 2)))
                * np.cos(th := rng.uniform(0, 2 * np.pi)),
                rr * np.sin(th),
                rng.uniform(*z),
            ],
            random_rotation(rng),
        )),
    ]:
        counts = np.zeros(9, int)
        for _ in range(n):
            counts[len(kin.configs_in_limits(gen()))] += 1
        reachable = counts[1:].sum()
        mean = sum(k * c for k, c in enumerate(counts)) / n
        print(f"\n{label:<11} configs in limits: "
              + ", ".join(f"{k}->{counts[k]}" for k in range(9) if counts[k]))
        print(f"{'':11} reachable {100*reachable/n:.1f}%   mean modes {mean:.2f}")

    print("\n  Joint limits only. Collision filtering happens downstream and will"
          "\n  cut these numbers further.")
    return True


def main():
    model, data, _ = build_model()
    kin = UR5eKinematics(model, data)

    rule("CALIBRATED PARAMETERS (measured from the compiled model)")
    print("  " + "  ".join(f"{k}={getattr(kin, k):.6f}"
                           for k in ["d1", "a2", "a3", "d4", "d5", "d6"]))
    print(f"  pinch site  : '{kin.pinch_site_name}'")
    print(f"  joint lower : {np.round(kin.jnt_lower, 4)}")
    print(f"  joint upper : {np.round(kin.jnt_upper, 4)}")

    results = [test_math(kin), test_fk(kin), test_ik(kin), test_coverage(kin)]
    rule("SUMMARY")
    print("ALL TESTS PASSED" if all(results) else "SOME TESTS FAILED")


if __name__ == "__main__":
    main()
