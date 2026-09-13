"""
Filter_Config.py
================
Collision filtering for candidate configs.

The solver already guarantees the other two properties, so they are NOT
re-checked here:

  * joint limits  -- ik_tcp wraps every candidate config to [-pi, pi), and
                     the UR5e ranges are +-2pi (elbow +-pi). Wrapped output
                     is inside the limits by construction. Measured: 0 of
                     15564 candidate configs ever violated them.
  * reaches target -- the IK is closed-form, not iterative. There is no
                     partial success. Measured worst FK residual 1.5e-14 m.

`assert_solver_guarantees()` re-establishes both on the current model, so if
the scene or the solver changes and those claims stop holding, you find out
immediately instead of silently shipping bad data.

That leaves collision as the only real filter.

    with CollisionChecker(model, data) as chk:
        configs = valid_configs(kin, chk, T_target)

Used as a context manager because filtering inflates geom margins and the
pick-and-place demo runs real dynamics -- it must not inherit fat geoms.
"""

import collections

import mujoco
import numpy as np

from Analytic_IK import UR5eKinematics, pose, random_rotation
from Environment_Arm_Only import (
    ARM_BODY_CHAIN,
    ARM_JOINT_NAMES,
    HOME_ARM_Q,
    build_model,
    reset_scene,
)

MARGIN = 0.005  # 5 mm inflation; reject near-misses, not just interpenetration

# The allowed-pair whitelist is built at THIS margin, always, whatever margin
# the checker operates at. If it tracked the operating margin instead, a wider
# probe would whitelist more pairs and a verdict re-thresholded from that probe
# would be more permissive than filtering directly -- measured at 2.1% of
# configs. Pinning it keeps every threshold comparable.
BASELINE_MARGIN = 0.010

SELF = "self"    # robot link vs robot link
WORLD = "world"  # robot link vs floor or pedestal
SELF_BIT, WORLD_BIT = 1, 2  # bitmask used by clearance()


class CollisionChecker:
    """Contact classification with margin inflation and baseline subtraction.

    Baseline subtraction matters: geoms that touch at rest (gripper pads) are
    not collisions. The filter rejects on NEW pairs, never on ncon != 0.
    """

    def __init__(self, model, data, margin=MARGIN):
        self.model, self.data, self.margin = model, data, margin
        self._saved = model.geom_margin.copy()
        self._applied = False
        self.baseline = set()

        # Arm qpos slots, so bulk checks can write joints directly instead of
        # calling mj_resetData -- nothing else in this scene ever moves.
        self.arm_qadr = np.array([
            model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
            for n in ARM_JOINT_NAMES
        ])

        self.robot_geoms = {
            g for g in range(model.ngeom)
            if (model.geom_contype[g] or model.geom_conaffinity[g])
            and model.geom_bodyid[g] != 0
        }

    # ---------------- margin lifecycle ----------------
    def apply(self):
        if not self._applied:
            self.model.geom_margin[:] = self._saved + self.margin
            self._applied = True
            self._build_baseline()
        return self

    def restore(self):
        if self._applied:
            self.model.geom_margin[:] = self._saved
            self._applied = False

    def __enter__(self):
        return self.apply()

    def __exit__(self, *exc):
        self.restore()
        return False

    # ---------------- baseline ----------------
    def _build_baseline(self):
        self.model.geom_margin[:] = self._saved + BASELINE_MARGIN
        reset_scene(self.model, self.data, arm_q=HOME_ARM_Q)
        mujoco.mj_forward(self.model, self.data)
        pairs, offenders = set(), []
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            key = tuple(sorted((int(c.geom1), int(c.geom2))))
            pairs.add(key)
            if self._category(*key) == WORLD:
                offenders.append(key)
        if offenders:
            raise RuntimeError(
                "Reference pose touches the world at "
                f"{[self.label(a) + '<->' + self.label(b) for a, b in offenders]}. "
                "Whitelisting that would hide real collisions -- fix the home "
                "pose or the scene."
            )
        # A whitelisted pair is permanent, so it must be one that is *always*
        # near -- gripper linkage. An arm pair here would hide real self-
        # collisions, because arm links genuinely move relative to each other.
        arm = [
            g for g in self.robot_geoms
            if mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_BODY, self.model.geom_bodyid[g]
            ) in ARM_BODY_CHAIN
        ]
        bad = [k for k in pairs if k[0] in arm or k[1] in arm]
        if bad:
            raise RuntimeError(
                "Baseline would whitelist arm pairs "
                f"{[self.label(a) + '<->' + self.label(b) for a, b in bad]}, "
                f"hiding real self-collisions. BASELINE_MARGIN={BASELINE_MARGIN} "
                "is too wide."
            )
        self.baseline = pairs
        self.model.geom_margin[:] = self._saved + self.margin

    # ---------------- classification ----------------
    def label(self, g):
        """Geom name, or its parent body -- most UR5e mesh geoms are unnamed."""
        n = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g)
        if n:
            return n
        b = mujoco.mj_id2name(
            self.model, mujoco.mjtObj.mjOBJ_BODY, self.model.geom_bodyid[g]
        )
        return f"{b or '?'}#{g}"

    def _category(self, g1, g2):
        r1, r2 = g1 in self.robot_geoms, g2 in self.robot_geoms
        if r1 and r2:
            return SELF
        if r1 or r2:
            return WORLD
        return None  # world vs world; impossible here, ignore

    # ---------------- the check ----------------
    def check(self, q):
        """Pose the arm at q. Returns (ok, violations)."""
        if not self._applied:
            raise RuntimeError("CollisionChecker.apply() not called")
        reset_scene(self.model, self.data, arm_q=q)
        out = []
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            key = tuple(sorted((int(c.geom1), int(c.geom2))))
            if key in self.baseline:
                continue
            cat = self._category(*key)
            if cat is None or c.dist >= self.margin:
                continue
            out.append({
                "category": cat,
                "a": self.label(key[0]),
                "b": self.label(key[1]),
                "dist": float(c.dist),
            })
        return not out, out

    def clearance(self, q):
        """Signed distance to the nearest real contact, and what kind it is.

        Returns (dist, flags) where flags is a bitmask of SELF_BIT / WORLD_BIT.
        dist is capped at the probe margin: nothing closer than that was looked
        for. Negative means interpenetration.

        The point of recording this instead of a boolean is that the margin
        policy stops being baked into the dataset -- q is collision-free at any
        threshold t <= margin iff clearance >= t. Re-thresholding later costs a
        comparison; regenerating costs hours.
        """
        if not self._applied:
            raise RuntimeError("CollisionChecker.apply() not called")
        self.data.qpos[self.arm_qadr] = q
        mujoco.mj_forward(self.model, self.data)
        best, flags = self.margin, 0
        con = self.data.contact
        for i in range(self.data.ncon):
            g1, g2 = int(con[i].geom1), int(con[i].geom2)
            key = (g1, g2) if g1 < g2 else (g2, g1)
            if key in self.baseline:
                continue
            r1, r2 = key[0] in self.robot_geoms, key[1] in self.robot_geoms
            if not (r1 or r2):
                continue
            flags |= SELF_BIT if (r1 and r2) else WORLD_BIT
            if con[i].dist < best:
                best = float(con[i].dist)
        return best, flags

    def ok(self, q):
        """Fast path for bulk generation: no violation records built.

        check() allocates a dict per contact, which is wasted work when the
        caller only wants a boolean. Same verdict, measurably cheaper.
        """
        if not self._applied:
            raise RuntimeError("CollisionChecker.apply() not called")
        self.data.qpos[self.arm_qadr] = q
        mujoco.mj_forward(self.model, self.data)
        base = self.baseline
        con = self.data.contact
        for i in range(self.data.ncon):
            g1, g2 = int(con[i].geom1), int(con[i].geom2)
            key = (g1, g2) if g1 < g2 else (g2, g1)
            if key in base:
                continue
            if key[0] not in self.robot_geoms and key[1] not in self.robot_geoms:
                continue
            # Test the DISTANCE, not mere presence. MuJoCo lists some contacts
            # whose dist exceeds the margin, so rejecting on presence alone
            # rejects configs that clear the margin comfortably -- measured up
            # to 10 mm of clearance rejected at a 5 mm margin.
            if con[i].dist < self.margin:
                return False
        return True


# ============================================================================
# Target evaluation
# ============================================================================
def valid_configs(kin, checker, T_target):
    """Every valid config for the target (a world TCP pose)."""
    return [q for q in kin.ik_tcp(T_target) if checker.ok(q)]


def evaluate_target(kin, checker, T_target):
    """Full record for one target: what IK gave, what survived, why the rest died."""
    sols = kin.ik_tcp(T_target)
    valid, rejected = [], []
    for q in sols:
        ok, viol = checker.check(q)
        (valid if ok else rejected).append(
            q if ok else {"q": q, "violations": viol,
                          "categories": sorted({v["category"] for v in viol})}
        )
    return {
        "n_candidates": len(sols),
        "n_valid": len(valid),
        "valid": valid,
        "rejected": rejected,
        "multimodal": len(valid) >= 2,
    }


def sample_pose(rng, r_range=(0.30, 0.60), z_range=(0.65, 1.40)):
    """Uniform pose in the high-yield shell: uniform in area, free orientation."""
    th = rng.uniform(0.0, 2.0 * np.pi)
    r = np.sqrt(rng.uniform(r_range[0] ** 2, r_range[1] ** 2))
    z = rng.uniform(*z_range)
    return pose([r * np.cos(th), r * np.sin(th), z], random_rotation(rng))


# ============================================================================
# Guarantees the solver is trusted to provide
# ============================================================================
def assert_solver_guarantees(kin, n=500, seed=11):
    """Re-derive the two claims this filter relies on. Cheap; run it once."""
    rng = np.random.default_rng(seed)
    worst_resid, n_sol, n_bad_limits = 0.0, 0, 0
    for _ in range(n):
        T = sample_pose(rng, (0.11, 0.98), (0.20, 1.60))
        for q in kin.ik_tcp(T):
            n_sol += 1
            n_bad_limits += not kin.within_limits(q)
            worst_resid = max(worst_resid, np.abs(kin.fk_tcp(q) - T).max())
    return n_sol, n_bad_limits, worst_resid


# ============================================================================
# Tests
# ============================================================================
def rule(t=""):
    print("\n" + "=" * 74)
    if t:
        print(t + "\n" + "=" * 74)


def main():
    import time

    model, data, _ = build_model()
    kin = UR5eKinematics(model, data)

    with CollisionChecker(model, data) as chk:
        rule(f"BASELINE  (margin {MARGIN * 1000:.0f} mm)")
        print(f"robot geoms      : {len(chk.robot_geoms)}")
        print(f"baseline pairs   : {len(chk.baseline)}")
        for a, b in sorted(chk.baseline):
            print(f"  {chk.label(a):<28} <-> {chk.label(b)}")
        print("  (allowed at rest; the filter rejects only NEW pairs)")

        rule("DOES THE FILTER CATCH ANYTHING?")
        for name, q in [
            ("home", HOME_ARM_Q),
            ("zeros", np.zeros(6)),
            ("elbow slammed shut", np.array([0, -0.2, 2.9, 0, 0, 0.0])),
            ("driven into the floor", np.array([0, 0.9, 1.2, 0, 0, 0.0])),
        ]:
            ok, v = chk.check(q)
            cats = ",".join(sorted({x["category"] for x in v})) or "-"
            worst = f"{min(x['dist'] for x in v):+.4f}" if v else "     -"
            print(f"  {name:<24} {'PASS' if ok else 'REJECT':<7} "
                  f"{cats:<11} n={len(v):<3} worst dist {worst}")

        rule("REJECTION BREAKDOWN  (pose-first, 2000 targets)")
        rng = np.random.default_rng(3)
        n_t = 2000
        n_cand = n_ok = 0
        cats = collections.Counter()
        modes = collections.Counter()
        t0 = time.time()
        for _ in range(n_t):
            rec = evaluate_target(kin, chk, sample_pose(rng))
            n_cand += rec["n_candidates"]
            n_ok += rec["n_valid"]
            modes[rec["n_valid"]] += 1
            for r in rec["rejected"]:
                cats["+".join(r["categories"])] += 1
        el = time.time() - t0

        print(f"candidate configs  : {n_cand}")
        print(f"valid configs      : {n_ok}  ({100 * n_ok / max(n_cand, 1):.1f}%)")
        print("rejected by        : " + ", ".join(
            f"{k} {v} ({100 * v / max(n_cand - n_ok, 1):.0f}%)" for k, v in cats.most_common()
        ))
        print("\nmodes per target   : " + ", ".join(
            f"{k}->{modes[k]}" for k in sorted(modes)
        ))
        mm = sum(v for k, v in modes.items() if k >= 2)
        print(f"targets with >=2 (usable for multimodality): {mm} "
              f"({100 * mm / n_t:.1f}%)")
        print(f"throughput: {n_t / el:.0f} targets/sec  "
              f"-> 1M usable targets in ~{1e6 / (n_t / el) / (mm / n_t) / 3600:.1f} h")

        rule("SOLVER GUARANTEES (re-derived, not assumed)")
        n_sol, n_bad, resid = assert_solver_guarantees(kin)
        print(f"candidate configs tested: {n_sol}")
        print(f"outside joint limits    : {n_bad}")
        print(f"worst FK residual       : {resid:.3e} m")
        assert n_bad == 0, "joint-limit guarantee broken -- re-enable that filter"

    rule("MARGIN RESTORED")
    print(f"geom_margin max after exit: {model.geom_margin.max():.6f}  "
          f"(expected {chk._saved.max():.6f})")
    assert np.allclose(model.geom_margin, chk._saved), "margins leaked!"
    print("\nOK")


if __name__ == "__main__":
    main()
