"""
Integration_Check.py
====================
Run this before generating a dataset.

Each module already tests itself. This tests the SEAMS -- the places where they
share a mutable MuJoCo model, a frame convention, or a definition of "valid",
and where a disagreement would silently corrupt every row of the dataset rather
than crash.

Seven checks:

  1 SHARED STATE     filtering must not leave the model or data altered
  2 CLOSURE          every accepted config must actually land on its target
                     when replayed through MuJoCo, and MuJoCo must agree it is
                     collision-free -- verified independently of the filter
  3 COMPLETENESS     IK must return the config that generated the target;
                     a solver that drops candidates under-reports modes
  4 DISTINCTNESS     valid configs must be distinct; duplicates would
                     inflate the mode count the whole thesis rests on
  5 BOUNDS           Reach_Bounds must agree with what IK + filter accept
  6 MARGIN           margin must be monotone, and must be restored on exit
  7 DETERMINISM      same seed, same dataset

Exit code is non-zero if any check fails, so this can gate generation.
"""

import sys

import mujoco
import numpy as np

import Environment_Arm_Only as env
import Filter_Config as F
from Analytic_IK import UR5eKinematics, pose, random_rotation

POS_TOL = 1e-6      # m, replayed TCP position
ROT_TOL = 1e-6      # rotation matrix entries
SHELL_R = (0.30, 0.60)
SHELL_Z = (0.65, 1.40)

results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  --  {detail}" if detail else ""))
    return ok


def site_pose(kin):
    return pose(
        kin.data.site_xpos[kin.pinch_site_id],
        kin.data.site_xmat[kin.pinch_site_id].reshape(3, 3),
    )


def rule(t):
    print("\n" + "=" * 74 + f"\n{t}\n" + "=" * 74)


# ============================================================================
def main():
    model, data, meta = env.build_model()
    kin = UR5eKinematics(model, data)          # verify=True: FK vs MuJoCo
    rng = np.random.default_rng(0)

    # ---------------------------------------------------------------- 1
    rule("1  SHARED STATE -- filtering must not corrupt the model")
    margin_before = model.geom_margin.copy()
    qpos_before = data.qpos.copy()
    dh_before = [kin.d1, kin.a2, kin.a3, kin.d4, kin.d5, kin.d6]
    Tw_before = kin.T_world_dhbase.copy()

    arm = meta["arm_qpos_adr"]
    non_arm = np.setdiff1d(np.arange(model.nq), arm)

    with F.CollisionChecker(model, data) as chk:
        r = np.random.default_rng(1)
        for _ in range(300):
            F.valid_configs(kin, chk, F.sample_pose(r, SHELL_R, SHELL_Z))
        drift = np.abs(data.qpos[non_arm] - qpos_before[non_arm]).max()

    check("non-arm qpos untouched during filtering", drift < 1e-12, f"drift {drift:.2e}")
    check("geom_margin restored after exit",
          np.allclose(model.geom_margin, margin_before))
    check("kin calibration constants unchanged",
          np.allclose(dh_before, [kin.d1, kin.a2, kin.a3, kin.d4, kin.d5, kin.d6])
          and np.allclose(Tw_before, kin.T_world_dhbase))

    # ---------------------------------------------------------------- 2
    rule("2  CLOSURE -- accepted configs replayed independently through MuJoCo")
    worst_pos = worst_rot = 0.0
    n_acc = 0
    disagree = 0
    with F.CollisionChecker(model, data) as chk:
        base_pairs = set(chk.baseline)
        r = np.random.default_rng(2)
        targets = [F.sample_pose(r, SHELL_R, SHELL_Z) for _ in range(400)]
        for T in targets:
            for q in F.valid_configs(kin, chk, T):
                n_acc += 1
                # replay from a clean reset, not the filter's own fast path
                env.reset_scene(model, data, arm_q=q)
                Tm = site_pose(kin)
                worst_pos = max(worst_pos, np.abs(Tm[:3, 3] - T[:3, 3]).max())
                worst_rot = max(worst_rot, np.abs(Tm[:3, :3] - T[:3, :3]).max())
                # Independent collision verdict from raw contacts. Note the
                # dist test: MuJoCo lists some contacts beyond the margin, so
                # presence alone is not collision -- the same rule the filter
                # uses, re-derived here rather than called.
                hit = False
                for i in range(data.ncon):
                    c = data.contact[i]
                    g1, g2 = int(c.geom1), int(c.geom2)
                    key = (g1, g2) if g1 < g2 else (g2, g1)
                    if key in base_pairs or c.dist >= chk.margin:
                        continue
                    if key[0] in chk.robot_geoms or key[1] in chk.robot_geoms:
                        hit = True
                        break
                disagree += hit

    check("accepted configs reach their target position", worst_pos < POS_TOL,
          f"{n_acc} configs, worst {worst_pos:.2e} m")
    check("accepted configs reach their target orientation", worst_rot < ROT_TOL,
          f"worst {worst_rot:.2e}")
    check("MuJoCo independently agrees they are collision-free", disagree == 0,
          f"{disagree} disagreements")

    # ---------------------------------------------------------------- 3
    rule("3  COMPLETENESS -- IK must return the config that made the target")
    r = np.random.default_rng(3)
    missing = 0
    tested = 0
    for _ in range(3000):
        q = r.uniform(-np.pi, np.pi, 6)
        S = kin.ik_tcp(kin.fk_tcp(q))
        tested += 1
        if not S or min(np.abs(np.array(S) - q).max(axis=1)) > 1e-6:
            missing += 1
    check("FK config always recovered by IK", missing == 0,
          f"{missing}/{tested} generating configs not returned")

    # ---------------------------------------------------------------- 4
    rule("4  DISTINCTNESS -- modes must not be duplicates")
    r = np.random.default_rng(4)
    dup_targets = 0
    with F.CollisionChecker(model, data) as chk:
        for _ in range(2000):
            V = F.valid_configs(kin, chk, F.sample_pose(r, SHELL_R, SHELL_Z))
            if len(V) < 2:
                continue
            A = np.array(V)
            D = np.abs(A[:, None, :] - A[None, :, :]).max(-1)
            np.fill_diagonal(D, np.inf)
            if D.min() < 1e-6:
                dup_targets += 1
    check("no duplicate configs among a target's modes", dup_targets == 0,
          f"{dup_targets} targets had duplicates")

    # ---------------------------------------------------------------- 5
    rule("5  BOUNDS -- Reach_Bounds must agree with IK + filter")
    r = np.random.default_rng(5)
    outside = 0
    n_in = 0
    with F.CollisionChecker(model, data) as chk:
        for _ in range(1500):
            T = F.sample_pose(r, SHELL_R, SHELL_Z)
            V = F.valid_configs(kin, chk, T)
            if not V:
                continue
            n_in += 1
            for q in V:
                p = kin.fk_tcp(q)[:3, 3]
                rr, zz = np.hypot(p[0], p[1]), p[2]
                if not (0.005 <= rr <= 1.103 and 0.010 <= zz <= 2.052):
                    outside += 1
    check("every accepted TCP lies inside the measured reach envelope", outside == 0,
          f"{outside} of {n_in} targets fell outside")

    far = pose([1.5, 0.0, 1.0], random_rotation(r))
    check("target beyond max reach yields no candidate configs",
          len(kin.ik_tcp(far)) == 0)

    inside_axis = pose([0.0, 0.0, 0.30], np.eye(3))
    check("target on the base axis at low height is unreachable",
          len(kin.ik_tcp(inside_axis)) == 0, "shoulder-offset hole")

    # ---------------------------------------------------------------- 6
    rule("6  MARGIN -- monotone, and no leakage")
    r = np.random.default_rng(6)
    qs = []
    for _ in range(400):
        qs.extend(kin.ik_tcp(F.sample_pose(r, SHELL_R, SHELL_Z)))
    with F.CollisionChecker(model, data, margin=0.0) as c0:
        v0 = np.array([c0.ok(q) for q in qs])
    with F.CollisionChecker(model, data, margin=0.005) as c5:
        v5 = np.array([c5.ok(q) for q in qs])
    check("5 mm acceptance is a subset of 0 mm acceptance", bool((v5 <= v0).all()),
          f"0mm {v0.mean()*100:.1f}% free, 5mm {v5.mean()*100:.1f}% free, "
          f"margin costs {100*(v0&~v5).sum()/len(qs):.1f}%")
    check("margins restored after nested use",
          np.allclose(model.geom_margin, margin_before))

    # ---------------------------------------------------------------- 7
    rule("7  DETERMINISM -- same seed, same dataset")

    def build(seed, n=250):
        rr = np.random.default_rng(seed)
        rows = []
        with F.CollisionChecker(model, data) as c:
            for _ in range(n):
                T = F.sample_pose(rr, SHELL_R, SHELL_Z)
                for q in F.valid_configs(kin, c, T):
                    rows.append(np.concatenate([T[:3, 3], T[:3, :3].ravel(), q]))
        return np.array(rows)

    a, b = build(42), build(42)
    check("two runs at seed 42 are bit-identical",
          a.shape == b.shape and np.array_equal(a, b), f"{a.shape[0]} rows")
    check("a different seed gives different data", not np.array_equal(a, build(43)))

    # ----------------------------------------------------------------
    rule("SUMMARY")
    n_ok = sum(results)
    print(f"{n_ok}/{len(results)} checks passed")
    if n_ok != len(results):
        print("\nDO NOT GENERATE until these are resolved.")
        sys.exit(1)
    print("\nSafe to generate.")


if __name__ == "__main__":
    main()
