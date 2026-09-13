"""
Reach_Bounds.py
===============
Reachable workspace of the UR5e with FREE tool orientation.

Orientation is unconstrained, so the question is not "does IK solve this pose"
but "does any config put the TCP here". That is answered forwards:
sample joint space, run FK, keep the collision-free ones, and read off the
envelope.

Only q2..q5 matter. The TCP lies on the joint-6 axis so q6 does not move it,
and the scene is axisymmetric so q1 only spins the result about z. Both are
asserted at startup rather than assumed.

Reports the inner and outer radius of the reachable shell, and the same split
by height so you can see where the shell is actually thick.
"""

import Environment_Arm_Only as env
import numpy as np
from Analytic_IK import UR5eKinematics

N = 200_000
SEED = 0
Z_STEP = 0.10


def main():
    model, data, _ = env.build_model()
    kin = UR5eKinematics(model, data, verify=False)
    baseline = env.baseline_contacts(model, data)
    base_z = env.BASE_Z
    shoulder_z = base_z + kin.d1

    # --- the two symmetries this script rests on ----------------------------
    q = np.array([0.3, -1.0, 1.2, -0.7, 0.9, 0.0])
    p = kin.fk_tcp(q)[:3, 3]
    for i, label in [(5, "q6 must not move the TCP"), (0, "q1 must only rotate it")]:
        qq = q.copy()
        qq[i] += 0.8
        pp = kin.fk_tcp(qq)[:3, 3]
        moved = (
            np.linalg.norm(pp - p) if i == 5
            else abs(np.hypot(*pp[:2]) - np.hypot(*p[:2])) + abs(pp[2] - p[2])
        )
        assert moved < 1e-9, f"{label} (moved {moved:.2e})"

    # --- sample q2..q5; +-pi covers every distinct pose (angles wrap) --------
    rng = np.random.default_rng(SEED)
    Q = np.zeros((N, 6))
    Q[:, 1:5] = rng.uniform(-np.pi, np.pi, (N, 4))

    r_all, z_all, free = np.empty(N), np.empty(N), np.zeros(N, bool)
    for i, qi in enumerate(Q):
        p = kin.fk_tcp(qi)[:3, 3]
        r_all[i], z_all[i] = np.hypot(p[0], p[1]), p[2]
        free[i] = env.is_collision_free(model, data, qi, baseline)

    print(f"{N} samples, {free.mean()*100:.1f}% collision-free")
    print(f"base z = {base_z:.3f}   shoulder z = {shoulder_z:.3f}")

    for label, mask in [("kinematic only", np.ones(N, bool)), ("collision-free", free)]:
        r, z = r_all[mask], z_all[mask]
        d = np.hypot(r, z - shoulder_z)          # distance from the shoulder
        print(f"\n--- {label} ---")
        print(f"radius from z-axis : {r.min():.3f} .. {r.max():.3f} m")
        print(f"height z           : {z.min():.3f} .. {z.max():.3f} m")
        print(f"shell about shoulder: {d.min():.3f} .. {d.max():.3f} m")

    # --- envelope by height, collision-free ---------------------------------
    r, z = r_all[free], z_all[free]
    # Hard min/max is a knife edge -- almost no configs live there, so a
    # target placed on it is reachable in principle and useless in practice.
    # p2/p98 is the shell that is actually densely populated.
    print("\nenvelope by height (collision-free)")
    print(f"{'z':>7} {'r_min':>7} {'r_p2':>7} {'r_p98':>7} {'r_max':>7} {'n':>7}")
    edges = np.arange(np.floor(z.min() / Z_STEP) * Z_STEP, z.max() + Z_STEP, Z_STEP)
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = r[(z >= lo) & (z < hi)]
        if s.size < 50:
            continue
        lo2, hi2 = np.percentile(s, [2, 98])
        print(
            f"{(lo+hi)/2:7.2f} {s.min():7.3f} {lo2:7.3f} {hi2:7.3f} "
            f"{s.max():7.3f} {s.size:7d}"
        )

    # --- a shell that is dense at every height it spans ---------------------
    band = (z >= 0.20) & (z <= 1.60)
    lo2, hi2 = np.percentile(r[band], [2, 98])
    print(
        f"\nSAMPLE SHELL : r in [{lo2:.2f}, {hi2:.2f}]  z in [0.20, 1.60]"
        "  (theta free, orientation free)"
        "\n  Hard limits are wider, but the outer edge is a knife edge: reachable"
        "\n  by a single config only, so there are no modes to learn there."
    )


if __name__ == "__main__":
    main()
