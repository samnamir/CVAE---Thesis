"""
Test_CVAE.py
============

    python Test_CVAE.py --model model_z2_L3_w256_b0.01_lr0.001.pt \
                        --data data/1M_configs.h5

RQ6 is the same script pointed at a wider-shell dataset:

    python Test_CVAE.py --model <same .pt> --data data/boundary_eval.h5 \
                        --split all --tag boundary

What each outputs: answers
------------------------
rq1_3_samples_<tag>.csv
rq4_unreachable_<tag>.csv  
rq5_swap_<tag>.csv        
rq5_latent_grid_<tag>.csv 

Everything is measured on the TEST split, which no training run ever saw.
"""

import argparse
import csv
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import Filter_Config as F
import h5py
import numpy as np
import torch
from Analytic_IK import UR5eKinematics
from Environment_Arm_Only import build_model
from Generate_Dataset import SPLIT_NAMES, load, rot6d_inv
from Train_CVAE import CVAE, DEV, from_cs, to_cs, wrap

torch.set_num_threads(1)        # the other half of the OpenMP guard above


# ---------------------------------------------------------------- model
def load_model(path):
    """Rebuild the network from the checkpoint's own recorded config."""
    ck = torch.load(path, map_location=DEV, weights_only=False)
    c = ck["cfg"]
    net = CVAE(c["latent"], c["layers"], c["width"]).to(DEV)
    net.load_state_dict(ck["state"])
    net.eval()
    print(f"loaded {path}")
    print("  " + "  ".join(f"{k}={ck[k]}" for k in
                           ("epoch", "dP_mm", "dR_deg", "feas_pct") if k in ck))
    return net, c


def target_pose(c):
    """The 4x4 TCP pose a condition row encodes."""
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = rot6d_inv(c[3:]), c[:3]
    return T


def branch_of(kin, T, q, eps):
    """Which analytic IK branch a config belongs to, or -1 for none.

    The label is the config's fixed slot in the solver's enumeration
    (shoulder x wrist x elbow), so it denotes the same posture at every
    target even when some slots have no solution. An index into a variable-
    length list would NOT work: a target missing a shoulder case returns
    four solutions, and everything after the gap shifts.
    """
    sols = kin.ik_tcp(T, slots=True)
    d = [
        np.inf if s is None else np.linalg.norm(wrap(np.asarray(s)[:5] - q[:5]))
        for s in sols
    ]
    j = int(np.argmin(d))
    return j if d[j] < eps else -1


def pose_error(kin, q, p_t, R_t):
    """FK the config, return (position error m, orientation error deg)."""
    T = kin.fk_tcp(q)
    dp = float(np.linalg.norm(T[:3, 3] - p_t))
    ct = (np.trace(T[:3, :3].T @ R_t) - 1) / 2
    return dp, float(np.degrees(np.arccos(np.clip(ct, -1, 1))))


# ---------------------------------------------------------------- RQ1/2/3/6
def evaluate(net, kin, chk, cond, modes, n_samp, eps, small, tag):
    """One pass over the test targets. Writes both CSVs, returns a summary."""
    srows, trows, spread = [], [], []
    with torch.no_grad():
        Q = net.sample(torch.from_numpy(cond).float().to(DEV),
                       n_samp).cpu().numpy()

    for i, c in enumerate(cond):
        p_t, R_t = c[:3], rot6d_inv(c[3:])
        M = modes[i]
        dp, dr, fe = [], [], []
        # hit[k] = has any of the first k+1 samples matched each true mode
        hit_s = np.zeros(len(M), bool)      # first `small` samples
        hit_a = np.zeros(len(M), bool)      # all n_samp samples

        for j, q in enumerate(Q[i]):
            a, b = pose_error(kin, q, p_t, R_t)
            ok = kin.within_limits(q) and chk.ok(q)
            dp.append(a * 1000)             # mm
            dr.append(b)
            fe.append(ok)
            dist = np.linalg.norm(wrap(M[:, :5] - q[:5]), axis=1)
            near = dist < eps
            hit_a |= near
            if j < small:
                hit_s |= near
            srows.append([i, j, f"{a * 1000:.3f}", f"{b:.3f}", int(ok),
                          f"{dist.min():.4f}"])

        dp, dr, fe = np.array(dp), np.array(dr), np.array(fe)
        # Same measure as RQ4 uses, so the two are comparable: if the model
        # scattered its samples on impossible targets, this is what would
        # differ. Without the reachable baseline the RQ4 number means nothing.
        spread.append(float(np.linalg.norm(Q[i].std(axis=0))))
        trows.append([
            f"{c[0]:.6f}", f"{c[1]:.6f}", f"{c[2]:.6f}", len(M),
            f"{np.median(dp[:small]):.2f}", f"{np.median(dp):.2f}",
            f"{np.median(dr[:small]):.3f}", f"{np.median(dr):.3f}",
            f"{100 * fe[:small].mean():.1f}", f"{100 * fe.mean():.1f}",
            f"{100 * hit_s.mean():.1f}" if len(M) > 1 else "",
            f"{100 * hit_a.mean():.1f}" if len(M) > 1 else "",
        ])

    with open(f"rq1_3_samples_{tag}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["target", "sample", "dP_mm", "dR_deg", "feasible",
                    "dist_to_branch"])
        w.writerows(srows)
    with open(f"rq1_3_per_target_{tag}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["x", "y", "z", "n_true_modes",
                    f"dP_med@{small}", f"dP_med@{n_samp}",
                    f"dR_med@{small}", f"dR_med@{n_samp}",
                    f"feas@{small}", f"feas@{n_samp}",
                    f"cov@{small}", f"cov@{n_samp}"])
        w.writerows(trows)

    d = np.array([float(r[2]) for r in srows])
    r = np.array([float(r[3]) for r in srows])
    f = np.array([int(r[4]) for r in srows], bool)
    cs = [float(t[10]) for t in trows if t[10]]
    ca = [float(t[11]) for t in trows if t[11]]
    return dict(n_targets=len(cond), n_samples=len(srows),
                dP_mean=d.mean(), dP_med=np.median(d), dP_p90=np.percentile(d, 90),
                dR_mean=r.mean(), dR_med=np.median(r), dR_p90=np.percentile(r, 90),
                feas=100 * f.mean(), feas_small=100 * np.mean(
                    [int(x[4]) for x in srows if x[1] < small]),
                cov_small=np.mean(cs), cov_all=np.mean(ca), n_multi=len(cs),
                spread=float(np.mean(spread)))


# ---------------------------------------------------------------- RQ4
def unreachable(net, kin, chk, path, split, n_max, n_samp, tag):
    """Targets with no IK solution at all. The model answers regardless."""
    with h5py.File(path, "r") as fh:
        nc, cond_all = fh["n_candidates"][:], fh["cond"][:]
        sel = nc == 0
        if split in SPLIT_NAMES:
            sel &= fh["split"][:] == SPLIT_NAMES.index(split)
    idx = np.flatnonzero(sel)
    if not len(idx):
        print("RQ4: no unreachable targets in this split -- skipped")
        return None
    idx = idx[:n_max]
    cond = cond_all[idx]

    with torch.no_grad():
        Q = net.sample(torch.from_numpy(cond).float().to(DEV),
                       n_samp).cpu().numpy()

    rows = []
    for i, c in enumerate(cond):
        p_t, R_t = c[:3], rot6d_inv(c[3:])
        e = np.array([pose_error(kin, q, p_t, R_t) for q in Q[i]])
        fe = np.mean([kin.within_limits(q) and chk.ok(q) for q in Q[i]])
        # Spread across samples: if the model "knew" a target were impossible
        # it might scatter. Compare this with the same figure on reachable
        # targets -- that comparison IS the result.
        spread = float(np.linalg.norm(Q[i].std(axis=0)))
        rows.append([f"{c[0]:.6f}", f"{c[1]:.6f}", f"{c[2]:.6f}",
                     f"{np.hypot(c[0], c[1]):.4f}",
                     f"{1000 * e[:, 0].mean():.2f}", f"{e[:, 1].mean():.3f}",
                     f"{100 * fe:.1f}", f"{spread:.4f}"])
    with open(f"rq4_unreachable_{tag}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["x", "y", "z", "radius", "dP_mm", "dR_deg",
                    "feas_pct", "sample_spread"])
        w.writerows(rows)
    dp = np.array([float(r[4]) for r in rows])
    return dict(n=len(rows), dP_mean=dp.mean(), dP_med=np.median(dp),
                feas=np.mean([float(r[6]) for r in rows]),
                spread=np.mean([float(r[7]) for r in rows]))


# ---------------------------------------------------------------- RQ5
def latent_grid(net, kin, chk, cond, modes, eps, n, tag):
    """Decode a grid over the latent for the target with the most modes."""
    i = int(np.argmax([len(m) for m in modes]))
    c, M = cond[i], modes[i]
    p_t, R_t = c[:3], rot6d_inv(c[3:])
    g = np.linspace(-3.0, 3.0, n)
    z = np.stack(np.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)

    cc = torch.from_numpy(np.repeat(c[None], len(z), 0)).float().to(DEV)
    with torch.no_grad():
        from Train_CVAE import from_cs
        out = net.dec(torch.cat([torch.from_numpy(z).float().to(DEV), cc], 1))
        Q = from_cs(out).cpu().numpy()

    rows = []
    for (z1, z2), q in zip(z, Q):
        d = np.linalg.norm(wrap(M[:, :5] - q[:5]), axis=1)
        b = int(np.argmin(d))
        dp, dr = pose_error(kin, q, p_t, R_t)
        rows.append([f"{z1:.4f}", f"{z2:.4f}", b, f"{d[b]:.4f}",
                     int(d[b] < eps), f"{1000 * dp:.2f}", f"{dr:.3f}",
                     int(kin.within_limits(q) and chk.ok(q))])
    with open(f"rq5_latent_grid_{tag}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["z1", "z2", "nearest_branch", "dist_to_branch",
                    "matched", "dP_mm", "dR_deg", "feasible"])
        w.writerows(rows)
    m = np.array([int(r[4]) for r in rows], bool)
    b = np.array([int(r[2]) for r in rows])
    return dict(target=i, n_modes=len(M), grid=n * n,
                matched=100 * m.mean(),
                branches_reached=len(set(b[m])) if m.any() else 0)


# ---------------------------------------------------------------- RQ5 swap
def latent_swap(net, kin, chk, cond, modes, eps, n_pairs, rng, tag):
    """Encode a config at target A, decode that latent at target B.

    If the latent encodes WHICH BRANCH and the condition supplies WHERE, then
    a latent taken from target A's elbow-up solution, decoded against target
    B, should come back as target B's elbow-up solution -- a target the model
    never saw paired with that latent.

    The random-latent control is what makes the number mean anything: a
    branch has to be hit more often than chance to count as transferred.
    """
    T = [target_pose(c) for c in cond]
    # branch label for every stored config, by target
    lab = [[branch_of(kin, T[i], q, eps) for q in modes[i]]
           for i in range(len(cond))]

    rows = []
    for _ in range(n_pairs * 6):          # oversample; most draws share nothing
        if len(rows) >= n_pairs:
            break
        i, j = rng.choice(len(cond), 2, replace=False)
        shared = sorted((set(lab[i]) & set(lab[j])) - {-1})
        if not shared:
            continue
        b = int(rng.choice(shared))
        q_a = modes[i][lab[i].index(b)]
        q_b = modes[j][lab[j].index(b)]

        with torch.no_grad():
            x = torch.from_numpy(to_cs(q_a[None])).to(DEV)
            c_a = torch.from_numpy(cond[i][None]).float().to(DEV)
            c_b = torch.from_numpy(cond[j][None]).float().to(DEV)
            mu = net.enc(torch.cat([x, c_a], 1))[:, :net.z]    # mean, not a draw
            q_s = from_cs(net.dec(torch.cat([mu, c_b], 1))).cpu().numpy()[0]
            z_r = torch.randn(1, net.z, device=DEV)
            q_c = from_cs(net.dec(torch.cat([z_r, c_b], 1))).cpu().numpy()[0]

        got = branch_of(kin, T[j], q_s, eps)
        ctl = branch_of(kin, T[j], q_c, eps)
        dp, dr = pose_error(kin, q_s, cond[j][:3], rot6d_inv(cond[j][3:]))
        rows.append([int(i), int(j), b, got, int(got == b), int(ctl == b),
                     f"{np.linalg.norm(wrap(q_s[:5] - q_b[:5])):.4f}",
                     f"{1000 * dp:.2f}", f"{dr:.3f}",
                     int(kin.within_limits(q_s) and chk.ok(q_s))])

    if not rows:
        print("RQ5 swap: no target pairs shared a branch -- skipped")
        return None
    with open(f"rq5_swap_{tag}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["target_a", "target_b", "branch_in", "branch_out",
                    "matched", "control_matched", "dist_to_B_branch",
                    "dP_mm", "dR_deg", "feasible"])
        w.writerows(rows)
    m = np.array([r[4] for r in rows])
    c = np.array([r[5] for r in rows])
    return dict(n=len(rows), matched=100 * m.mean(), control=100 * c.mean(),
                dP=np.mean([float(r[7]) for r in rows]),
                feas=100 * np.mean([r[9] for r in rows]))


# ---------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--split", default="test",
                   help="test (default), val, train, or all")
    p.add_argument("--tag", default="test", help="suffix for the output files")
    p.add_argument("--n-targets", type=int, default=500, dest="n_targets")
    p.add_argument("--n-samples", type=int, default=100, dest="n_samples")
    p.add_argument("--small", type=int, default=16,
                   help="the smaller sample budget to compare against")
    p.add_argument("--mode-eps", type=float, default=0.10, dest="mode_eps")
    p.add_argument("--margin", type=float, default=F.MARGIN)
    p.add_argument("--dedup-eps", type=float, default=0.10, dest="dedup_eps")
    p.add_argument("--grid", type=int, default=200, help="RQ5 grid side")
    p.add_argument("--n-pairs", type=int, default=300, dest="n_pairs",
                   help="RQ5 latent-swap target pairs")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    torch.manual_seed(a.seed)
    # MuJoCo BEFORE torch touches a tensor. Both link Intel's OpenMP runtime
    # (libiomp5md.dll on Windows); whichever initialises first owns it, and
    # the second aborts with "OMP: Error #15". torch initialises lazily, on
    # its first tensor operation -- which torch.load() in load_model() counts
    # as. Train_CVAE.py happens to build the scene before any tensor work,
    # which is why training never hit this. Same order here, deliberately:
    # it is a real fix, unlike KMP_DUPLICATE_LIB_OK, which Intel warns can
    # silently corrupt results.
    model, mdata, _ = build_model()
    kin = UR5eKinematics(model, mdata)

    net, cfg = load_model(a.model)
    split = None if a.split == "all" else a.split
    cond_r, q_r, tid = load(a.data, margin=a.margin, split=split,
                            dedup_eps=a.dedup_eps)
    print(f"{a.data} [{a.split}]: {len(q_r)} configs, "
          f"{len(np.unique(tid))} targets")

    rng = np.random.default_rng(a.seed)
    u = np.unique(tid)
    pick = rng.choice(u, size=min(a.n_targets, len(u)), replace=False)
    rows = {t: np.flatnonzero(tid == t) for t in pick}
    cond = np.stack([cond_r[r[0]] for r in rows.values()])
    modes = [q_r[r] for r in rows.values()]
    n_cand = [len(kin.ik_tcp(target_pose(c))) for c in cond]
    print("IK candidates per target:", np.bincount(n_cand, minlength=9).tolist())

    with F.CollisionChecker(model, mdata, margin=a.margin) as chk:
        print(f"\nRQ1/2/3 -- {len(cond)} targets x {a.n_samples} samples")
        s = evaluate(net, kin, chk, cond, modes, a.n_samples,
                     a.mode_eps, a.small, a.tag)
        print(f"  dP   mean {s['dP_mean']:6.1f} mm   median {s['dP_med']:6.1f}"
              f"   p90 {s['dP_p90']:6.1f}")
        print(f"  dR   mean {s['dR_mean']:6.2f} deg  median {s['dR_med']:6.2f}"
              f"   p90 {s['dR_p90']:6.2f}")
        print(f"  feasible  {s['feas_small']:.1f}% at {a.small} samples, "
              f"{s['feas']:.1f}% at {a.n_samples}")
        print(f"  coverage  {s['cov_small']:.1f}% at {a.small} samples, "
              f"{s['cov_all']:.1f}% at {a.n_samples}   "
              f"({s['n_multi']} multi-mode targets)")
        print(f"  sample spread {s['spread']:.3f} rad  "
              f"<- the RQ4 baseline")

        print("\nRQ4 -- unreachable targets")
        r = unreachable(net, kin, chk, a.data, a.split, a.n_targets,
                        a.small, a.tag)
        if r:
            print(f"  {r['n']} targets   dP mean {r['dP_mean']:.1f} mm, "
                  f"median {r['dP_med']:.1f}")
            print(f"  feasible {r['feas']:.1f}%   sample spread "
                  f"{r['spread']:.3f} rad")
            print(f"  vs reachable: dP {s['dP_mean']:.1f} mm, "
                  f"spread {s['spread']:.3f} rad, feasible {s['feas']:.1f}%")

        print(f"\nRQ5 -- latent grid {a.grid}x{a.grid}")
        g = latent_grid(net, kin, chk, cond, modes, a.mode_eps, a.grid, a.tag)
        print(f"  target {g['target']} has {g['n_modes']} true branches")
        print(f"  {g['matched']:.1f}% of the grid lands within "
              f"{a.mode_eps} rad of a branch")
        print(f"  {g['branches_reached']} of {g['n_modes']} branches "
              f"reachable from the latent")

        print(f"\nRQ5 -- latent swap ({a.n_pairs} target pairs)")
        w = latent_swap(net, kin, chk, cond, modes, a.mode_eps,
                        a.n_pairs, rng, a.tag)
        if w:
            print(f"  branch preserved in {w['matched']:.1f}% of swaps"
                  f"   (random latent: {w['control']:.1f}%)")
            print(f"  swapped configs: dP {w['dP']:.1f} mm, "
                  f"feasible {w['feas']:.1f}%")

    print(f"\nwrote rq1_3_samples_{a.tag}.csv, rq1_3_per_target_{a.tag}.csv, "
          f"rq4_unreachable_{a.tag}.csv, rq5_latent_grid_{a.tag}.csv, "
          f"rq5_swap_{a.tag}.csv")


if __name__ == "__main__":
    main()
