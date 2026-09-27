"""
Generate_Dataset.py
===================
6-orientation dataset for the CVAE: condition = target TCP pose, sample = a
valid config reaching it. Every sampled position gets 6 fixed orientations
(down, up, front, back, left, right -- see DIRECTIONS), not a random one.
Generation runs until a set number of TARGETS (not raw draws) are valid, where
a target is valid if at least one of its candidate configs clears the default
margin.

    python Generate_Dataset.py --valid-targets 1000000 --out data/configs.h5
    cond, q, tid = load("data/configs.h5", margin=0.005, dedup_eps=0.10)

RECORD NOW, DECIDE LATER. 

  * unreachable targets   -- IK found nothing. Useless for training the CVAE,
                             but they are the reachability statistic, and the
                             negatives if you ever want a reachability head.
  * rejected configs      -- reached the target but hit something. Negatives,
                             and the only way to report WHY configs were lost.
  * signed clearance      -- not a valid/invalid boolean. A config is free at
                             any threshold t <= PROBE_MARGIN iff clearance >= t,
                             so the margin policy is no longer baked in and can
                             be swept without regenerating.
  * contact flags         -- self / world, so rejections stay attributable.

Two encoding choices that are NOT free to change later:

ROTATION IS 6D (first two columns of R). Quaternions double-cover -- q and -q
are one rotation but look maximally different to an encoder -- and Euler angles
wrap. Either injects variance the CVAE can absorb into the latent, corrupting
the mode structure being measured. rot6d_inv rebuilds R by Gram-Schmidt.

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

import argparse
import json
import time
from pathlib import Path

import Filter_Config as F
import h5py
import numpy as np
from Analytic_IK import UR5eKinematics
from Environment_Arm_Only import build_model

SHELL_R = (0.30, 0.60)    # high-yield shell; see Reach_Bounds.py
SHELL_Z = (0.65, 1.40)
PROBE_MARGIN = 0.010      # clearance is measured, and capped, at this
SPLIT_NAMES = ("train", "val", "test")   # stored as int8 codes into this tuple


def rot6d(R):
    """Rotation matrix -> continuous 6D encoding (its first two columns)."""
    return R[:, :2].T.ravel()


def rot6d_inv(v):
    """6D encoding -> rotation matrix, via Gram-Schmidt."""
    a, b = np.asarray(v, float)[:3], np.asarray(v, float)[3:]
    c1 = a / np.linalg.norm(a)
    c2 = b - c1 * (c1 @ b)
    c2 /= np.linalg.norm(c2)
    return np.stack([c1, c2, np.cross(c1, c2)], axis=1)


DIRECTIONS = ("down", "up", "front", "back", "left", "right")   # fixed order

# Every position gets all 6 of these as separate targets. Approach vector
# (the tool's z-axis) is +-radial / +-tangential / +-vertical IN THE LOCAL
# FRAME at that position, not fixed world axes -- so "left" means the same
# thing relative to the arm's reach at every angle theta around the base,
# matching the scene's own rotational symmetry (verified: rotating theta by
# a small step rotates the resulting pose by the same small step, no jumps).
# "front" = pointing away from the base, "left" = direction of increasing
# theta -- both arbitrary but fixed conventions, documented here once.
#
# There is no free yaw: the remaining rotation about the approach axis is
# pinned by a consistent up-reference (world Z for the four horizontal
# directions; the local radial direction for up/down, since world Z is
# parallel to the approach vector there and can't be used as a reference).


def local_frame(x, y):
    """radial, tangential, vertical unit vectors at a base-relative position."""
    th = np.arctan2(y, x)
    r_hat = np.array([np.cos(th), np.sin(th), 0.0])
    t_hat = np.array([-np.sin(th), np.cos(th), 0.0])
    return r_hat, t_hat, np.array([0.0, 0.0, 1.0])


def look_rotation(z_axis, up_hint):
    """Right-handed rotation with the given z-axis, x/y fixed via Gram-Schmidt
    against up_hint. up_hint must not be parallel to z_axis."""
    z = z_axis / np.linalg.norm(z_axis)
    x = np.cross(up_hint, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)


def canonical_pose(p, direction):
    """One of the 6 fixed approach orientations at position p = [x, y, z]."""
    r_hat, t_hat, z_hat = local_frame(p[0], p[1])
    approach = {
        "down": -z_hat, "up": z_hat,
        "front": r_hat, "back": -r_hat,
        "left": t_hat, "right": -t_hat,
    }[direction]
    up_hint = r_hat if direction in ("up", "down") else z_hat
    T = np.eye(4)
    T[:3, :3] = look_rotation(approach, up_hint)
    T[:3, 3] = p
    return T


def sample_position(rng, r_range=SHELL_R, z_range=SHELL_Z):
    """Uniform position in the shell (uniform in area, per Reach_Bounds.py)."""
    th = rng.uniform(0.0, 2.0 * np.pi)
    r = np.sqrt(rng.uniform(r_range[0] ** 2, r_range[1] ** 2))
    z = rng.uniform(*z_range)
    return np.array([r * np.cos(th), r * np.sin(th), z])


def generate(n_valid_targets, seed, margin=F.MARGIN, log_every=50_000):
    """Draw positions; build all 6 canonical-direction targets at each; solve,
    measure clearance. Stops once n_valid_targets targets have >=1 config with
    clearance >= margin (margin is ONLY the stopping rule here -- everything
    generated is still kept at every margin up to PROBE_MARGIN, same as before).

    Nothing is discarded: unreachable targets and rejected configs are kept.
    """
    model, data, _ = build_model()
    kin = UR5eKinematics(model, data)
    rng = np.random.default_rng(seed)

    cond_rows, direction_rows, n_cand_rows = [], [], []
    tid, qs, clr, flg = [], [], [], []
    n_valid = 0
    t0 = time.time()

    with F.CollisionChecker(model, data, margin=PROBE_MARGIN) as chk:
        while n_valid < n_valid_targets:
            p = sample_position(rng, SHELL_R, SHELL_Z)
            for d_idx, d in enumerate(DIRECTIONS):
                T = canonical_pose(p, d)
                i = len(cond_rows)
                cond_rows.append(np.concatenate([T[:3, 3], rot6d(T[:3, :3])]))
                direction_rows.append(d_idx)

                n_cand = 0
                target_valid = False
                for q in kin.ik_tcp(T):     # every candidate config, kept or not
                    dist, flags = chk.clearance(q)
                    tid.append(i)
                    qs.append(q)
                    clr.append(dist)
                    flg.append(flags)
                    n_cand += 1
                    target_valid |= dist >= margin
                n_cand_rows.append(n_cand)
                n_valid += target_valid

            if log_every and len(cond_rows) % log_every < len(DIRECTIONS):
                rate = n_valid / (time.time() - t0)
                print(f"  valid {n_valid:>9}/{n_valid_targets}  "
                      f"targets drawn {len(cond_rows):>9}  configs {len(qs):>9}  "
                      f"{rate:>6.0f} valid/s  eta {(n_valid_targets-n_valid)/rate/60:>5.1f} min")

    return dict(
        cond=np.asarray(cond_rows, np.float32),       # (T, 9) xyz + 6D rotation
        direction=np.asarray(direction_rows, np.int8), # (T,) index into DIRECTIONS
        n_candidates=np.asarray(n_cand_rows, np.int16),# (T,) candidates found
        target_id=np.asarray(tid, np.int32),           # (N,) row -> target
        q=np.asarray(qs, np.float32),                  # (N, 6) joint angles
        clearance=np.asarray(clr, np.float32),         # (N,) signed metres
        flags=np.asarray(flg, np.int8),                # (N,) 1=self 2=world 3=both
    ), time.time() - t0


def split_by_target(n_targets, seed, frac=(0.8, 0.1, 0.1)):
    """Assign whole targets to train/val/test. No target spans two splits.

    Returned as int8 codes indexing SPLIT_NAMES -- HDF5 stores fixed-width
    numeric arrays far more happily than strings.
    """
    idx = np.arange(n_targets)
    np.random.default_rng(seed).shuffle(idx)
    a, b = int(frac[0] * n_targets), int((frac[0] + frac[1]) * n_targets)
    out = np.empty(n_targets, np.int8)
    out[idx[:a]], out[idx[a:b]], out[idx[b:]] = 0, 1, 2
    return out


def save(path, arrays, meta):
    """Write one HDF5 file: every array a gzipped dataset, meta as an attribute.

    h5py does not create missing parent directories -- unlike np.savez, it
    fails with a bare FileNotFoundError that doesn't say which path is missing.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["meta"] = json.dumps(meta)
        for k, v in arrays.items():
            f.create_dataset(k, data=v, compression="gzip", compression_opts=4)
        f["split"].attrs["names"] = list(SPLIT_NAMES)
        f["direction"].attrs["names"] = list(DIRECTIONS)


def read_meta(path):
    """The JSON metadata block, without loading any arrays."""
    with h5py.File(path, "r") as f:
        return json.loads(f.attrs["meta"])


def wrap_to_pi(x):
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def dedup_modes(tid, q, clearance, eps):
    """Collapse configs of the same target that differ by less than eps.

    Distance is the wrapped L2 over q1..q5 only. q6 is excluded because it
    spins the tool about its own axis without changing arm posture -- though
    note that for a fully specified target pose q6 is determined, so in practice
    this catches ELBOW-BRANCH CONVERGENCE (elbow-up and elbow-down meeting as
    q3 -> 0, the arm straightening), not tool spin.

    Wrapping matters: -3.14 and +3.14 are 0.003 apart, not 6.28.

    Within a target the survivor is the one with the most clearance, so the
    kept representative is the safest, and the result does not depend on IK
    ordering. At most 8 configs per target, so this is ~28 comparisons of
    5-vectors -- negligible next to the mj_forward that produced them.

    Returns a boolean mask over the input rows.
    """
    keep = np.zeros(len(tid), bool)
    order = np.lexsort((-clearance, tid))           # target, then safest first
    tid_s, q_s = tid[order], q[order]
    for g in np.split(np.arange(len(order)), np.flatnonzero(np.diff(tid_s)) + 1):
        chosen = []
        for i in g:
            d = q_s[i, :5]
            if all(np.linalg.norm(wrap_to_pi(d - q_s[j, :5])) > eps for j in chosen):
                chosen.append(i)
                keep[order[i]] = True
    return keep


def load(path, margin=0.005, split=None, min_modes=1, dedup_eps=0.0):
    """Apply a margin (and optionally a split) to a stored dataset.

    Returns (cond, q, target_id), one row per valid config. Filtering happens
    here, not at generation, so the same file serves any margin up to
    PROBE_MARGIN and any dedup threshold. Rejected configs live in the file
    too -- never read `q` directly, or they come with it.

    dedup_eps: radians, wrapped L2 over q1..q5. 0.0 disables. See dedup_modes.
    """
    with h5py.File(path, "r") as f:
        clr = f["clearance"][:]
        rows = np.flatnonzero(clr >= margin)
        tid = f["target_id"][:][rows]
        q_all = f["q"][:]

        if split is not None:
            if split not in SPLIT_NAMES:
                raise ValueError(f"split must be one of {SPLIT_NAMES}")
            sel = f["split"][:][tid] == SPLIT_NAMES.index(split)
            rows, tid = rows[sel], tid[sel]

        if dedup_eps > 0:
            sel = dedup_modes(tid, q_all[rows], clr[rows], dedup_eps)
            rows, tid = rows[sel], tid[sel]

        if min_modes > 1:
            u, c = np.unique(tid, return_counts=True)
            sel = np.isin(tid, u[c >= min_modes])
            rows, tid = rows[sel], tid[sel]

        return f["cond"][:][tid], q_all[rows], tid


def main():
    p = argparse.ArgumentParser(description="Generate the pose-first CVAE dataset.")
    p.add_argument("--valid-targets", type=int, default=1_000_000, dest="valid_targets")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="config_dataset.h5")
    args = p.parse_args()

    print(f"generating until {args.valid_targets} valid targets "
          f"(seed {args.seed}, stopping margin {F.MARGIN*1000:.0f} mm, "
          f"probe margin {PROBE_MARGIN*1000:.0f} mm)")
    A, secs = generate(args.valid_targets, args.seed, margin=F.MARGIN)
    n_targets = len(A["cond"])
    A["split"] = split_by_target(n_targets, args.seed)

    free = A["clearance"] >= F.MARGIN
    _, counts = np.unique(A["target_id"][free], return_counts=True)
    meta = dict(
        valid_targets_requested=args.valid_targets,
        positions_drawn=n_targets // len(DIRECTIONS),
        targets_drawn=n_targets,                       # positions x 6 directions
        targets_unreachable=int((A["n_candidates"] == 0).sum()),
        candidate_configs=int(len(A["q"])),
        valid_configs_at_default=int(free.sum()),
        valid_targets_actual=int(len(counts)),
        mean_modes_at_default=round(float(counts.mean()), 3) if len(counts) else 0.0,
        multimodal_frac=round(float((counts >= 2).mean()), 4) if len(counts) else 0.0,
        directions=list(DIRECTIONS),
        seed=args.seed, probe_margin=PROBE_MARGIN, default_margin=F.MARGIN,
        shell_r=SHELL_R, shell_z=SHELL_Z, rotation="6d_first_two_columns",
        seconds=round(secs, 1),
    )
    save(args.out, A, meta)
    print("\n" + json.dumps(meta, indent=2))

    valid_target_mask = np.zeros(n_targets, bool)
    valid_target_mask[np.unique(A["target_id"][free])] = True
    print("\nvalid targets by direction:")
    for d_idx, d in enumerate(DIRECTIONS):
        sel = A["direction"] == d_idx
        print(f"  {d:<6} {sel.sum():>8} targets drawn  {valid_target_mask[sel].sum():>8} valid  "
              f"({100*valid_target_mask[sel].mean():5.1f}%)")

    print("\nmodes retained vs margin (re-thresholded, no regeneration):")
    for t in (0.0, 0.002, 0.005, 0.008, PROBE_MARGIN):
        _, c = np.unique(A["target_id"][A["clearance"] >= t], return_counts=True)
        print(f"  {t*1000:5.1f} mm  configs {int((A['clearance'] >= t).sum()):>8}  "
              f"mean modes {c.mean() if len(c) else 0:4.2f}  "
              f"multimodal {100*(c >= 2).mean() if len(c) else 0:5.1f}%")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
