"""
Train_CVAE.py
=============
Train the CVAE, or sweep it. One value per flag -> one training run.
A comma-separated list -> a loop over every combination.

    python Train_CVAE.py --data config_dataset.h5
    python Train_CVAE.py --data config_dataset.h5 --latent 2,3 --beta 0.002,0.005 \
                         --layers 3,4 --epochs 20,50,100

Epochs are CHECKPOINTS, not separate runs: each run trains once to the largest
value and logs the full metric set at every listed value.

Condition : 9D target pose (xyz + 6D rotation), straight from the dataset.
Output    : 12D (cos, sin) of the six joints. Angles wrap, so plain MSE on
            radians punishes the net for being right at +-pi. atan2 rebuilds q.

Metrics
-------
every epoch (cheap, pure torch)
    mse         reconstruction, on the 12D cos/sin vector
    kl          KL divergence in latent space
    gnorm/med   average and median gradient size over the epoch, before
                clipping. Read the MEDIAN: one 500-million gradient makes the
                average meaningless, the median says what a normal batch was
    gnorm_max   the single largest batch gradient in the epoch
    n_clipped   how many batches hit the --clip threshold that epoch
    n_spikes    running count of logged gradient spikes

The saved model is the best one by dP, and only from --save-after onwards.
NOT by val_mse: while beta is ramping the latent is barely penalised

    val_loss    val_mse + beta * val_kl -- the training objective on val data
every --eval-every epochs, at each --epochs checkpoint, and at the end (MuJoCo)
    dP          FK position error of decoded configs, mm
    dR          FK orientation error, degrees
    feas%       decoded configs inside joint limits AND collision-free
    cov%        fraction of a target's true modes that sampling recovers,
                one column per --mode-eps threshold

"""

import argparse
import csv
import itertools
import time

import Filter_Config as F
import numpy as np
import torch
import torch.nn as nn
from Analytic_IK import UR5eKinematics
from Environment_Arm_Only import build_model
from Generate_Dataset import load, rot6d_inv

DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------- encoding
def to_cs(q):
    """(N,6) radians -> (N,12) [cos..., sin...]"""
    return np.concatenate([np.cos(q), np.sin(q)], 1).astype(np.float32)


def from_cs(v):
    """(N,12) torch -> (N,6) radians"""
    return torch.atan2(v[:, 6:], v[:, :6])


def wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


# ---------------------------------------------------------------- model
class CVAE(nn.Module):
    def __init__(self, z, layers, width, n_cond=9, n_out=12):
        super().__init__()
        self.z = z

        def mlp(n_in, n_final):
            seq, d = [], n_in
            for _ in range(layers):
                seq += [nn.Linear(d, width), nn.SiLU()]
                d = width
            return nn.Sequential(*seq, nn.Linear(d, n_final))

        self.enc = mlp(n_out + n_cond, 2 * z)
        self.dec = mlp(z + n_cond, n_out)

    def forward(self, x, c):
        h = self.enc(torch.cat([x, c], 1))
        mu, logvar = h[:, :self.z], h[:, self.z:]
        logvar = logvar.clamp(-10, 10)
        zs = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return self.dec(torch.cat([zs, c], 1)), mu, logvar

    def sample(self, c, n):
        """n configs per condition row. Returns (rows, n, 6) radians."""
        c = c.repeat_interleave(n, 0)
        zs = torch.randn(c.shape[0], self.z, device=c.device)
        return from_cs(self.dec(torch.cat([zs, c], 1))).view(-1, n, 6)


# ---------------------------------------------------------------- data
def get_data(path, margin, dedup_eps):
    out = {}
    for s in ("train", "val"):
        cond, q, tid = load(path, margin=margin, split=s, dedup_eps=dedup_eps)
        out[s] = (cond.astype(np.float32), q.astype(np.float32), tid)
    return out


def eval_set(cond, q, tid, n_targets, rng):
    """Pick n_targets distinct validation targets and their true modes."""
    u = np.unique(tid)
    pick = rng.choice(u, size=min(n_targets, len(u)), replace=False)
    rows = {t: np.flatnonzero(tid == t) for t in pick}
    conds = np.stack([cond[r[0]] for r in rows.values()])
    modes = [q[r] for r in rows.values()]
    return conds, modes


# ---------------------------------------------------------------- heavy eval
def task_metrics(net, kin, chk, conds, modes, n_samp, mode_eps):
    """FK errors, feasibility, mode coverage. Runs MuJoCo, so it is the slow one.

    mode_eps is a LIST. Coverage at several match thresholds costs nothing extra
    -- the decoded configs are already in hand -- and tells you whether a
    coverage number is real or an artefact of where the threshold was put.
    """
    net.eval()
    with torch.no_grad():
        Q = net.sample(torch.from_numpy(conds).to(DEV), n_samp).cpu().numpy()

    dp, dr, feas, cov, per_target = [], [], [], [], []
    for i, c in enumerate(conds):
        R_t, p_t = rot6d_inv(c[3:]), c[:3]
        hit = np.zeros((len(mode_eps), len(modes[i])), bool)
        for q in Q[i]:
            T = kin.fk_tcp(q)
            dp.append(np.linalg.norm(T[:3, 3] - p_t))
            ct = (np.trace(T[:3, :3].T @ R_t) - 1) / 2
            dr.append(np.degrees(np.arccos(np.clip(ct, -1, 1))))
            feas.append(kin.within_limits(q) and chk.ok(q))
            if len(modes[i]) > 1:
                d = np.linalg.norm(wrap(modes[i][:, :5] - q[:5]), axis=1)
                hit |= d[None, :] < np.asarray(mode_eps)[:, None]
        if len(modes[i]) > 1:
            cov.append(hit.mean(1))
            per_target.append((c[0], c[1], c[2], len(modes[i]), hit[0].mean()))
    net.train()
    cov = 100 * np.mean(cov, 0) if cov else np.full(len(mode_eps), np.nan)
    return np.mean(dp) * 1000, np.mean(dr), 100 * np.mean(feas), cov, per_target


# ---------------------------------------------------------------- one run
def head(mode_eps):
    return (f"{'ep':>4} {'mse':>9} {'kl':>8} {'gmed':>8} {'gmax':>10} {'nclip':>6} "
            f"{'beta':>7} {'val_mse':>9} {'val_kl':>8} "
            f"{'val_loss':>9} {'dP mm':>8} {'dR deg':>8} {'feas%':>7} "
            + " ".join(f"{'cov@' + str(e):>8}" for e in mode_eps))


def run(cfg, data, kin, chk, rng, writer):
    torch.manual_seed(cfg["seed"])
    ctr, qtr, _ = data["train"]
    cva, qva, tva = data["val"]

    Xtr = torch.from_numpy(to_cs(qtr)).to(DEV)
    Ctr = torch.from_numpy(ctr).to(DEV)
    Xva = torch.from_numpy(to_cs(qva)).to(DEV)
    Cva = torch.from_numpy(cva).to(DEV)
    e_cond, e_modes = eval_set(cva, qva, tva, cfg["eval_targets"], rng)

    net = CVAE(cfg["latent"], cfg["layers"], cfg["width"]).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    sched = (torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, factor=0.5, patience=15) if cfg["sched"] else None)
    n, bs = Xtr.shape[0], cfg["batch"]
    max_ep = max(cfg["epochs"])
    warm = max(1, int(0.3 * max_ep))            # beta ramps up over the first 30%

    # Checkpoint selection.
    #
    # NOT val_mse: while beta is still ramping up the latent is barely
    # penalised, so the model reconstructs almost perfectly and val_mse is at
    # its lowest around epoch 5-12 -- an under-regularised model with a dP of
    # ~100 mm. Picking the lowest val_mse therefore picks the WORST useful
    # model of the run. Two guards instead:
    #   1. ignore everything before beta has finished ramping (--save-after)
    #   2. judge on dP, the number that actually matters, measured on the
    #      heavy-eval epochs
    save_after = cfg["save_after"] if cfg["save_after"] > 0 else warm
    best_dp, best_ep, saved = float("inf"), 0, False

    print("\n" + "=" * 100)
    print("  ".join(f"{k}={cfg[k]}" for k in
                    ("latent", "layers", "width", "beta", "lr", "batch", "seed")))
    print("=" * 100 + "\n" + head(cfg["mode_eps"]))

    # Spike logs. TWO files, because 512 target rows per spike buries the shape
    # of the thing: the summary is one line per spike and is the one to read,
    # the targets file is for asking "where in the workspace were they".
    spikes = 0
    if cfg["spike"] > 0:
        with open(cfg["spike_out"], "w", newline="") as fh:
            csv.writer(fh).writerow(
                ["spike", "epoch", "batch", "of_batches", "time", "gnorm",
                 "typical_gnorm", "ratio", "clipped", "batch_mse",
                 "batch_kl", "beta"])
        with open(cfg["spike_xyz"], "w", newline="") as fh:
            csv.writer(fh).writerow(["spike", "epoch", "batch", "x", "y", "z"])

    for ep in range(1, max_ep + 1):
        beta = cfg["beta"] * min(1.0, ep / warm)
        perm = torch.randperm(n, device=DEV)
        tm = tk = tg = 0.0
        gmax, nclip = 0.0, 0
        gsum, gcnt, glist = 0.0, 0, []
        for nb, i in enumerate(range(0, n, bs), 1):
            b = perm[i:i + bs]
            out, mu, lv = net(Xtr[b], Ctr[b])
            mse = ((out - Xtr[b]) ** 2).mean()
            kl = (-0.5 * (1 + lv - mu ** 2 - lv.exp()).sum(1)).mean()
            opt.zero_grad()
            (mse + beta * kl).backward()
            # Measured BEFORE clipping, so the number tells you what the
            # gradients actually are -- not what the threshold forced them to
            # be. Set --clip a few times bigger than the typical value here.
            #
            # foreach=False: the fast path gathers every gradient tensor into
            # one contiguous workspace block. After ~100k updates the CUDA
            # cache is fragmented enough that no single block that size is
            # free, even with plenty of total VRAM, and it raises OOM. The
            # per-tensor path needs no such block.
            lim = cfg["clip"] if cfg["clip"] > 0 else 1e9
            gn = float(nn.utils.clip_grad_norm_(net.parameters(), lim,
                                                foreach=False))
            opt.step()

            # A spike is judged against this epoch's own typical gradient, not
            # a fixed number, because the normal size shrinks a lot as training
            # settles. "Typical" excludes batches already called spikes, so one
            # huge gradient cannot drag the reference up and hide the next.
            # The first 20 batches of an epoch set the reference and are never
            # judged, and epochs up to --spike-after are skipped entirely:
            # early training is jumpy by nature and those spikes are not the
            # interesting ones.
            ref = gsum / gcnt if gcnt else gn
            if (cfg["spike"] > 0 and ep > cfg["spike_after"] and gcnt > 20
                    and gn > cfg["spike"] * ref):
                if spikes < cfg["spike_max"]:
                    spikes += 1
                    stamp = time.strftime("%H:%M:%S")
                    nbatch = (n + bs - 1) // bs
                    with open(cfg["spike_out"], "a", newline="") as fh:
                        csv.writer(fh).writerow(
                            [spikes, ep, nb, nbatch, stamp, f"{gn:.4f}",
                             f"{ref:.4f}", f"{gn / ref:.1f}",
                             int(gn > lim), f"{mse.item():.6f}",
                             f"{kl.item():.4f}", f"{beta:.4f}"])
                    with open(cfg["spike_xyz"], "a", newline="") as fh:
                        w = csv.writer(fh)
                        for p in Ctr[b][:, :3].detach().cpu().numpy():
                            w.writerow([spikes, ep, nb, p[0], p[1], p[2]])
                    print(f"   ! spike {spikes:>3}  epoch {ep} batch {nb}/{nbatch}"
                          f" at {stamp}:  gnorm {gn:.3f} = {gn / ref:.1f}x"
                          f" typical {ref:.3f}")
            else:                       # only ordinary batches set the reference
                gsum += gn
                gcnt += 1
            glist.append(gn)

            tg += gn * len(b)
            gmax = max(gmax, gn)
            nclip += gn > lim
            tm += mse.item() * len(b)
            tk += kl.item() * len(b)
        tm, tk, tg = tm / n, tk / n, tg / n
        # The mean is wrecked by a single 500-million gradient; the median says
        # what a normal batch looked like. Read gmed for the trend, gmax for
        # the worst moment, and the gap between them for how spiky the run was.
        gmed = float(np.median(glist))
        if DEV == "cuda":
            torch.cuda.empty_cache()     # return fragmented blocks to the pool

        net.eval()
        with torch.no_grad():
            out, mu, lv = net(Xva, Cva)
            vm = ((out - Xva) ** 2).mean().item()
            vk = (-0.5 * (1 + lv - mu ** 2 - lv.exp()).sum(1)).mean().item()
        net.train()
        vl = vm + beta * vk

        if sched:
            sched.step(vl)

        heavy = ep % cfg["eval_every"] == 0 or ep in cfg["epochs"] or ep == max_ep
        dp = dr = fe = float("nan")
        cv = np.full(len(cfg["mode_eps"]), np.nan)
        if heavy:
            dp, dr, fe, cv, per_t = task_metrics(net, kin, chk, e_cond, e_modes,
                                                 cfg["n_samples"], cfg["mode_eps"])
            # Best model so far, on the terms set out above. The coverage map
            # is kept from this same epoch so the two files describe one model.
            if ep >= save_after and dp < best_dp:
                best_dp, best_ep, saved = dp, ep, True
                best_per_t = per_t
                torch.save({"state": net.state_dict(), "cfg": cfg, "epoch": ep,
                            "dP_mm": dp, "dR_deg": dr, "feas_pct": fe,
                            "val_mse": vm, "val_kl": vk}, cfg["ckpt"])

        print(f"{ep:>4} {tm:9.5f} {tk:8.4f} {gmed:8.3f} {gmax:10.3g} {nclip:6d} "
              f"{beta:7.4f} {vm:9.5f} {vk:8.4f} "
              f"{vl:9.5f} {dp:8.2f} {dr:8.2f} {fe:7.1f} "
              + " ".join(f"{c:8.1f}" for c in cv)
              + (f" lr {opt.param_groups[0]['lr']:.1e}" if sched else ""))
        row = dict(cfg, epoch=ep, mse=tm, kl=tk, gnorm=tg, gnorm_med=gmed,
                   gnorm_max=gmax, n_clipped=int(nclip), n_spikes=spikes,
                   beta_now=beta,
                   val_mse=vm, val_kl=vk, val_loss=vl,
                   dP_mm=dp, dR_deg=dr, feas_pct=fe,
                   epochs=max_ep, checkpoint=ep in cfg["epochs"])
        row.update({f"cov_pct@{e}": c for e, c in zip(cfg["mode_eps"], cv)})
        writer.writerow(row)

    # A run can only fail to save if it never reached save_after -- i.e. it was
    # far shorter than the beta ramp. Keep the last model rather than nothing.
    if not saved:
        best_ep, best_dp, best_per_t = ep, dp, per_t
        torch.save({"state": net.state_dict(), "cfg": cfg, "epoch": ep,
                    "dP_mm": dp, "note": "run ended before save_after"},
                   cfg["ckpt"])
        print(f"  ! run shorter than save_after={save_after}; kept final epoch")

    # Where coverage failed, in task space -- for the saved model's epoch.
    with open(cfg["cov_out"], "w", newline="") as fh:
        cw = csv.writer(fh)
        cw.writerow(["x", "y", "z", "n_true_modes", f"covered@{cfg['mode_eps'][0]}"])
        cw.writerows(best_per_t)
    print(f"  saved epoch {best_ep}  dP {best_dp:.1f} mm -> {cfg['ckpt']}\n"
          f"  coverage map -> {cfg['cov_out']}"
          + (f"   spikes -> {cfg['spike_out']} ({spikes})" if spikes else ""))
    return vl


# ---------------------------------------------------------------- main
def ints(s):
    return [int(x) for x in s.split(",")]


def floats(s):
    return [float(x) for x in s.split(",")]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="config_dataset.h5")
    p.add_argument("--out", default="sweep_log.csv")
    # swept (comma-separated = sweep)
    p.add_argument("--latent", type=ints, default=[2])
    p.add_argument("--layers", type=ints, default=[3])
    p.add_argument("--width", type=ints, default=[256])
    p.add_argument("--beta", type=floats, default=[0.005])
    p.add_argument("--epochs", type=ints, default=[100])
    # fixed
    p.add_argument("--lr", type=floats, default=[3e-4])
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--margin", type=float, default=F.MARGIN)
    p.add_argument("--dedup-eps", type=float, default=0.10, dest="dedup_eps")
    p.add_argument("--eval-targets", type=int, default=200, dest="eval_targets")
    p.add_argument("--n-samples", type=int, default=16, dest="n_samples")
    p.add_argument("--mode-eps", type=floats, default=[0.25], dest="mode_eps")
    p.add_argument("--eval-every", type=int, default=5, dest="eval_every")
    p.add_argument("--clip", type=float, default=0.0,
                   help="max gradient norm; 0 disables")
    p.add_argument("--spike", type=float, default=10.0,
                   help="log a batch whose gradient is this many times the "
                        "epoch's typical one; 0 disables")
    p.add_argument("--spike-max", type=int, default=500, dest="spike_max",
                   help="stop logging after this many spikes per run")
    p.add_argument("--spike-after", type=int, default=20, dest="spike_after",
                   help="ignore spikes up to this epoch (early training is "
                        "jumpy by nature)")
    p.add_argument("--save-after", type=int, default=0, dest="save_after",
                   help="epoch from which the best model may be saved; "
                        "0 means once beta has finished ramping up")
    p.add_argument("--sched", action="store_true",
                   help="ReduceLROnPlateau on val_loss (factor 0.5, patience 15)")
    a = p.parse_args()

    print(f"device {DEV}   loading {a.data}")
    data = get_data(a.data, a.margin, a.dedup_eps)
    print(f"train {len(data['train'][1])} configs   val {len(data['val'][1])}")

    model, mdata, _ = build_model()
    kin = UR5eKinematics(model, mdata)
    rng = np.random.default_rng(a.seed)

    cols = ["latent", "layers", "width", "beta", "lr", "batch", "seed", "epochs",
            "eval_targets", "n_samples", "mode_eps", "clip", "sched",
            "eval_every", "spike", "spike_after", "save_after",
            "epoch", "checkpoint",
            "mse", "kl", "gnorm", "gnorm_med", "gnorm_max", "n_clipped",
            "n_spikes", "beta_now", "val_mse", "val_kl", "val_loss",
            "dP_mm", "dR_deg", "feas_pct"] + [f"cov_pct@{e}" for e in a.mode_eps]

    grid = list(itertools.product(a.latent, a.layers, a.width, a.beta, a.lr))
    t0 = time.time()
    with open(a.out, "w", newline="") as fh, F.CollisionChecker(model, mdata,
                                                               margin=a.margin) as chk:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for z, L, W, b, lr in grid:
            tag = f"z{z}_L{L}_w{W}_b{b}_lr{lr}"
            cfg = dict(latent=z, layers=L, width=W, beta=b, lr=lr,
                       batch=a.batch, seed=a.seed, epochs=sorted(a.epochs),
                       eval_targets=a.eval_targets, n_samples=a.n_samples,
                       mode_eps=a.mode_eps, clip=a.clip, sched=a.sched,
                       eval_every=a.eval_every,
                       spike=a.spike, spike_max=a.spike_max,
                       spike_after=a.spike_after, save_after=a.save_after,
                       ckpt=f"model_{tag}.pt", cov_out=f"coverage_{tag}.csv",
                       spike_out=f"spikes_{tag}.csv",
                       spike_xyz=f"spikes_targets_{tag}.csv")
            run(cfg, data, kin, chk, rng, w)
            fh.flush()

    print(f"\n{len(grid)} run(s) in {(time.time()-t0)/60:.1f} min -> {a.out}")


if __name__ == "__main__":
    main()
