import argparse
import os
import pickle
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from common import headless  # noqa: F401
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from common import controllers as lc
from common import litho_model
from common import cnn_cost
from common import profiles

PROFILE_FACTORY = profiles.factory(4)

DIE_L = 0.032
WAFER_R = 0.150
T_STEP = 0.15
T_REF = 15.0
BOUNDS = np.array([[0.2, 1.5],
                   [5.0, 45.0],
                   [200.0, 5000.0],
                   [50.0, 5e5]])
N_PARTICLES = 6
N_ITER = 8
THRESH = 7e-9


def window_for(v):
    return max(1, int((5.5e-3 / (v / lc.ALPHA)) / lc.DT))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario", choices=["e3", "e4"])
    ap.add_argument("--config", default="case3b")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", choices=["linear", "log"], default="linear")
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--out", default="figures")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    dies = litho_model.get_wafer_dies(WAFER_R, DIE_L)

    print(f"[{args.scenario}] config={args.config} init={args.init} dies={len(dies)} "
          f"t_step={T_STEP}s T_ref={T_REF}s gamma={args.gamma}", flush=True)
    print("Calibrating feedforward at baseline ...", flush=True)
    ff_w, ff_r = lc.calibrate_lag(0.8, 20.0, 1600.0, 1e5)

    print(f"threshold = {THRESH*1e9:.1f} nm (absolute exposure-window spec)", flush=True)

    cmap = ListedColormap(["#f0f0f0", "#4caf50", "#f44336"])
    history = []

    def evaluate(x):
        v, a, j, s = x
        per_die = lc.run_wafer(dies, DIE_L, v, a, j, s, args.config,
                               ff_w=ff_w, ff_r=ff_r, collect="all", t_step=T_STEP,
                               profile_factory=PROFILE_FACTORY)
        w = window_for(v)
        msd = np.array([lc.moving_stats(seg[len(seg)//3:2*len(seg)//3], w)[1]
                        for seg in per_die])
        status = np.where(msd > THRESH, 2, 1)
        wm = cnn_cost.dies_to_map(dies, DIE_L, WAFER_R, status)
        d_ramp = v * ((v / a) + (a / j))
        t_scan = PROFILE_FACTORY(v, a, j, s, DIE_L + d_ramp).t_total
        t_wafer = len(dies) * (t_scan + T_STEP)
        tput = args.gamma * t_wafer / T_REF
        j_map, info = cnn_cost.map_cost(wm)
        nfail = int((status == 2).sum())
        if args.scenario == "e4":
            cost = j_map + tput
        else:
            cost = float(np.mean(msd)) / THRESH + tput
        history.append((tuple(x), cost, nfail, info["label"], j_map, t_wafer))
        print(f"  eval v={v:.3f} a={a:5.1f} j={j:6.0f} s={s:8.0f} "
              f"-> cost={cost:8.4f} fail={nfail:>2d}/{len(dies)} "
              f"CNN={info['label']:<9s} J_map={j_map:.4f} T={t_wafer:5.1f}s",
              flush=True)
        return cost, wm

    tag = args.scenario
    if args.init == "log":
        tag += "_spread"
    if args.gamma != 1.0:
        tag += f"_g{args.gamma:g}"
    ckpt = os.path.join(args.out, f".pso_ckpt_{tag}_{args.config}_{args.seed}.pkl")
    dim = 4
    start_it = 0
    if os.path.exists(ckpt):
        with open(ckpt, "rb") as f:
            ck = pickle.load(f)
        rng, pos, vel, pbest, pbest_cost = (ck["rng"], ck["pos"], ck["vel"],
                                            ck["pbest"], ck["pbest_cost"])
        gbest, gbest_cost, gbest_map = ck["gbest"], ck["gbest_cost"], ck["gbest_map"]
        history = ck["history"]
        start_it = ck["it"] + 1
        print(f"resumed from checkpoint at iteration {start_it}/{N_ITER}", flush=True)
    else:
        rng = np.random.default_rng(args.seed)
        if args.init == "log":
            log_lo, log_hi = np.log10(BOUNDS[:, 0]), np.log10(BOUNDS[:, 1])
            pos = 10 ** (log_lo + rng.random((N_PARTICLES, dim)) * (log_hi - log_lo))
        else:
            pos = BOUNDS[:, 0] + rng.random((N_PARTICLES, dim)) * (BOUNDS[:, 1] - BOUNDS[:, 0])
        pos[0] = [0.8, 20.0, 1600.0, 1e5]
        vel = np.zeros((N_PARTICLES, dim))
        pbest = pos.copy()
        pbest_cost = np.full(N_PARTICLES, np.inf)
        gbest = None
        gbest_cost = np.inf
        gbest_map = None

    for it in range(start_it, N_ITER):
        print(f"--- iteration {it+1}/{N_ITER} ---", flush=True)
        for k in range(N_PARTICLES):
            cost, wm = evaluate(pos[k])
            if cost < pbest_cost[k]:
                pbest_cost[k] = cost
                pbest[k] = pos[k].copy()
            if cost < gbest_cost:
                gbest_cost = cost
                gbest = pos[k].copy()
                gbest_map = wm
        for k in range(N_PARTICLES):
            r1, r2 = rng.random(dim), rng.random(dim)
            vel[k] = (0.6 * vel[k]
                      + 1.4 * r1 * (pbest[k] - pos[k])
                      + 1.4 * r2 * (gbest - pos[k]))
            span = BOUNDS[:, 1] - BOUNDS[:, 0]
            vel[k] = np.clip(vel[k], -0.5 * span, 0.5 * span)
            pos[k] = np.clip(pos[k] + vel[k], BOUNDS[:, 0], BOUNDS[:, 1])
        with open(ckpt, "wb") as f:
            pickle.dump(dict(rng=rng, pos=pos, vel=vel, pbest=pbest,
                             pbest_cost=pbest_cost, gbest=gbest,
                             gbest_cost=gbest_cost, gbest_map=gbest_map,
                             history=history, it=it), f)

    if os.path.exists(ckpt):
        os.remove(ckpt)

    print(f"\nBEST [{args.scenario}] cost={gbest_cost:.4f} at "
          f"v={gbest[0]:.3f} a={gbest[1]:.2f} j={gbest[2]:.0f} s={gbest[3]:.0f}",
          flush=True)
    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    ax.imshow(gbest_map, cmap=cmap, vmin=0, vmax=2, origin="lower", interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    fig.savefig(os.path.join(args.out, f"pso_best_map_{tag}.png"), dpi=160, bbox_inches="tight")

    with open(os.path.join(args.out, f"pso_history_{tag}.csv"), "w") as f:
        f.write("v,a,j,s,cost,failing,label,j_map,t_wafer\n")
        for (x, cost, nfail, label, j_map, t_wafer) in history:
            f.write(f"{x[0]:.4f},{x[1]:.2f},{x[2]:.0f},{x[3]:.0f},"
                    f"{cost:.5f},{nfail},{label},{j_map:.5f},{t_wafer:.2f}\n")
    print("history saved", flush=True)


if __name__ == "__main__":
    main()
