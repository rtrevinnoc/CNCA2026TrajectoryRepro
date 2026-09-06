import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from common import headless  # noqa: F401
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import controllers as lc
from common import profiles

V, A, J, S = 0.8, 20.0, 1600.0, 1e5
S_TIGHT = 2e4
DIE_L = 0.032
DIES = [(-0.048, 0.0), (-0.016, 0.0), (0.016, 0.0), (0.048, 0.0)]
T_EXP = 5.5e-3 / (V / lc.ALPHA)
WINDOW = max(1, int(T_EXP / lc.DT))

PROFILES = [
    ("2nd order (accel-bounded)", profiles.factory(2)),
    ("3rd order (jerk-bounded)", profiles.factory(3)),
    ("4th order (snap-bounded)", profiles.factory(4)),
    (f"4th order, tight snap ({S_TIGHT:g})", profiles.factory(4, s_tight=S_TIGHT)),
]
CONTROLLERS = [("case1", "Case 1 (PID only)"), ("case3b", "Case 3b (FF + HIGS)")]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    print("Calibrating feedforward ...", flush=True)
    ff_w, ff_r = lc.calibrate_lag(V, A, J, S)

    results, traces = {}, {}
    for pname, pf in PROFILES:
        t_scan = pf(V, A, J, S, 0.074).t_total
        for cfg, cname in CONTROLLERS:
            per_die = lc.run_wafer(DIES, DIE_L, V, A, J, S, cfg,
                                   ff_w=ff_w, ff_r=ff_r, collect="all",
                                   profile_factory=pf)
            stats = [lc.moving_stats(seg[len(seg)//3:2*len(seg)//3], WINDOW) for seg in per_die]
            ma = max(x[0] for x in stats)
            msd = max(x[1] for x in stats)
            full_msd = max(lc.moving_stats(seg, WINDOW)[1] for seg in per_die)
            results[(pname, cfg)] = (ma, msd, full_msd, t_scan)
            traces[(pname, cfg)] = per_die[1]
            print(f"[{pname:<28s} | {cfg}] t_scan={t_scan*1e3:6.1f}ms  "
                  f"cruise MA={ma*1e9:>12,.1f}nm  cruise MSD={msd*1e9:>11,.1f}nm  "
                  f"full-scan MSD={full_msd*1e9:>12,.1f}nm", flush=True)

    print("\nSUMMARY")
    print(f"{'profile':<30s} {'ctrl':<8s} {'t_scan':>8s} {'cruise MA':>13s} "
          f"{'cruise MSD':>13s} {'full MSD':>13s}")
    for (pname, cfg), (ma, msd, fmsd, ts) in results.items():
        print(f"{pname:<30s} {cfg:<8s} {ts*1e3:>6.1f}ms {ma*1e9:>11,.1f}nm "
              f"{msd*1e9:>11,.1f}nm {fmsd*1e9:>11,.1f}nm")

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=False)
    colors = {"2nd": "#c0392b", "3rd": "#e67e22", "4th order (snap-bounded)": "#0A3D62",
              "4th order, tight": "#27ae60"}
    for ax, (cfg, cname) in zip(axes, CONTROLLERS):
        for pname, _ in PROFILES:
            key = next(k for k in colors if pname.startswith(k))
            seg = traces[(pname, cfg)]
            t_ms = np.arange(len(seg)) * lc.DT * 1e3
            ax.plot(t_ms, np.array(seg) * 1e9, lw=0.8, label=pname, color=colors[key])
        ax.set_title(cname, fontsize=9, loc="left")
        ax.set_ylabel("$e_{syn}$ [nm]", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6.5, loc="upper right")
    axes[-1].set_xlabel("time within scan [ms]", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "order_comparison_esyn.png"), dpi=160, bbox_inches="tight")

    with open(os.path.join(args.out, "order_comparison_esyn.csv"), "w") as f:
        headers = [f"{p}_{c}" for c, _ in CONTROLLERS for p, _ in PROFILES]
        f.write("time_ms," + ",".join(headers) + "\n")
        t_ms = np.arange(len(traces[(PROFILES[0][0], CONTROLLERS[0][0])])) * lc.DT * 1e3
        for i in range(len(t_ms)):
            row = [f"{t_ms[i]:.3f}"] + [f"{traces[(p, c)][i]*1e9:.3f}"
                                        for c, _ in CONTROLLERS for p, _ in PROFILES]
            f.write(",".join(row) + "\n")
    print(f"\nSaved {os.path.join(args.out, 'order_comparison_esyn.png')}")


if __name__ == "__main__":
    main()
