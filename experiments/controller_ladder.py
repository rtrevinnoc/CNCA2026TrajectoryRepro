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

V, A, J, S = 0.8, 20.0, 1600.0, 1e5
DIE_L = 0.032
DIES = [(-0.048, 0.0), (-0.016, 0.0), (0.016, 0.0), (0.048, 0.0)]
T_EXP = 5.5e-3 / (V / lc.ALPHA)
WINDOW = max(1, int(T_EXP / lc.DT))

CONFIGS = ["case1", "case2", "case3a", "case3b"]
LABELS = {
    "case1": "Case 1: long-stroke PD only",
    "case2": "Case 2: + lag-compensation FF",
    "case3a": "Case 3a: + short-stroke PI",
    "case3b": "Case 3b: + short-stroke HIGS",
}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    print("Calibrating lag model lag = lv*v + la*a ...", flush=True)
    ff_w, ff_r = lc.calibrate_lag(V, A, J, S, verbose=True)
    print(f"  wafer  : lv = {ff_w[0]*1e3:.4f} mm/(m/s)   la = {ff_w[1]*1e6:.3f} um/(m/s^2)")
    print(f"  reticle: lv = {ff_r[0]*1e3:.4f} mm/(m/s)   la = {ff_r[1]*1e6:.3f} um/(m/s^2)")

    results, traces = {}, {}
    for cfg in CONFIGS:
        per_die = lc.run_wafer(DIES, DIE_L, V, A, J, S, cfg,
                               ff_w=ff_w, ff_r=ff_r, collect="all")
        stats = [lc.moving_stats(seg[len(seg)//3:2*len(seg)//3], WINDOW) for seg in per_die]
        ma = max(s[0] for s in stats)
        msd = max(s[1] for s in stats)
        results[cfg] = (ma, msd)
        traces[cfg] = per_die
        print(f"[{cfg}] cruise max|MA| = {ma*1e9:,.2f} nm   "
              f"cruise max MSD = {msd*1e9:,.2f} nm", flush=True)

    print("\nSUMMARY (spec: |MA| within [-1.25,+2] nm, MSD <= 7 nm)")
    for cfg in CONFIGS:
        ma, msd = results[cfg]
        ok = "PASS" if (ma < 2e-9 and msd < 7e-9) else "fail"
        print(f"  {LABELS[cfg]:<38s} MA {ma*1e9:>12,.2f} nm  "
              f"MSD {msd*1e9:>12,.2f} nm  [{ok}]")

    fig, axes = plt.subplots(4, 1, figsize=(8, 9), sharex=False)
    for ax, cfg in zip(axes, CONFIGS):
        seg = traces[cfg][1]
        t_ms = np.arange(len(seg)) * lc.DT * 1e3
        ax.plot(t_ms, np.array(seg) * 1e9, lw=0.7, color="#0A3D62")
        n = len(seg)
        ax.axvspan(t_ms[n // 3], t_ms[2 * n // 3], color="gold", alpha=0.25,
                   label="cruise (exposure)")
        ax.set_title(LABELS[cfg], fontsize=12, loc="left")
        ax.set_ylabel("$e_{syn}$ [nm]", fontsize=11)
        ax.tick_params(labelsize=10)
        ax.legend(fontsize=10, loc="upper right")
    axes[-1].set_xlabel("time within scan [ms]", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "controllers_esyn.png"), dpi=160, bbox_inches="tight")

    with open(os.path.join(args.out, "controllers_esyn.csv"), "w") as f:
        f.write("time_ms," + ",".join(CONFIGS) + "\n")
        t_ms = np.arange(len(traces[CONFIGS[0]][1])) * lc.DT * 1e3
        for i in range(len(t_ms)):
            row = [f"{t_ms[i]:.3f}"] + [f"{traces[cfg][1][i]*1e9:.3f}" for cfg in CONFIGS]
            f.write(",".join(row) + "\n")
    print(f"\nSaved {os.path.join(args.out, 'controllers_esyn.png')}")


if __name__ == "__main__":
    main()
