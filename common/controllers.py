import numpy as np

import mujoco

from common import litho_model

ALPHA = 0.25
DT = 1e-4
HOLD_TIME = 0.4
CTRL = {
    "w_ls_x": 0, "w_ls_y": 1, "w_ss_y": 4,
    "r_ls_x": 6, "r_ls_y": 7, "r_ss_y": 10,
    "w_ls_x_v": 12, "w_ls_y_v": 13, "r_ls_x_v": 14, "r_ls_y_v": 15,
}


class HIGS:
    def __init__(self, dt, kp, wi, wh, kh, clip):
        self.dt = dt
        self.kp = kp
        self.wi = wi
        self.wh = wh
        self.kh = kh
        self.clip = clip
        self.x1 = 0.0
        self.x2 = 0.0

    def update(self, e):
        if e * self.x1 >= (self.x1 ** 2) / self.kh:
            self.x1 += self.wh * e * self.dt
        else:
            self.x1 = self.kh * e
        dx2 = self.wi * self.x1 * self.dt
        u = self.kp * e + self.x2 + dx2
        if abs(u) < self.clip or u * dx2 < 0:
            self.x2 += dx2
        return float(np.clip(self.kp * e + self.x2, -self.clip, self.clip))


class PI:
    def __init__(self, dt, kp, ki, clip):
        self.dt = dt
        self.kp = kp
        self.ki = ki
        self.clip = clip
        self.i = 0.0

    def update(self, e):
        di = self.ki * e * self.dt
        u = self.kp * e + self.i + di
        if abs(u) < self.clip or u * di < 0:
            self.i += di
        return float(np.clip(self.kp * e + self.i, -self.clip, self.clip))


FINE_PI = dict(kp=16.0, ki=480.0)
FINE_HIGS = dict(kp=16.0, wi=2 * np.pi * 200.0, wh=2 * np.pi * 80.0, kh=1.0)
FINE_CLIP_W = 5e-3
FINE_CLIP_R = 2e-2


def build(dies, die_l):
    die_grid_xml = ""
    for cx, cy in dies:
        die_grid_xml += (
            f'<geom type="box" size="{die_l/2 - 0.0005} {die_l/2 - 0.0005} 0.0011" '
            f'pos="{cx} {cy} 0.001" rgba="0.3 0.3 0.4 1" contype="0" conaffinity="0"/>\n'
        )
    xml = litho_model.get_model_xml(die_l, die_grid_xml)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    return model, data


def ref_kin(controller, t):
    xw, yw, xr, yr = controller.get_ref(t)
    dt_loc = t - controller.t_start
    dir_y = 1 if (controller.die_idx % 2 == 0) else -1
    vw = aw = vr = ar = vxw = 0.0
    if controller.state == "SCANNING":
        _, v, a, _, _ = controller.profile.get_kinematics(dt_loc)
        vw, aw = dir_y * v, dir_y * a
        vr, ar = -dir_y * v / controller.alpha, -dir_y * a / controller.alpha
    elif controller.state == "STEPPING":
        cx, cy = controller.dies[controller.die_idx]
        next_idx = (controller.die_idx + 1) % len(controller.dies)
        nx, ny = controller.dies[next_idx]
        next_dir_y = 1 if (next_idx % 2 == 0) else -1
        start_yw = -cy + dir_y * (-controller.d_offset + controller.profile.distance)
        end_yw = -ny - next_dir_y * controller.d_offset
        start_yr = -dir_y * (-controller.d_offset + controller.profile.distance) / controller.alpha
        end_yr = next_dir_y * controller.d_offset / controller.alpha
        T = controller.t_step
        prog = min(1.0, dt_loc / T)
        dsm = (np.pi / (2 * T)) * np.sin(np.pi * prog)
        d2sm = (np.pi ** 2 / (2 * T ** 2)) * np.cos(np.pi * prog)
        vw, aw = (end_yw - start_yw) * dsm, (end_yw - start_yw) * d2sm
        vr, ar = (end_yr - start_yr) * dsm, (end_yr - start_yr) * d2sm
        vxw = (-nx - (-cx)) * dsm
    return xw, yw, xr, yr, vw, aw, vr, ar, vxw


def calibrate_lag(v=0.8, a=20.0, j=1600.0, s=1e5, iterations=3, verbose=False):
    die_l = 0.032
    dies = [(-0.016, 0.0), (0.016, 0.0)]
    ff_w = np.zeros(2)
    ff_r = np.zeros(2)
    for it in range(iterations):
        cfg = "case1" if it == 0 else "case2"
        _, extras = _run(dies, die_l, v, a, j, s, cfg,
                         ff_w=tuple(ff_w), ff_r=tuple(ff_r),
                         collect="all", want_lag_samples=True)
        (Vw, Aw, Lw), (Vr, Ar, Lr) = extras
        cw, *_ = np.linalg.lstsq(np.column_stack([Vw, Aw]), Lw, rcond=None)
        cr, *_ = np.linalg.lstsq(np.column_stack([Vr, Ar]), Lr, rcond=None)
        ff_w += cw
        ff_r += cr
        if verbose:
            print(f"  calib pass {it+1}: residual rms "
                  f"w={np.sqrt(np.mean(Lw**2))*1e6:.2f} um  "
                  f"r={np.sqrt(np.mean(Lr**2))*1e6:.2f} um", flush=True)
    return (float(ff_w[0]), float(ff_w[1])), (float(ff_r[0]), float(ff_r[1]))


def run_wafer(dies, die_l, v, a, j, s, config, ff_w=(0.0, 0.0), ff_r=(0.0, 0.0),
              collect="all", t_step=None, profile_factory=None):
    per, _ = _run(dies, die_l, v, a, j, s, config, ff_w=ff_w, ff_r=ff_r,
                  collect=collect, t_step=t_step, profile_factory=profile_factory)
    return per


def _run(dies, die_l, v, a, j, s, config, ff_w=(0.0, 0.0), ff_r=(0.0, 0.0),
         collect="all", t_step=None, want_lag_samples=False, profile_factory=None):
    controller = litho_model.StepperController(dies, die_l, v, a, j, s, ALPHA)
    if profile_factory is not None:
        controller.profile = profile_factory(v, a, j, s,
                                             controller.profile.distance)
    if t_step is not None:
        controller.t_step = t_step
    model, data = build(dies, die_l)
    cancel_reactions = litho_model.make_reaction_canceller(model)

    fine_w = fine_r = None
    if config == "case3a":
        fine_w = PI(DT, clip=FINE_CLIP_W, **FINE_PI)
        fine_r = PI(DT, clip=FINE_CLIP_R, **FINE_PI)
    elif config == "case3b":
        fine_w = HIGS(DT, clip=FINE_CLIP_W, **FINE_HIGS)
        fine_r = HIGS(DT, clip=FINE_CLIP_R, **FINE_HIGS)

    xw0, yw0, xr0, yr0 = controller.get_ref(0.0)
    controller.state = "IDLE"
    controller.die_idx = 0
    hold_w, hold_r = [], []
    while data.time < HOLD_TIME:
        data.ctrl[CTRL["w_ls_x"]] = xw0
        data.ctrl[CTRL["w_ls_y"]] = yw0
        data.ctrl[CTRL["r_ls_x"]] = xr0
        data.ctrl[CTRL["r_ls_y"]] = yr0
        cancel_reactions(data)
        mujoco.mj_step(model, data)
        if data.time > HOLD_TIME - 0.05:
            hold_w.append(data.body("wafer").xpos[1] - yw0)
            hold_r.append(data.body("mask").xpos[1] - yr0)
    off_w = float(np.mean(hold_w))
    off_r = float(np.mean(hold_r))

    t_origin = data.time
    per_die, cur = [], []
    lag_samples = ([], [], []), ([], [], [])
    last_die_idx = -1
    max_time = t_origin + len(dies) * (controller.profile.t_total
                                       + controller.t_step + 0.1)
    while data.time < max_time:
        t = data.time - t_origin
        xw, yw, xr, yr, vw, aw, vr, ar, vxw = ref_kin(controller, t)

        if controller.die_idx != last_die_idx and last_die_idx != -1:
            seg = np.array(cur)
            if collect == "cruise" and len(seg) >= 3:
                seg = seg[len(seg) // 3: 2 * len(seg) // 3]
            per_die.append(seg)
            cur = []
            if len(per_die) >= len(dies):
                break
        last_die_idx = controller.die_idx

        yw_cmd, yr_cmd = yw, yr
        if config in ("case2", "case3a", "case3b"):
            yw_cmd = yw - (ff_w[0] * vw + ff_w[1] * aw)
            yr_cmd = yr - (ff_r[0] * vr + ff_r[1] * ar)

        data.ctrl[CTRL["w_ls_x"]] = xw
        data.ctrl[CTRL["w_ls_y"]] = yw_cmd
        data.ctrl[CTRL["r_ls_x"]] = xr
        data.ctrl[CTRL["r_ls_y"]] = yr_cmd
        data.ctrl[CTRL["w_ls_x_v"]] = vxw
        data.ctrl[CTRL["w_ls_y_v"]] = vw
        data.ctrl[CTRL["r_ls_x_v"]] = 0.0
        data.ctrl[CTRL["r_ls_y_v"]] = vr

        e_w = data.body("wafer").xpos[1] - off_w - yw
        e_r = data.body("mask").xpos[1] - off_r - yr
        if fine_w is not None:
            data.ctrl[CTRL["w_ss_y"]] = fine_w.update(-e_w)
            data.ctrl[CTRL["r_ss_y"]] = fine_r.update(-e_r)

        if want_lag_samples:
            lag_samples[0][0].append(vw); lag_samples[0][1].append(aw)
            lag_samples[0][2].append(e_w)
            lag_samples[1][0].append(vr); lag_samples[1][1].append(ar)
            lag_samples[1][2].append(e_r)
        if controller.state == "SCANNING":
            cur.append(e_w - ALPHA * e_r)

        cancel_reactions(data)
        mujoco.mj_step(model, data)

    extras = None
    if want_lag_samples:
        extras = (tuple(np.array(x) for x in lag_samples[0]),
                  tuple(np.array(x) for x in lag_samples[1]))
    return per_die, extras


def moving_stats(seg, window):
    seg = np.asarray(seg)
    if len(seg) < window or window < 2:
        return float(np.abs(seg.mean())), float(seg.std())
    ma = np.convolve(seg, np.ones(window) / window, mode="valid")
    ma2 = np.convolve(seg ** 2, np.ones(window) / window, mode="valid")
    msd = np.sqrt(np.maximum(ma2 - ma ** 2, 0.0))
    return float(np.abs(ma).max()), float(msd.max())
