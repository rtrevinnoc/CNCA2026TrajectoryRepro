import numpy as np

from common.profiles import Profile4

M_BASE = 1400.0
M_ACT  = 0.5
M_LS_F = 49.0
M_SS_F = 28.0
M_ST_F = 10.5
M_MASK = 0.04
M_WAFR = 0.2

M_METRO = 520.0
M_OPT_B = 200.0
M_LENS  = 0.3

K8_W = 0.03158 * 1e9
C8_W = 0.0000355 * 1e8
K8_R = 0.0063165 * 1e9
C8_R = 710.0
K_LS_F = 0.01398 * 1e9; C_LS_F = 0.0003701 * 1e8
K_SS_F = 0.01462 * 1e9; C_SS_F = 0.0002861 * 1e8
K_ST_F = 0.01343 * 1e9; C_ST_F = 0.0001679 * 1e8
K1 = 0.00553 * 1e9; C1 = 0.001244 * 1e8
K2_O = 0.4619 * 1e9; C2_O = 0.0069308 * 1e8
K3_O = 6.396e7;      C3_O = 1.5994e5
K4_O = 1.07e6;       C4_O = 800.0

K1_ROT     = 1e9;  C1_ROT     = 1e8
K_LS_F_ROT = 1e9;  C_LS_F_ROT = 1e8
K_SS_F_ROT = 1e9;  C_SS_F_ROT = 1e8
K_ST_F_ROT = 1e9;  C_ST_F_ROT = 1e8
K8_W_ROT   = 1e9;  C8_W_ROT   = 1e8
K8_R_ROT   = 1e9;  C8_R_ROT   = 1e8
K2_O_ROT   = 1e9;  C2_O_ROT   = 1e8
K3_O_ROT   = 20.0; C3_O_ROT   = 60.0
K4_O_ROT   = 20.0; C4_O_ROT   = 60.0

I_ACT_XX  = 0.00400833;  I_ACT_ZZ  = 0.00801666
I_LS_F_XX = 4.10334;     I_LS_F_ZZ = 8.20668
I_SS_F_XX = 2.34477;     I_SS_F_ZZ = 4.68954
I_ST_F_XX = 0.879287;    I_ST_F_ZZ = 1.758574
I_MASK_XX = 0.000133667; I_MASK_ZZ = 0.000267334
I_WAFR_XX = 0.00512166;  I_WAFR_ZZ = 0.01024332
I_BASE_FULL = "1761.8 3921.3 3093.3 0 -799 0"
I_METRO_DIAG = "173.767 293.367 466.267"
I_OPT_B_DIAG = "50.6667 50.6667 16.0"
I_LENS_DIAG  = "0.006 0.006 0.0108"

KP_ACT = 1e7
KV_ACT_TRANS = KP_ACT * 0.01
KV_ACT_ROT   = KP_ACT * 0.1


class StepperController:
    def __init__(self, dies, die_l, v_scan, a_scan, j_max, s_max, alpha):
        self.dies = dies
        self.die_l = die_l
        self.v_scan = v_scan
        self.a_scan = a_scan
        self.j_max = j_max
        self.s_max = s_max
        self.alpha = alpha
        self.die_idx = 0
        self.state = "IDLE"
        self.t_start = 0
        t_ramp = (v_scan / a_scan) + (a_scan / j_max)
        d_ramp = v_scan * t_ramp
        self.profile = Profile4(v_scan, a_scan, j_max, s_max, self.die_l + d_ramp)
        self.d_offset = d_ramp / 2
        self.t_step = 0.2

    def get_ref(self, t):
        cx, cy = self.dies[self.die_idx]
        dt = t - self.t_start
        dir_y = 1 if (self.die_idx % 2 == 0) else -1
        next_idx = (self.die_idx + 1) % len(self.dies)
        nx, ny = self.dies[next_idx]
        if self.state == "IDLE":
            self.state = "SCANNING"; self.t_start = t
            yw = -cy - dir_y * self.d_offset; xw = -cx
            yr = self.d_offset / self.alpha; xr = 0
            return xw, yw, xr, yr
        elif self.state == "SCANNING":
            if dt > self.profile.t_total:
                self.state = "STEPPING"; self.t_start = t; dt = 0
            dist = self.profile.get_pos(dt)
            y_offset = -self.d_offset + dist
            yw = -cy + dir_y * y_offset; xw = -cx
            yr = -dir_y * y_offset / self.alpha; xr = 0
            return xw, yw, xr, yr
        elif self.state == "STEPPING":
            if dt > self.t_step:
                self.die_idx = next_idx; self.state = "SCANNING"; self.t_start = t; dt = self.t_step
            progress = min(1.0, dt / self.t_step)
            smooth_p = (1 - np.cos(progress * np.pi)) / 2
            start_yw = -cy + dir_y * (-self.d_offset + self.profile.distance)
            start_xw = -cx
            next_dir_y = 1 if (next_idx % 2 == 0) else -1
            end_yw = -ny - next_dir_y * self.d_offset; end_xw = -nx
            xw = start_xw + (end_xw - start_xw) * smooth_p
            yw = start_yw + (end_yw - start_yw) * smooth_p
            start_yr = -dir_y * (-self.d_offset + self.profile.distance) / self.alpha
            end_yr = next_dir_y * self.d_offset / self.alpha
            yr = start_yr + (end_yr - start_yr) * smooth_p
            xr = 0
            return xw, yw, xr, yr


BAL_JOINTS = (("w_ls_x", 0), ("w_ls_y", 1), ("r_ls_x", 0), ("r_ls_y", 1))


def make_reaction_canceller(model):
    base_id = model.body("base_frame").id
    entries = []
    for name, axis in BAL_JOINTS:
        joint = model.joint(name)
        entries.append((joint.qposadr[0], joint.dofadr[0],
                        model.actuator(name).id,
                        model.actuator(name + "_v").id, axis))

    def cancel(data):
        f_xy = [0.0, 0.0]
        for qadr, dadr, a_pos, a_vel, axis in entries:
            f = (KP_ACT * (data.ctrl[a_pos] - data.qpos[qadr])
                 + KV_ACT_TRANS * (data.ctrl[a_vel] - data.qvel[dadr]))
            f_xy[axis] += f
        data.xfrc_applied[base_id][0] = f_xy[0]
        data.xfrc_applied[base_id][1] = f_xy[1]

    return cancel


def get_wafer_dies(radius, die_l):
    dies = []
    n = int(np.ceil(2 * radius / die_l)) + 2
    for i in range(-n, n):
        for j in range(-n, n):
            cx = i * die_l; cy = j * die_l
            corners = [(cx-die_l/2, cy-die_l/2), (cx+die_l/2, cy-die_l/2),
                       (cx-die_l/2, cy+die_l/2), (cx+die_l/2, cy+die_l/2)]
            if all(np.sqrt(x**2 + y**2) < radius for x, y in corners): dies.append((cx, cy))
    rows = {}
    for x, y in dies:
        if y not in rows: rows[y] = []
        rows[y].append(x)
    sorted_dies = []
    y_coords = sorted(rows.keys())
    for i, y in enumerate(y_coords):
        x_coords = sorted(rows[y])
        if i % 2 == 1: x_coords = x_coords[::-1]
        for x in x_coords: sorted_dies.append((x, y))
    return sorted_dies


def get_model_xml(die_l, die_grid_xml):
    return f"""
<mujoco model="LITHO_FULL_MACHINE">
    <option integrator="implicit" timestep="0.0001" gravity="0 0 0"/>
    <visual>
        <headlight ambient="0.3 0.3 0.3" diffuse="0.8 0.8 0.8"/>
        <map shadowclip="2" shadowscale="0.6"/>
        <rgba haze="0.15 0.15 0.18 1"/>
    </visual>
    <asset>
        <material name="granite"   rgba="0.15 0.15 0.17 1"   specular="0.3"/>
        <material name="metrology" rgba="0.4 0.4 0.45 1"     specular="0.5"/>
        <material name="ls_steel"  rgba="0.3 0.35 0.45 1"    specular="0.6"/>
        <material name="ss_alum"   rgba="0.6 0.6 0.65 1"     specular="0.8"/>
        <material name="ws_alum"   rgba="0.7 0.7 0.75 1"     specular="0.9"/>
        <material name="silicon"   rgba="0.2 0.2 0.3 0.9"    specular="1.0" shininess="1"/>
        <material name="glass"     rgba="0.8 0.9 1.0 0.3"    specular="1.0"/>
        <material name="floor"     rgba="0.5 0.5 0.5 1"/>
    </asset>
    <worldbody>
        <light pos="1 1 5" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
        <geom type="plane" size="5 5 0.1" material="floor" contype="0" conaffinity="0"/>
        <body name="base_frame" pos="0 0 0.2">
            <inertial pos="0 0 0" mass="{M_BASE}" fullinertia="{I_BASE_FULL}"/>
            <joint name="base_x"  type="slide" axis="1 0 0" stiffness="{K1}"     damping="{C1}"/>
            <joint name="base_y"  type="slide" axis="0 1 0" stiffness="{K1}"     damping="{C1}"/>
            <joint name="base_rz" type="hinge" axis="0 0 1" stiffness="{K1_ROT}" damping="{C1_ROT}"/>
            <geom type="box" size="1.2 1.0 0.2" material="granite" contype="0" conaffinity="0"/>
            <body name="w_ls_act" pos="0 0 0.25">
                <inertial pos="0 0 0" mass="{M_ACT}" diaginertia="{I_ACT_XX} {I_ACT_XX} {I_ACT_ZZ}"/>
                <joint name="w_ls_x"  type="slide" axis="1 0 0" stiffness="0" damping="0"/>
                <joint name="w_ls_y"  type="slide" axis="0 1 0" stiffness="0" damping="0"/>
                <joint name="w_ls_rz" type="hinge" axis="0 0 1" stiffness="0" damping="{KV_ACT_ROT}"/>
                <geom type="box" size="0.4 0.3 0.02" material="ls_steel" contype="0" conaffinity="0"/>
                <body name="w_ls_flex" pos="0 0 0.03">
                    <inertial pos="0 0 0" mass="{M_LS_F}" diaginertia="{I_LS_F_XX} {I_LS_F_XX} {I_LS_F_ZZ}"/>
                    <joint name="w_lsf_x"  type="slide" axis="1 0 0" stiffness="{K_LS_F}"     damping="{C_LS_F}"/>
                    <joint name="w_lsf_y"  type="slide" axis="0 1 0" stiffness="{K_LS_F}"     damping="{C_LS_F}"/>
                    <joint name="w_lsf_rz" type="hinge" axis="0 0 1" stiffness="{K_LS_F_ROT}" damping="{C_LS_F_ROT}"/>
                    <geom type="box" size="0.38 0.28 0.03" material="ls_steel" contype="0" conaffinity="0"/>
                    <body name="w_ss_act" pos="0 0 0.04">
                        <inertial pos="0 0 0" mass="{M_ACT}" diaginertia="{I_ACT_XX} {I_ACT_XX} {I_ACT_ZZ}"/>
                        <joint name="w_ss_x"  type="slide" axis="1 0 0" stiffness="0" damping="{KV_ACT_TRANS}"/>
                        <joint name="w_ss_y"  type="slide" axis="0 1 0" stiffness="0" damping="{KV_ACT_TRANS}"/>
                        <joint name="w_ss_rz" type="hinge" axis="0 0 1" stiffness="0" damping="{KV_ACT_ROT}"/>
                        <geom type="box" size="0.2 0.2 0.02" material="ss_alum" contype="0" conaffinity="0"/>
                        <body name="w_ss_flex" pos="0 0 0.03">
                            <inertial pos="0 0 0" mass="{M_SS_F}" diaginertia="{I_SS_F_XX} {I_SS_F_XX} {I_SS_F_ZZ}"/>
                            <joint name="w_ssf_x"  type="slide" axis="1 0 0" stiffness="{K_SS_F}"     damping="{C_SS_F}"/>
                            <joint name="w_ssf_y"  type="slide" axis="0 1 0" stiffness="{K_SS_F}"     damping="{C_SS_F}"/>
                            <joint name="w_ssf_rz" type="hinge" axis="0 0 1" stiffness="{K_SS_F_ROT}" damping="{C_SS_F_ROT}"/>
                            <geom type="box" size="0.18 0.18 0.03" material="ss_alum" contype="0" conaffinity="0"/>
                            <body name="w_stage" pos="0 0 0.04">
                                <inertial pos="0 0 0" mass="{M_ST_F}" diaginertia="{I_ST_F_XX} {I_ST_F_XX} {I_ST_F_ZZ}"/>
                                <joint name="w_stage_x"  type="slide" axis="1 0 0" stiffness="{K_ST_F}"     damping="{C_ST_F}"/>
                                <joint name="w_stage_y"  type="slide" axis="0 1 0" stiffness="{K_ST_F}"     damping="{C_ST_F}"/>
                                <joint name="w_stage_rz" type="hinge" axis="0 0 1" stiffness="{K_ST_F_ROT}" damping="{C_ST_F_ROT}"/>
                                <geom type="cylinder" size="0.16 0.015" material="ws_alum" contype="0" conaffinity="0"/>
                                <body name="wafer" pos="0 0 0.02">
                                    <inertial pos="0 0 0" mass="{M_WAFR}" diaginertia="{I_WAFR_XX} {I_WAFR_XX} {I_WAFR_ZZ}"/>
                                    <joint name="wafer_x"  type="slide" axis="1 0 0" stiffness="{K8_W}"     damping="{C8_W}"/>
                                    <joint name="wafer_y"  type="slide" axis="0 1 0" stiffness="{K8_W}"     damping="{C8_W}"/>
                                    <joint name="wafer_rz" type="hinge" axis="0 0 1" stiffness="{K8_W_ROT}" damping="{C8_W_ROT}"/>
                                    <geom type="cylinder" size="0.15 0.001" material="silicon" contype="0" conaffinity="0"/>
{die_grid_xml}                                </body>
                            </body>
                        </body>
                    </body>
                </body>
            </body>
            <body name="r_ls_act" pos="0 0 1.2">
                <inertial pos="0 0 0" mass="{M_ACT}" diaginertia="{I_ACT_XX} {I_ACT_XX} {I_ACT_ZZ}"/>
                <joint name="r_ls_x"  type="slide" axis="1 0 0" stiffness="0" damping="0"/>
                <joint name="r_ls_y"  type="slide" axis="0 1 0" stiffness="0" damping="0"/>
                <joint name="r_ls_rz" type="hinge" axis="0 0 1" stiffness="0" damping="{KV_ACT_ROT}"/>
                <geom type="box" size="0.4 0.3 0.02" material="ls_steel" contype="0" conaffinity="0"/>
                <body name="r_ls_flex" pos="0 0 -0.03">
                    <inertial pos="0 0 0" mass="{M_LS_F}" diaginertia="{I_LS_F_XX} {I_LS_F_XX} {I_LS_F_ZZ}"/>
                    <joint name="r_lsf_x"  type="slide" axis="1 0 0" stiffness="{K_LS_F}"     damping="{C_LS_F}"/>
                    <joint name="r_lsf_y"  type="slide" axis="0 1 0" stiffness="{K_LS_F}"     damping="{C_LS_F}"/>
                    <joint name="r_lsf_rz" type="hinge" axis="0 0 1" stiffness="{K_LS_F_ROT}" damping="{C_LS_F_ROT}"/>
                    <geom type="box" size="0.38 0.28 0.03" material="ls_steel" contype="0" conaffinity="0"/>
                    <body name="r_ss_act" pos="0 0 -0.04">
                        <inertial pos="0 0 0" mass="{M_ACT}" diaginertia="{I_ACT_XX} {I_ACT_XX} {I_ACT_ZZ}"/>
                        <joint name="r_ss_x"  type="slide" axis="1 0 0" stiffness="0" damping="{KV_ACT_TRANS}"/>
                        <joint name="r_ss_y"  type="slide" axis="0 1 0" stiffness="0" damping="{KV_ACT_TRANS}"/>
                        <joint name="r_ss_rz" type="hinge" axis="0 0 1" stiffness="0" damping="{KV_ACT_ROT}"/>
                        <geom type="box" size="0.2 0.2 0.02" material="ss_alum" contype="0" conaffinity="0"/>
                        <body name="r_ss_flex" pos="0 0 -0.03">
                            <inertial pos="0 0 0" mass="{M_SS_F}" diaginertia="{I_SS_F_XX} {I_SS_F_XX} {I_SS_F_ZZ}"/>
                            <joint name="r_ssf_x"  type="slide" axis="1 0 0" stiffness="{K_SS_F}"     damping="{C_SS_F}"/>
                            <joint name="r_ssf_y"  type="slide" axis="0 1 0" stiffness="{K_SS_F}"     damping="{C_SS_F}"/>
                            <joint name="r_ssf_rz" type="hinge" axis="0 0 1" stiffness="{K_SS_F_ROT}" damping="{C_SS_F_ROT}"/>
                            <geom type="box" size="0.18 0.18 0.03" material="ss_alum" contype="0" conaffinity="0"/>
                            <body name="r_stage" pos="0 0 -0.04">
                                <inertial pos="0 0 0" mass="{M_ST_F}" diaginertia="{I_ST_F_XX} {I_ST_F_XX} {I_ST_F_ZZ}"/>
                                <joint name="r_stage_x"  type="slide" axis="1 0 0" stiffness="{K_ST_F}"     damping="{C_ST_F}"/>
                                <joint name="r_stage_y"  type="slide" axis="0 1 0" stiffness="{K_ST_F}"     damping="{C_ST_F}"/>
                                <joint name="r_stage_rz" type="hinge" axis="0 0 1" stiffness="{K_ST_F_ROT}" damping="{C_ST_F_ROT}"/>
                                <geom type="box" size="0.1 0.1 0.015" material="ws_alum" contype="0" conaffinity="0"/>
                                <body name="mask" pos="0 0 -0.02">
                                    <inertial pos="0 0 0" mass="{M_MASK}" diaginertia="{I_MASK_XX} {I_MASK_XX} {I_MASK_ZZ}"/>
                                    <joint name="mask_x"  type="slide" axis="1 0 0" stiffness="{K8_R}"     damping="{C8_R}"/>
                                    <joint name="mask_y"  type="slide" axis="0 1 0" stiffness="{K8_R}"     damping="{C8_R}"/>
                                    <joint name="mask_rz" type="hinge" axis="0 0 1" stiffness="{K8_R_ROT}" damping="{C8_R_ROT}"/>
                                    <geom type="box" size="0.05 0.05 0.005" material="glass" contype="0" conaffinity="0"/>
                                    <geom type="box" size="{die_l*2} {die_l*2} 0.006" pos="0 0 -0.005" rgba="1 1 0 0.2" contype="0" conaffinity="0"/>
                                </body>
                            </body>
                        </body>
                    </body>
                </body>
            </body>
            <body name="metro_frame" pos="0 0 0.5">
                <inertial pos="0 0 0" mass="{M_METRO}" diaginertia="{I_METRO_DIAG}"/>
                <joint name="metro_x"  type="slide" axis="1 0 0" stiffness="{K2_O}"     damping="{C2_O}"/>
                <joint name="metro_y"  type="slide" axis="0 1 0" stiffness="{K2_O}"     damping="{C2_O}"/>
                <joint name="metro_rz" type="hinge" axis="0 0 1" stiffness="{K2_O_ROT}" damping="{C2_O_ROT}"/>
                <geom type="box" size="0.1 0.1 0.5" pos="0.8 0.6 0" material="metrology" mass="{M_METRO/4}" contype="0" conaffinity="0"/>
                <geom type="box" size="0.1 0.1 0.5" pos="-0.8 0.6 0" material="metrology" mass="{M_METRO/4}" contype="0" conaffinity="0"/>
                <geom type="box" size="0.1 0.1 0.5" pos="0.8 -0.6 0" material="metrology" mass="{M_METRO/4}" contype="0" conaffinity="0"/>
                <geom type="box" size="0.1 0.1 0.5" pos="-0.8 -0.6 0" material="metrology" mass="{M_METRO/4}" contype="0" conaffinity="0"/>
                <geom type="box" size="0.9 0.7 0.05" pos="0 0 0.5" material="metrology" density="1" contype="0" conaffinity="0"/>
                <body name="optics_box" pos="0 0 0.3">
                    <inertial pos="0 0 0" mass="{M_OPT_B}" diaginertia="{I_OPT_B_DIAG}"/>
                    <joint name="optics_x"  type="slide" axis="1 0 0" stiffness="{K3_O}"     damping="{C3_O}"/>
                    <joint name="optics_y"  type="slide" axis="0 1 0" stiffness="{K3_O}"     damping="{C3_O}"/>
                    <joint name="optics_rz" type="hinge" axis="0 0 1" stiffness="{K3_O_ROT}" damping="{C3_O_ROT}"/>
                    <geom type="cylinder" size="0.2 0.2" material="metrology" mass="{M_OPT_B}" contype="0" conaffinity="0"/>
                    <body name="lens" pos="0 0 -0.1">
                        <inertial pos="0 0 0" mass="{M_LENS}" diaginertia="{I_LENS_DIAG}"/>
                        <joint name="lens_x"  type="slide" axis="1 0 0" stiffness="{K4_O}"     damping="{C4_O}"/>
                        <joint name="lens_y"  type="slide" axis="0 1 0" stiffness="{K4_O}"     damping="{C4_O}"/>
                        <joint name="lens_rz" type="hinge" axis="0 0 1" stiffness="{K4_O_ROT}" damping="{C4_O_ROT}"/>
                        <geom type="cylinder" size="0.1 0.05" material="glass" mass="{M_LENS}" contype="0" conaffinity="0"/>
                        <geom name="die_indicator" type="box" size="{die_l/2} {die_l/2} 0.002" pos="0 0 -0.25" rgba="1 1 0 0.6" contype="0" conaffinity="0"/>
                    </body>
                </body>
            </body>
        </body>
        <body name="exposure_field" pos="0 0 0.72">
            <geom type="box" size="{die_l/2} {die_l/2} 0.5" pos="0 0 0.25" rgba="0 1 1 0.05" contype="0" conaffinity="0"/>
        </body>
    </worldbody>
    <actuator>
        <position name="w_ls_x"  joint="w_ls_x"  kp="{KP_ACT}"/>
        <position name="w_ls_y"  joint="w_ls_y"  kp="{KP_ACT}"/>
        <position name="w_ls_rz" joint="w_ls_rz" kp="{KP_ACT}"/>
        <position name="w_ss_x"  joint="w_ss_x"  kp="{KP_ACT}"/>
        <position name="w_ss_y"  joint="w_ss_y"  kp="{KP_ACT}"/>
        <position name="w_ss_rz" joint="w_ss_rz" kp="{KP_ACT}"/>
        <position name="r_ls_x"  joint="r_ls_x"  kp="{KP_ACT}"/>
        <position name="r_ls_y"  joint="r_ls_y"  kp="{KP_ACT}"/>
        <position name="r_ls_rz" joint="r_ls_rz" kp="{KP_ACT}"/>
        <position name="r_ss_x"  joint="r_ss_x"  kp="{KP_ACT}"/>
        <position name="r_ss_y"  joint="r_ss_y"  kp="{KP_ACT}"/>
        <position name="r_ss_rz" joint="r_ss_rz" kp="{KP_ACT}"/>
        <velocity name="w_ls_x_v" joint="w_ls_x" kv="{KV_ACT_TRANS}"/>
        <velocity name="w_ls_y_v" joint="w_ls_y" kv="{KV_ACT_TRANS}"/>
        <velocity name="r_ls_x_v" joint="r_ls_x" kv="{KV_ACT_TRANS}"/>
        <velocity name="r_ls_y_v" joint="r_ls_y" kv="{KV_ACT_TRANS}"/>
    </actuator>
</mujoco>
"""
