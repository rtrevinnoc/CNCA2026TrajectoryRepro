import math

import numpy as np


class _SegmentProfile:

    def __init__(self, segs, order, distance):
        self.order = order
        self.distance = distance
        segs = [(dt, u) for dt, u in segs if dt > 1e-12]
        self.t_bounds = np.zeros(len(segs) + 1)
        self.states = np.zeros((len(segs) + 1, 4))
        self.u = np.array([u for _, u in segs])
        for k, (dt, u) in enumerate(segs):
            self.t_bounds[k + 1] = self.t_bounds[k] + dt
            p, v, a, j = self.states[k]
            if order == 2:
                self.states[k + 1] = (p + v * dt + u * dt ** 2 / 2,
                                      v + u * dt, u, 0.0)
            elif order == 3:
                self.states[k + 1] = (p + v * dt + a * dt ** 2 / 2 + u * dt ** 3 / 6,
                                      v + a * dt + u * dt ** 2 / 2,
                                      a + u * dt, u)
            else:
                self.states[k + 1] = (
                    p + v * dt + a * dt ** 2 / 2 + j * dt ** 3 / 6 + u * dt ** 4 / 24,
                    v + a * dt + j * dt ** 2 / 2 + u * dt ** 3 / 6,
                    a + j * dt + u * dt ** 2 / 2,
                    j + u * dt)
        self.t_total = float(self.t_bounds[-1])

    def get_kinematics(self, t):
        if t <= 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0
        if t >= self.t_total:
            return self.states[-1][0], 0.0, 0.0, 0.0, 0.0
        k = int(np.searchsorted(self.t_bounds, t, side="right") - 1)
        dt = t - self.t_bounds[k]
        p, v, a, j = self.states[k]
        u = self.u[k]
        if self.order == 2:
            return (p + v * dt + u * dt ** 2 / 2, v + u * dt, u, 0.0, 0.0)
        if self.order == 3:
            return (p + v * dt + a * dt ** 2 / 2 + u * dt ** 3 / 6,
                    v + a * dt + u * dt ** 2 / 2, a + u * dt, u, 0.0)
        return (p + v * dt + a * dt ** 2 / 2 + j * dt ** 3 / 6 + u * dt ** 4 / 24,
                v + a * dt + j * dt ** 2 / 2 + u * dt ** 3 / 6,
                a + j * dt + u * dt ** 2 / 2, j + u * dt, u)

    def get_pos(self, t):
        return self.get_kinematics(t)[0]


class Profile2(_SegmentProfile):

    def __init__(self, v_max, a_max, distance, **_):
        v = min(v_max, math.sqrt(distance * a_max))
        t_acc = v / a_max
        t_v = (distance - v * t_acc) / v
        segs = [(t_acc, +a_max), (t_v, 0.0), (t_acc, -a_max)]
        super().__init__(segs, 2, distance)


class Profile3(_SegmentProfile):

    def __init__(self, v_max, a_max, j_max, distance, **_):
        def ramp(v):
            a = a_max
            t_j = a / j_max
            t_a = v / a - t_j
            if t_a < 0:
                a = math.sqrt(v * j_max)
                t_j = a / j_max
                t_a = 0.0
            return [(t_j, +j_max), (t_a, 0.0), (t_j, -j_max)]

        v = v_max
        d_acc = _SegmentProfile(ramp(v), 3, distance).states[-1][0]
        if 2 * d_acc > distance:
            lo, hi = 1e-6, v_max
            for _ in range(80):
                v = 0.5 * (lo + hi)
                d_acc = _SegmentProfile(ramp(v), 3, distance).states[-1][0]
                if 2 * d_acc > distance:
                    hi = v
                else:
                    lo = v
            v = lo
            d_acc = _SegmentProfile(ramp(v), 3, distance).states[-1][0]
        t_v = max(0.0, (distance - 2 * d_acc) / v)
        acc = ramp(v)
        dec = [(dt, -u) for dt, u in acc]
        super().__init__(acc + [(t_v, 0.0)] + dec, 3, distance)


class Profile4(_SegmentProfile):

    def __init__(self, v_max, a_max, j_max, s_max, distance, **_):
        s = s_max

        def ramp(v):
            j = j_max
            t_s = j / s
            if a_max / j < t_s:
                j = math.sqrt(a_max * s)
                t_s = j / s
            t_j = a_max / j - t_s
            a = a_max
            T_ar = 2 * t_s + t_j
            t_a = v / a - T_ar
            if t_a < 0:
                a_new = (-t_s + math.sqrt(t_s * t_s + 4 * v / j)) * j / 2
                if a_new / j >= t_s:
                    a = a_new
                    t_j = a / j - t_s
                else:
                    t_s = (v / (2 * s)) ** (1.0 / 3.0)
                    t_j = 0.0
                    a = s * t_s * t_s
                t_a = 0.0
            return [(t_s, +s), (t_j, 0.0), (t_s, -s), (t_a, 0.0),
                    (t_s, -s), (t_j, 0.0), (t_s, +s)]

        v = v_max
        d_acc = _SegmentProfile(ramp(v), 4, distance).states[-1][0]
        if 2 * d_acc > distance:
            lo, hi = 1e-6, v_max
            for _ in range(80):
                v = 0.5 * (lo + hi)
                d_acc = _SegmentProfile(ramp(v), 4, distance).states[-1][0]
                if 2 * d_acc > distance:
                    hi = v
                else:
                    lo = v
            v = lo
            d_acc = _SegmentProfile(ramp(v), 4, distance).states[-1][0]
        t_v = max(0.0, (distance - 2 * d_acc) / v)
        acc = ramp(v)
        dec = [(dt, -u) for dt, u in acc]
        super().__init__(acc + [(t_v, 0.0)] + dec, 4, distance)


def factory(order, s_tight=None):
    if order == 2:
        return lambda v, a, j, s, d: Profile2(v, a, d)
    if order == 3:
        return lambda v, a, j, s, d: Profile3(v, a, j, d)
    if s_tight is not None:
        return lambda v, a, j, s, d: Profile4(v, a, j, s_tight, d)
    return lambda v, a, j, s, d: Profile4(v, a, j, s, d)
