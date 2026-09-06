"""Scalar sensed field phi(p, t) over the coverage area.

SmoothField : sum of drifting Gaussian bumps with sinusoidal amplitude.
EventField  : SmoothField + a localised anomaly that grows linearly
              from t_event over `grow_time` seconds (spreading fire/plume).

All values are in arbitrary units; `dynamic_range` is used to scale
noise, the acceptable range epsilon, and naive-attack offsets.
"""
import numpy as np


class SmoothField:
    def __init__(self, area, rng, n_bumps=3, drift=0.2, period=600.0,
                 base=20.0, amp=10.0, width=120.0):
        self.area = area
        self.rng = rng
        self.n = n_bumps
        self.c0 = rng.uniform(0, area, size=(n_bumps, 2))
        ang = rng.uniform(0, 2 * np.pi, size=n_bumps)
        self.v = drift * np.stack([np.cos(ang), np.sin(ang)], axis=1)
        self.phase = rng.uniform(0, 2 * np.pi, size=n_bumps)
        self.period = period
        self.base = base
        self.amp = amp
        self.width = width
        self.dynamic_range = amp * 2.0

    def centres(self, t):
        c = self.c0 + self.v * t
        # reflect within area
        c = np.abs(c) % (2 * self.area)
        c = np.where(c > self.area, 2 * self.area - c, c)
        return c

    def value(self, p, t):
        """p: (..., 2) array. Returns field value(s)."""
        p = np.asarray(p, dtype=float)
        c = self.centres(t)
        a = self.amp * (0.6 + 0.4 * np.sin(2 * np.pi * t / self.period + self.phase))
        d2 = ((p[..., None, :] - c) ** 2).sum(-1)          # (..., n)
        return self.base + (a * np.exp(-d2 / (2 * self.width ** 2))).sum(-1)


class EventField(SmoothField):
    def __init__(self, area, rng, t_event=None, radius=60.0, grow_time=300.0,
                 event_amp=None, horizon=2000, **kw):
        super().__init__(area, rng, **kw)
        self.t_event = rng.uniform(0.15, 0.4) * horizon if t_event is None else t_event
        self.radius = radius
        self.grow_time = grow_time
        self.event_amp = 1.5 * self.amp if event_amp is None else event_amp
        self.centre = rng.uniform(0.2 * area, 0.8 * area, size=2)

    def value(self, p, t):
        v = super().value(p, t)
        if t < self.t_event:
            return v
        g = min(1.0, (t - self.t_event) / self.grow_time)
        p = np.asarray(p, dtype=float)
        d2 = ((p - self.centre) ** 2).sum(-1)
        return v + self.event_amp * g * np.exp(-d2 / (2 * self.radius ** 2))

    def growth_rate(self):
        """Peak temporal slope of the anomaly (units/s) during growth."""
        return self.event_amp / self.grow_time
