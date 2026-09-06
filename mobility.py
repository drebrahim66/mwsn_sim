"""Mobility models and contact (encounter) detection.

RandomWaypoint : each node picks a random destination, moves at a
                 per-node speed drawn around `speed`, repeats.
TraceMobility  : replays a (time, node, x, y) CSV rescaled to the area.

contacts(pos, rng_range) returns the set of (i, j) pairs currently
within communication range, computed with a uniform grid.
"""
import numpy as np


class RandomWaypoint:
    def __init__(self, n, area, speed, rng, speed_jitter=0.3):
        self.n = n
        self.area = area
        self.rng = rng
        self.pos = rng.uniform(0, area, size=(n, 2))
        self.dest = rng.uniform(0, area, size=(n, 2))
        lo, hi = speed * (1 - speed_jitter), speed * (1 + speed_jitter)
        self.speed = rng.uniform(lo, hi, size=n)

    def step(self, dt=1.0):
        d = self.dest - self.pos
        dist = np.linalg.norm(d, axis=1)
        arrive = dist <= self.speed * dt
        move = ~arrive
        self.pos[move] += (d[move] / dist[move, None]) * (self.speed[move] * dt)[:, None]
        self.pos[arrive] = self.dest[arrive]
        k = arrive.sum()
        if k:
            self.dest[arrive] = self.rng.uniform(0, self.area, size=(k, 2))
        return self.pos


class TraceMobility:
    """Replay a trace. CSV columns: t,node,x,y (any units). Nodes beyond
    the trace count are assigned by wrapping; positions rescaled to area."""

    def __init__(self, n, area, path, rng):
        import csv
        rows = {}
        xs, ys = [], []
        with open(path) as f:
            for r in csv.DictReader(f):
                t = int(float(r["t"])); nid = int(r["node"])
                x, y = float(r["x"]), float(r["y"])
                rows.setdefault(t, {})[nid] = (x, y)
                xs.append(x); ys.append(y)
        self.tmax = max(rows)
        self.rows = rows
        self.ids = sorted({nid for d in rows.values() for nid in d})
        self.n = n; self.area = area
        self.sx = area / (max(xs) - min(xs) + 1e-9); self.sy = area / (max(ys) - min(ys) + 1e-9)
        self.x0, self.y0 = min(xs), min(ys)
        self.t = 0
        self.pos = np.zeros((n, 2))
        self.last = {}
        self.step(0)

    def step(self, dt=1.0):
        self.t += dt
        tt = int(self.t) % (self.tmax + 1)
        d = self.rows.get(tt, {})
        for k in range(self.n):
            nid = self.ids[k % len(self.ids)]
            if nid in d:
                self.last[nid] = d[nid]
            x, y = self.last.get(nid, (self.x0, self.y0))
            self.pos[k] = ((x - self.x0) * self.sx, (y - self.y0) * self.sy)
        return self.pos


def contacts(pos, r):
    """Return list of (i, j) with i < j and dist <= r, via grid hashing."""
    n = len(pos)
    cell = max(r, 1e-6)
    keys = np.floor(pos / cell).astype(np.int64)
    grid = {}
    for i, (cx, cy) in enumerate(keys):
        grid.setdefault((cx, cy), []).append(i)
    out = []
    r2 = r * r
    for (cx, cy), members in grid.items():
        cand = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                cand.extend(grid.get((cx + dx, cy + dy), ()))
        for i in members:
            pi = pos[i]
            for j in cand:
                if j > i:
                    d = pi - pos[j]
                    if d[0] * d[0] + d[1] * d[1] <= r2:
                        out.append((i, j))
    return out


def range_for_degree(n, area, degree):
    """Communication range giving average `degree` neighbours in a
    uniformly populated square area."""
    return float(np.sqrt(degree * area * area / (np.pi * (n - 1))))


class JavaWaypoint:
    """Replicates MovmentGenerator + PathBuilder from the ver7 Java engine:
    each node gets horizon/100 waypoints at uniformly random times; per-leg
    speed = speed * min(U(0,1), 0.7); node is stationary until its first
    waypoint time, then heads for the current waypoint at that leg's speed,
    switching to the next waypoint when t >= next.time - 2."""

    def __init__(self, n, area, speed, rng, horizon):
        self.n = n; self.area = area; self.rng = rng
        self.pos = rng.uniform(0, area, size=(n, 2))
        k = max(1, int(horizon / 100) - 1)
        self.wp_t = np.sort(rng.integers(0, horizon + 1, size=(n, k)), axis=1).astype(float)
        self.wp_p = rng.uniform(0, area, size=(n, k, 2))
        self.wp_v = speed * np.minimum(rng.uniform(0, 1, size=(n, k)), 0.7)
        self.idx = np.zeros(n, dtype=int)
        self.k = k
        self.t = 0.0

    def step(self, dt=1.0):
        self.t += dt
        t = self.t
        first = self.wp_t[:, 0]
        moving = t >= first - 1
        idx = self.idx
        dest = self.wp_p[np.arange(self.n), idx]
        v = self.wp_v[np.arange(self.n), idx]
        d = dest - self.pos
        dist = np.linalg.norm(d, axis=1)
        step = np.minimum(v * dt, dist)
        ok = moving & (dist > 1e-9)
        self.pos[ok] += (d[ok] / dist[ok, None]) * step[ok, None]
        # advance waypoint index
        nxt = np.minimum(idx + 1, self.k - 1)
        nxt_t = self.wp_t[np.arange(self.n), nxt]
        adv = (idx < self.k - 1) & (t >= nxt_t - 2)
        self.idx[adv] = nxt[adv]
        return self.pos
