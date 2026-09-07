"""Replica detection / quarantining protocols.

All protocols expose:
    on_contact(i, j, t)         -- run the encounter between physical nodes i, j
    refuses(i, pid)             -- does node i refuse to communicate with pid?
    memory_bytes(i)             -- bytes of protocol state at node i
    messages                    -- running message count (all nodes)
    cpu_ops                     -- running instruction-count estimate

Modelling notes
---------------
* Signatures. ver7 established that encounter-history signatures uniquely
  distinguish physical nodes sharing an id. Rather than re-simulate the
  signature exchange, SIG_RANGE and PROPOSED key their per-node state by
  the physical id `pid`, which is exactly the information a verified
  signature provides. Message and memory costs of the signature exchange
  are still charged (SIGN/TS round trips, w ids per signature).
* Nonces. Node i stores, for each id, the pid of the physical node it
  last gave a nonce to. A later node with the same id but a different
  pid cannot present that nonce: this is the nonce-mismatch event.
* NBHD_TRUST is a *strengthened* re-implementation of Teng et al. (2026):
  only the sensed-data check (their Eq. 6) feeds trust, so the data
  evidence is not diluted by five other factors; trust is per id, and a
  suspect id is quarantined outright (we grant them the "same ID"
  confirmation for free).
"""
import math
from collections import deque, defaultdict
import numpy as np

BYTES_ID = 2; BYTES_TS = 4; BYTES_VAL = 2; BYTES_POS = 2; BYTES_TAU = 1; BYTES_NONCE = 4


class Protocol:
    name = "base"

    def __init__(self, net, cfg, rng):
        self.net = net; self.cfg = cfg; self.rng = rng
        n = net.n_total
        self.n = n
        self.messages = 0
        self.cpu_ops = 0
        self.horizon = cfg["horizon"]

    def on_contact(self, i, j, t):
        raise NotImplementedError

    def refuses(self, i, pid):
        raise NotImplementedError

    def memory_bytes(self, i):
        return 0

    # helpers -----------------------------------------------------------
    def _node(self, pid):
        return self.net.nodes[pid]

    def _fake_claims(self, j, k):
        """Malicious node j fabricates k direct claims against random legit nodes."""
        if not (self.cfg.get("false_claim") and self._node(j).malicious):
            return []
        legit = self.net.legit
        idx = self.rng.choice(len(legit), size=min(k, len(legit)), replace=False)
        return [legit[x].pid for x in idx]


# ======================================================================
class XED(Protocol):
    """Nonce exchange on encounter; mismatch quarantines the *id*.
    Quarantine lists are gossiped on encounter (needed to reach full
    quarantining in finite time; XED alone has no referral)."""
    name = "XED"

    def __init__(self, net, cfg, rng):
        super().__init__(net, cfg, rng)
        self.nonce = [dict() for _ in range(self.n)]     # id -> pid
        self.ql = [set() for _ in range(self.n)]         # ids

    def refuses(self, i, pid):
        return self._node(pid).id in self.ql[i]

    def _side(self, i, j, t):
        nj = self._node(j)
        if nj.id in self.ql[i]:
            return
        prev = self.nonce[i].get(nj.id)
        if prev is not None and prev != j:
            self.ql[i].add(nj.id)
        self.nonce[i][nj.id] = j

    def on_contact(self, i, j, t):
        if self.refuses(i, j) or self.refuses(j, i):
            return
        self._side(i, j, t); self._side(j, i, t)
        self.messages += 2
        # gossip quarantine lists
        self.ql[i] |= self.ql[j]; self.ql[j] |= self.ql[i]
        self.messages += 2

    def memory_bytes(self, i):
        return len(self.nonce[i]) * (BYTES_ID + BYTES_NONCE) + len(self.ql[i]) * BYTES_ID


# ======================================================================
class SigRange(Protocol):
    """Alrashed et al. 2024 (ver7): signatures, nonces, acceptable-range
    test, count-based claims with cThresh doubling."""
    name = "SIG_RANGE"

    def __init__(self, net, cfg, rng):
        super().__init__(net, cfg, rng)
        n = self.n
        self.nonce = [dict() for _ in range(n)]              # id -> (pid, t_issued)
        self.rl = [defaultdict(set) for _ in range(n)]       # pid -> {claimant pid}
        self.ql = [dict() for _ in range(n)]                 # pid -> 'direct'|'referred'
        self.cth = [1.0] * n
        self.eps = cfg["eps"]
        self.w = cfg.get("sig_window", 5)
        self.gossip = cfg.get("nonce_gossip", False)
        self.claims_any = cfg.get("claims_from_any", False)
        self.strict = cfg.get("strict_threshold", False)

    def refuses(self, i, pid):
        return pid in self.ql[i]

    def _verify_legit(self, i, j, t):
        si = self.net.true_value(i, t) + self.rng.normal(0, self.net.sigma_n)
        sj = self.net.report(j, t)
        self.net.observe(i, t, sj); self.net.observe(j, t, si)
        self.messages += 2
        self.cpu_ops += 5
        if abs(si - sj) > self.eps:
            self.ql[i][j] = "direct"
            self._maybe_double(i)

    def _maybe_double(self, i):
        nd = sum(1 for v in self.ql[i].values() if v == "direct")
        if nd > self.cth[i]:
            self.cth[i] *= 2
            for k in [k for k, v in self.ql[i].items() if v == "referred"]:
                del self.ql[i][k]

    def _check_claims(self, i):
        for k, cl in list(self.rl[i].items()):
            n = len(cl)
            hit = n > self.cth[i] if self.strict else n >= self.cth[i]
            if hit and k not in self.ql[i]:
                self.ql[i][k] = "referred"

    def _direct_list(self, j):
        if self.claims_any:
            lst = list(self.ql[j].keys())
        else:
            lst = [k for k, v in self.ql[j].items() if v == "direct"]
        lst += self._fake_claims(j, self.cfg.get("false_claims_per_encounter", 2))
        return lst

    def _gossip_nonces(self, i, j):
        """Adopt j's newer nonce expectations (Java updateReceivedCode)."""
        for nid, (holder, ts) in self.nonce[j].items():
            cur = self.nonce[i].get(nid)
            if cur is None or ts > cur[1]:
                self.nonce[i][nid] = (holder, ts)

    def _side(self, i, j, t):
        nj = self._node(j)
        if j in self.rl[i]:                       # known replica
            self.messages += 4                    # SIGN + TS interrogation
            self._check_claims(i)
            self._verify_legit(i, j, t)
            return
        prev = self.nonce[i].get(nj.id)
        ok = prev is None or prev[0] == j
        self.nonce[i][nj.id] = (j, t)
        if ok:                                    # legitimate path
            self.messages += 2                    # QLIST exchange
            if self.gossip:
                self._gossip_nonces(i, j); self.messages += 1
            for k in self._direct_list(j):
                self.rl[i][k].add(j)
            self._check_claims(i)
        else:                                     # wrong nonce: new replica
            self.messages += 2                    # SIGN
            self.rl[i][j]                          # create entry
            self._verify_legit(i, j, t)

    def on_contact(self, i, j, t):
        if self.refuses(i, j) or self.refuses(j, i):
            return
        self.messages += 2                        # nonce exchange
        self._side(i, j, t); self._side(j, i, t)

    def memory_bytes(self, i):
        m = len(self.nonce[i]) * (BYTES_ID + BYTES_NONCE)
        m += sum(BYTES_ID + self.w * BYTES_ID + len(c) * BYTES_ID for c in self.rl[i].values())
        m += len(self.ql[i]) * (BYTES_ID + self.w * BYTES_ID + 1)
        return m


# ======================================================================
class NbhdTrust(Protocol):
    """Strengthened re-implementation of Teng et al. 2026's data check.

    Per-id Beta trust from (normal, abnormal) interaction counts, where
    an interaction is abnormal iff |x_j - mean| > A_th with mean and A_th
    from their Eq. (6) over the receiver's recent context. Trust is
    decayed over a sliding window of d interactions. A 2-cluster split
    of the trust values a node holds gives threshold K; ids with trust
    <= min(K, 0.5) are suspects and are quarantined. Suspect lists are
    gossiped on encounter."""
    name = "NBHD_TRUST"

    def __init__(self, net, cfg, rng):
        super().__init__(net, cfg, rng)
        n = self.n
        self.ctx = [deque() for _ in range(n)]          # (t, s)
        self.ab = [defaultdict(lambda: [0, 0]) for _ in range(n)]     # id -> [alpha, beta]
        self.win = [defaultdict(lambda: deque(maxlen=cfg.get("nbhd_d", 10))) for _ in range(n)]
        self.trust = [dict() for _ in range(n)]         # id -> value
        self.ql = [set() for _ in range(n)]             # ids
        self.delta = cfg["delta"]
        self.theta = 150.0
        self.kappa = 0.1
        self.min_ids = cfg.get("nbhd_min_ids", 6)

    def refuses(self, i, pid):
        return self._node(pid).id in self.ql[i]

    def _prune(self, i, t):
        c = self.ctx[i]
        while c and t - c[0][0] > self.delta:
            c.popleft()

    def _nbhd_ok(self, i, t, si, sj):
        self._prune(i, t)
        vals = [s for (_, s) in self.ctx[i]]
        m = len(vals)
        self.cpu_ops += 3 * m + 10
        if m == 0:
            return True
        mean = (si + sum(vals)) / (m + 1)
        ath = sum(abs(v - mean) for v in vals) / m
        # floor: their MAD, but never below the calibrated honest spatial
        # variation at contact distance (cfg nbhd_ath_floor * eps); without
        # this the baseline false-quarantines honest nodes near gradients.
        ath = max(ath, self.cfg.get("nbhd_ath_floor", 0.5) * self.net.eps)
        return abs(sj - mean) <= ath

    def _update_trust(self, i, nid, ok):
        ab = self.ab[i][nid]
        if ok: ab[0] += 1
        else:  ab[1] += 1
        a, b = ab
        omega = self.theta / (a + b) if (a + b) else 1.0
        rp = a / (a + b) if (a + b) else 1.0
        dtb = (omega * a + 1) / (omega * a + omega * b + 2) * rp
        w = self.win[i][nid]; w.append(dtb)
        d = len(w)
        ws = np.array([math.exp(-self.kappa * (d - h)) for h in range(1, d + 1)])
        self.trust[i][nid] = float((ws * np.array(w)).sum() / ws.sum())
        self.cpu_ops += 6 * d + 12

    def _threshold(self, i):
        vals = np.array(sorted(self.trust[i].values()))
        if len(vals) < self.min_ids:
            return None
        # 1-D two-cluster split maximising between-cluster separation
        best, bk = -1.0, None
        for k in range(1, len(vals)):
            lo, hi = vals[:k], vals[k:]
            sep = hi.mean() - lo.mean()
            if sep > best:
                best, bk = sep, k
        K = 0.5 * (vals[bk - 1] + vals[bk])
        self.cpu_ops += 4 * len(vals)
        # require a real gap between the clusters; a healthy network has none
        if vals[bk] - vals[bk - 1] < self.cfg.get("nbhd_min_gap", 0.2):
            return None
        return min(K, 0.5)

    def _side(self, i, j, t):
        nj = self._node(j)
        si = self.net.true_value(i, t) + self.rng.normal(0, self.net.sigma_n)
        sj = self.net.report(j, t)
        self.net.observe(i, t, sj); self.net.observe(j, t, si)
        ok = self._nbhd_ok(i, t, si, sj)
        self._update_trust(i, nj.id, ok)
        self.ctx[i].append((t, sj))
        K = self._threshold(i)
        a, b = self.ab[i][nj.id]
        if K is not None and a + b >= self.cfg.get("nbhd_min_obs", 3) and self.trust[i][nj.id] <= K:
            self.ql[i].add(nj.id)

    def on_contact(self, i, j, t):
        if self.refuses(i, j) or self.refuses(j, i):
            return
        self.messages += 2                        # data exchange
        self._side(i, j, t); self._side(j, i, t)
        # Teng et al. decide locally; no suspect-list gossip. False claims can
        # only act through the indirect-trust path, which we approximate by
        # letting a malicious node's accusation count as one abnormal outcome.
        for tgt in self._fake_claims(j, self.cfg.get("false_claims_per_encounter", 2)):
            self._update_trust(i, self._node(tgt).id, False)
        for tgt in self._fake_claims(i, self.cfg.get("false_claims_per_encounter", 2)):
            self._update_trust(j, self._node(tgt).id, False)

    def memory_bytes(self, i):
        return (len(self.ctx[i]) * (BYTES_TS + BYTES_VAL)
                + len(self.ab[i]) * (BYTES_ID + 2 * 2 + self.cfg.get("nbhd_d", 10) * BYTES_TAU)
                + len(self.ql[i]) * BYTES_ID)


# ======================================================================
class Proposed(Protocol):
    """Semantic, per-signature trust with selective quarantining."""
    name = "PROPOSED"

    def __init__(self, net, cfg, rng):
        super().__init__(net, cfg, rng)
        n = self.n
        self.nonce = [dict() for _ in range(n)]              # id -> (pid, t_issued)
        self.gossip = cfg.get("nonce_gossip", False)
        self.es = [deque() for _ in range(n)]                # (t, s, p)
        self.tr = [dict() for _ in range(n)]                 # pid -> tau
        self.rl = [defaultdict(dict) for _ in range(n)]      # pid -> {claimant: tau}
        self.ql = [dict() for _ in range(n)]                 # pid -> type
        self.cth = [1.0] * n
        self.delta = cfg["delta"]; self.m_min = cfg["m_min"]
        self.sp = cfg["sigma_p"]; self.st = cfg["sigma_t"]
        self.smin = cfg["sigma_min"]; self.kap = cfg["kappa"]
        self.alpha = cfg["alpha"]; self.tau0 = cfg["tau0"]; self.tauq = cfg["tau_q"]
        self.eps = cfg["eps"]; self.w = cfg.get("sig_window", 5)
        self.m_max = cfg.get("m_max", 20)

    def refuses(self, i, pid):
        return pid in self.ql[i]

    # semantic consistency ------------------------------------------------
    def _context(self, i, t):
        es = self.es[i]
        while es and t - es[0][0] > self.delta:
            es.popleft()
        if len(es) > self.m_max:
            return list(es)[-self.m_max:]
        return list(es)

    def sem_consistent(self, i, t, si, sj, pj):
        C = self._context(i, t)
        m = len(C)
        self.cpu_ops += 8 * m + 20
        if m < self.m_min:
            return abs(si - sj) <= self.eps
        ts = np.array([c[0] for c in C]); ss = np.array([c[1] for c in C])
        ps = np.array([c[2] for c in C])
        d2 = ((ps - pj) ** 2).sum(1)
        w = np.exp(-d2 / (2 * self.sp ** 2) - (t - ts) ** 2 / (2 * self.st ** 2))
        W = w.sum()
        if W < 1e-9:
            return abs(si - sj) <= self.eps
        shat = (w * ss).sum() / W
        tbar = (w * ts).sum() / W
        var_t = (w * (ts - tbar) ** 2).sum() / W
        beta = (w * (ts - tbar) * (ss - shat)).sum() / W / var_t if var_t > 1e-9 else 0.0
        shat2 = shat + beta * (t - tbar)
        sig2 = (w * (ss - shat) ** 2).sum() / W
        sig = math.sqrt(max(sig2, self.smin ** 2))
        return abs(sj - shat2) / sig <= self.kap

    # trust ------------------------------------------------------------------
    def _update_trust(self, i, j, ok):
        tau = self.tr[i].get(j, self.tau0)
        tau = (1 - self.alpha) * tau + self.alpha * (1.0 if ok else 0.0)
        self.tr[i][j] = tau
        return tau

    def _verify_legit(self, i, j, t):
        si = self.net.true_value(i, t) + self.rng.normal(0, self.net.sigma_n)
        sj = self.net.report(j, t)
        self.net.observe(i, t, sj); self.net.observe(j, t, si)
        pj = self.net.pos[j].copy()
        self.messages += 2
        ok = self.sem_consistent(i, t, si, sj, pj)
        tau = self._update_trust(i, j, ok)
        self.es[i].append((t, sj, pj))
        if tau < self.tauq and self.ql[i].get(j) != "direct":
            self.ql[i][j] = "direct"
            self._maybe_double(i)

    def _maybe_double(self, i):
        mass = sum(1 - self.tr[i].get(k, self.tau0) for k, v in self.ql[i].items() if v == "direct")
        if mass > self.cth[i]:
            self.cth[i] *= 2
            for k in [k for k, v in self.ql[i].items() if v == "referred"]:
                del self.ql[i][k]

    def _check_claims(self, i):
        for k, cl in list(self.rl[i].items()):
            mass = sum(cl.values())
            if mass >= self.cth[i] and k not in self.ql[i]:
                self.ql[i][k] = "referred"

    def _direct_list(self, j):
        lst = [k for k, v in self.ql[j].items() if v == "direct"]
        lst += self._fake_claims(j, self.cfg.get("false_claims_per_encounter", 2))
        return lst

    def _side(self, i, j, t):
        nj = self._node(j)
        if j in self.rl[i]:                       # known replica id/sig
            self.messages += 4
            self._check_claims(i)
            self._verify_legit(i, j, t)
            return
        prev = self.nonce[i].get(nj.id)
        ok = prev is None or prev[0] == j
        self.nonce[i][nj.id] = (j, t)
        if ok:                                    # legitimate path
            self.messages += 2                    # SIGN
            if self.gossip:
                for nid, (h, ts) in self.nonce[j].items():
                    c = self.nonce[i].get(nid)
                    if c is None or ts > c[1]:
                        self.nonce[i][nid] = (h, ts)
                self.messages += 1
            self._verify_legit(i, j, t)           # legit nodes build trust too
            self.messages += 2                    # QLIST
            tau_j = self.tr[i].get(j, self.tau0)
            if tau_j >= self.tauq:                # quarantined claimants weigh 0
                for k in self._direct_list(j):
                    self.rl[i][k][j] = tau_j
            self._check_claims(i)
        else:                                     # wrong nonce: new replica
            self.messages += 2
            self.rl[i][j]
            self._verify_legit(i, j, t)

    def on_contact(self, i, j, t):
        if self.refuses(i, j) or self.refuses(j, i):
            return
        self.messages += 2
        self._side(i, j, t); self._side(j, i, t)

    def memory_bytes(self, i):
        m = len(self.nonce[i]) * (BYTES_ID + BYTES_NONCE)
        m += len(self.es[i]) * (BYTES_ID + BYTES_TS + BYTES_VAL + BYTES_POS)
        m += len(self.tr[i]) * (BYTES_ID + BYTES_TAU)
        m += sum(BYTES_ID + self.w * BYTES_ID + len(c) * (BYTES_ID + BYTES_TAU) for c in self.rl[i].values())
        m += len(self.ql[i]) * (BYTES_ID + self.w * BYTES_ID + 1)
        return m


PROTOCOLS = {"XED": XED, "SIG_RANGE": SigRange, "NBHD_TRUST": NbhdTrust, "PROPOSED": Proposed}
