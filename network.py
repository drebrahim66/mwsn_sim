"""Physical nodes, replicas, and the reporting behaviour of attackers.

Each physical node has a unique `pid`. Legitimate nodes have id == pid.
Replicas of victim v have id == v.id but a different pid.

Attack classes (see evaluation section):
  naive        : report s + 3*eps (out of acceptable range)
  sem_suppress : report the value the field had `delay` seconds ago at
                 the replica's own position (stale value; hides a trend)
  sem_displace : report the field value at a position `disp` metres away
  Each may be combined with false_claim=True.
"""
import numpy as np


class Node:
    __slots__ = ("pid", "id", "is_replica", "malicious", "victim_pid")

    def __init__(self, pid, nid, is_replica=False, malicious=False, victim_pid=None):
        self.pid = pid
        self.id = nid
        self.is_replica = is_replica
        self.malicious = malicious
        self.victim_pid = victim_pid


class Network:
    def __init__(self, cfg, rng, field, mobility):
        self.cfg = cfg
        self.rng = rng
        self.field = field
        self.mob = mobility
        self.nodes = []
        n_legit = cfg["n_nodes"]
        for k in range(n_legit):
            self.nodes.append(Node(k, k))
        # victims and replicas
        victims = list(rng.choice(n_legit, size=cfg["n_victims"], replace=False))
        self.victims = victims
        pid = n_legit
        self.replicas = []
        for v in victims:
            r = cfg["replicas_per_victim"]
            n_mal = int(round(cfg["malicious_fraction"] * r))
            flags = [True] * n_mal + [False] * (r - n_mal)
            rng.shuffle(flags)
            for m in flags:
                nd = Node(pid, v, is_replica=True, malicious=m, victim_pid=v)
                self.nodes.append(nd); self.replicas.append(nd); pid += 1
        self.n_total = pid
        self.legit = [nd for nd in self.nodes if not nd.is_replica]
        self.malicious = [nd for nd in self.replicas if nd.malicious]
        self.benign_reps = [nd for nd in self.replicas if not nd.malicious]
        self.sigma_n = cfg["sigma_n"]
        self.eps = cfg["eps"]
        self.attack = cfg["attack"]
        self.false_claim = cfg.get("false_claim", False)
        self.delay = cfg.get("suppress_delay", 120.0)
        self.disp = cfg.get("displace_dist", 150.0)
        self._disp_dir = rng.uniform(0, 2 * np.pi, size=self.n_total)

    @property
    def pos(self):
        return self.mob.pos

    def true_value(self, pid, t):
        return float(self.field.value(self.pos[pid], t))

    def report(self, pid, t):
        """Sensed value the physical node `pid` reports at time t."""
        nd = self.nodes[pid]
        p = self.pos[pid]
        s = float(self.field.value(p, t)) + self.rng.normal(0, self.sigma_n)
        if not nd.malicious:
            return s
        a = self.attack
        if a == "naive":
            return s + 3.0 * self.eps
        if a == "sem_suppress":
            return float(self.field.value(p, max(0.0, t - self.delay))) + self.rng.normal(0, self.sigma_n)
        if a == "sem_displace":
            th = self._disp_dir[pid]
            q = p + self.disp * np.array([np.cos(th), np.sin(th)])
            q = np.clip(q, 0, self.cfg["area"])
            return float(self.field.value(q, t)) + self.rng.normal(0, self.sigma_n)
        if a == "none":
            return s
        raise ValueError(a)
