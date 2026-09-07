"""Single-run simulation and metrics.

run(cfg, seed) -> dict of metrics:
    T_q   steps until every legitimate node refuses every malicious replica
          (None if not reached within horizon)
    T_d   mean over malicious replicas of first step some legit node refuses it
    P_d   fraction of malicious replicas refused by >=1 legit node at horizon
    FNR   1 - P_d
    FPR   fraction of (legit observer, legit target) pairs refused at horizon
    S     fraction of legit nodes refused by <10% of legit nodes at horizon
    S_b   same for benign replicas
    M     mean memory bytes per node
    C     mean messages per node
    E_radio, E_cpu, E   energy per node (J)
"""
import numpy as np
from .field import SmoothField, EventField
from .mobility import RandomWaypoint, TraceMobility, JavaWaypoint, contacts, range_for_degree
from .network import Network
from .protocols import PROTOCOLS

DEFAULTS = dict(
    n_nodes=500, area=500.0, speed=3.0, degree=6, horizon=2000,
    n_victims=1, replicas_per_victim=4, malicious_fraction=0.5,
    field="event", attack="sem_suppress", false_claim=False,
    false_claims_per_encounter=2, suppress_delay=120.0, displace_dist=150.0,
    sigma_n_frac=0.02, eps_frac=0.32,
    # proposed
    delta=120.0, m_min=4, sigma_p=50.0, sigma_t=60.0, kappa=2.5,
    alpha=0.3, tau0=0.5, tau_q=0.2, sig_window=5, m_max=20,
    # nbhd trust
    nbhd_d=10, nbhd_min_ids=6, nbhd_min_obs=5, nbhd_ath_floor=0.5,
    # energy (first-order radio model + MSP430)
    E_elec=50e-9, eps_amp=100e-12, msg_bits=24 * 8, cpu_J_per_op=1.35e-9,
    protocol="PROPOSED", mobility="rwp", trace_path=None, comm_range=None,
    exclude_victims_as_observers=True, check_every=1,
    event_grow_time=300.0, field_period=600.0,
)


def make_cfg(**over):
    cfg = dict(DEFAULTS); cfg.update(over)
    return cfg


def run(cfg, seed=0, progress=False):
    cfg = dict(cfg)
    rng = np.random.default_rng(seed)
    area = cfg["area"]
    per = cfg.get("field_period", 600.0)
    if cfg["field"] == "event":
        field = EventField(area, rng, horizon=cfg["horizon"], grow_time=cfg["event_grow_time"], period=per)
    else:
        field = SmoothField(area, rng, period=per)
    cfg["sigma_n"] = cfg["sigma_n_frac"] * field.dynamic_range
    cfg["eps"] = cfg["eps_frac"] * field.dynamic_range
    cfg["sigma_min"] = cfg["sigma_n"]

    n_total = cfg["n_nodes"] + cfg["n_victims"] * cfg["replicas_per_victim"]
    if cfg["mobility"] == "trace" and cfg["trace_path"]:
        mob = TraceMobility(n_total, area, cfg["trace_path"], rng)
    elif cfg["mobility"] == "java":
        mob = JavaWaypoint(n_total, area, cfg["speed"], rng, cfg["horizon"])
    else:
        mob = RandomWaypoint(n_total, area, cfg["speed"], rng)
    net = Network(cfg, rng, field, mob)
    proto = PROTOCOLS[cfg["protocol"]](net, cfg, rng)
    r = cfg["comm_range"] if cfg.get("comm_range") else range_for_degree(cfg["n_nodes"], area, cfg["degree"])

    legit = [nd.pid for nd in net.legit]
    observers = [p for p in legit if p not in set(net.victims)] if cfg.get("exclude_victims_as_observers") else legit
    mal = [nd.pid for nd in net.malicious]
    ben = [nd.pid for nd in net.benign_reps]
    first_det = {m: None for m in mal}
    T_q = None; T_q_id = None
    same_id = {m: [q.pid for q in net.nodes if q.id == net.nodes[m].id] for m in mal}
    active = set()
    H = cfg["horizon"]
    check_every = cfg.get("check_every", 1)

    for t in range(1, H + 1):
        pos = mob.step(1.0)
        cur = set(contacts(pos, r))
        new = cur - active
        active = cur
        for (i, j) in new:
            proto.on_contact(i, j, t)
        if t % check_every == 0 or t == H:
            all_q = True
            for m in mal:
                cnt = 0
                for i in observers:
                    if proto.refuses(i, m):
                        cnt += 1
                if cnt and first_det[m] is None:
                    first_det[m] = t
                if cnt < len(observers):
                    all_q = False
            if all_q and T_q is None:
                T_q = t
            if T_q_id is None:
                ok = all(all(any(proto.refuses(i, q) for q in same_id[m]) for i in observers) for m in mal)
                if ok:
                    T_q_id = t
            if T_q is not None and T_q_id is not None and not cfg.get("run_full_horizon", True):
                break
        if progress and t % 200 == 0:
            print(f"  t={t} msgs={proto.messages}")

    # final metrics
    nL = len(legit)
    det = [m for m in mal if first_det[m] is not None]
    P_d = len(det) / max(1, len(mal))
    T_d = float(np.mean([first_det[m] for m in det])) if det else None

    def refused_frac(target):
        return sum(1 for i in legit if i != target and proto.refuses(i, target)) / max(1, nL - 1)

    fp_pairs = 0
    surv = 0
    for tgt in legit:
        f = refused_frac(tgt)
        fp_pairs += f
        if f < 0.10:
            surv += 1
    FPR = fp_pairs / nL
    S = surv / nL
    S_b = (sum(1 for b in ben if refused_frac(b) < 0.10) / len(ben)) if ben else float("nan")

    # quarantine coverage: fraction of observers refusing each malicious replica at horizon
    cov = float(np.mean([sum(1 for i in observers if proto.refuses(i, m)) / max(1, len(observers)) for m in mal])) if mal else float("nan")
    M = float(np.mean([proto.memory_bytes(i) for i in legit]))
    C = proto.messages / n_total
    d_avg = r * 0.66
    E_radio = C * cfg["msg_bits"] * (2 * cfg["E_elec"] + cfg["eps_amp"] * d_avg ** 2)
    E_cpu = proto.cpu_ops / n_total * cfg["cpu_J_per_op"]
    return dict(protocol=cfg["protocol"], attack=cfg["attack"], seed=seed,
                T_q=T_q, T_q_id=T_q_id, T_d=T_d, COV=cov, P_d=P_d, FNR=1 - P_d, FPR=FPR, S=S, S_b=S_b,
                M=M, C=C, E_radio=E_radio, E_cpu=E_cpu, E=E_radio + E_cpu,
                rho=_rho(cfg, field))


def _rho(cfg, field):
    """Normalised trend rho = |beta| Delta / (2 A_th) with A_th ~ noise-floor
    tolerance; reported for the event field as a diagnostic."""
    if not hasattr(field, "growth_rate"):
        return None
    beta = field.growth_rate()
    ath = cfg["sigma_n"]  # floor used by NBHD_TRUST when neighbourhood is flat
    return beta * cfg["delta"] / (2 * ath)


if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default="PROPOSED")
    ap.add_argument("--attack", default="sem_suppress")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--horizon", type=int, default=800)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--false_claim", action="store_true")
    a = ap.parse_args()
    cfg = make_cfg(protocol=a.protocol, attack=a.attack, n_nodes=a.n, horizon=a.horizon,
                   false_claim=a.false_claim)
    print(json.dumps(run(cfg, a.seed, progress=True), indent=1))
