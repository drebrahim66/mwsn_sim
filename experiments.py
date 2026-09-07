"""Experiment runner E1-E9 (see semantic_replica_evaluation.tex).

Usage:
    python3 experiments.py --runs 500 --cores 12 [--only E1,E2] [--quick]

--quick uses 8 runs and a 1000-step horizon for a smoke test.
Output: results_<timestamp>/E*.csv with one row per (protocol, parameter value):
    experiment, protocol, param, value, n_runs, then mean/std/ci95 of each metric.
Configuration mirrors the ver7 Java engine (Driver.java) as calibrated:
200 nodes, 200x200 m, radius 32 m, Java-style waypoints, nonce gossip on.
"""
import argparse, csv, os, time
from datetime import datetime
import numpy as np
import multiprocessing as mp
from mwsnsim.sim import make_cfg, run

PROTOS = ["XED", "SIG_RANGE", "NBHD_TRUST", "PROPOSED"]
METRICS = ["P_d", "FNR", "T_d", "T_q", "COV", "FPR", "S", "S_b", "M", "C", "E_radio", "E_cpu", "E"]

BASE = dict(n_nodes=200, area=200.0, comm_range=32.0, speed=3.0, horizon=2000,
            n_victims=1, replicas_per_victim=4, malicious_fraction=0.5,
            mobility="java", exclude_victims_as_observers=True, nonce_gossip=True,
            field="event", event_grow_time=300.0, attack="sem_suppress")


def _job(args):
    cfg, seed = args
    return run(cfg, seed)


def sweep(pool, name, param, values, runs, protos=PROTOS, base=None, cfg_fn=None):
    base = dict(BASE if base is None else base)
    rows = []
    for v in values:
        for p in protos:
            over = dict(base); over["protocol"] = p
            if cfg_fn: over = cfg_fn(over, v)
            else: over[param] = v
            cfg = make_cfg(**over)
            seed0 = (abs(hash((name, param, str(v)))) % 10000) * 1000
            t0 = time.time()
            res = pool.map(_job, [(cfg, seed0 + s) for s in range(runs)])
            row = dict(experiment=name, protocol=p, param=param, value=v, n_runs=runs)
            for m in METRICS:
                xs = np.array([r[m] for r in res if r[m] is not None and not (isinstance(r[m], float) and np.isnan(r[m]))], float)
                if m == "T_q":
                    row["T_q_completed"] = round(len(xs) / runs, 3)
                row[f"{m}_mean"] = round(float(xs.mean()), 4) if len(xs) else ""
                row[f"{m}_std"] = round(float(xs.std()), 4) if len(xs) > 1 else ""
                row[f"{m}_ci95"] = round(float(1.96 * xs.std() / np.sqrt(len(xs))), 4) if len(xs) > 1 else ""
            rows.append(row)
            print(f"  {name} {param}={v} {p:10s} S={row['S_mean']} S_b={row['S_b_mean']} P_d={row['P_d_mean']} "
                  f"COV={row['COV_mean']} T_q={row['T_q_mean']} FPR={row['FPR_mean']}  [{time.time()-t0:.0f}s]", flush=True)
    return rows


def save(rows, path):
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print(f"  saved {path}")


# ---------------------------------------------------------------- experiments
def E1(pool, runs):   # legitimate-node survival vs replicas per victim
    return sweep(pool, "E1", "replicas_per_victim", [1, 2, 3, 4, 5, 6], runs)

def E2(pool, runs):   # suppression detection vs trend (grow_time controls rho)
    def f(o, v): o["event_grow_time"] = v; return o
    return sweep(pool, "E2", "event_grow_time", [1200, 600, 300, 150, 75], runs, cfg_fn=f)

def E3(pool, runs):   # displacement detection vs distance
    def f(o, v): o["attack"] = "sem_displace"; o["displace_dist"] = v; return o
    return sweep(pool, "E3", "displace_dist", [50, 100, 150, 200, 300], runs, cfg_fn=f)

def E4(pool, runs):   # naive attack parity
    def f(o, v): o["attack"] = "naive"; o["replicas_per_victim"] = v; return o
    return sweep(pool, "E4", "replicas_per_victim", [1, 2, 3, 4, 5, 6], runs, cfg_fn=f)

def E5(pool, runs):   # false claims vs number of Byzantine replicas
    def f(o, v):
        o["false_claim"] = True; o["replicas_per_victim"] = v; o["malicious_fraction"] = 1.0; return o
    return sweep(pool, "E5", "byzantine_replicas", [1, 2, 4, 8, 16], runs,
                 protos=["SIG_RANGE", "PROPOSED"], cfg_fn=f)

def E6(pool, runs):   # scaling: nodes, area, speed (proposed vs SIG_RANGE)
    rows = []
    def fn(o, v): o["n_nodes"] = v; return o
    rows += sweep(pool, "E6a", "n_nodes", [100, 150, 200, 250, 300], runs, protos=["SIG_RANGE", "PROPOSED"], cfg_fn=fn)
    def fa(o, v): o["area"] = float(v); return o
    rows += sweep(pool, "E6b", "area", [200, 250, 300, 350, 400], runs, protos=["SIG_RANGE", "PROPOSED"], cfg_fn=fa)
    def fs(o, v): o["speed"] = float(v); return o
    rows += sweep(pool, "E6c", "speed", [2, 4, 6, 8, 10], runs, protos=["SIG_RANGE", "PROPOSED"], cfg_fn=fs)
    return rows

def E7(pool, runs):   # energy decomposition at default
    return sweep(pool, "E7", "default", ["default"], runs, cfg_fn=lambda o, v: o)

def E8(pool, runs, trace):   # trace-driven mobility: repeat E1/E2/E4
    if not trace: print("  E8 skipped: no --trace given"); return []
    b = dict(BASE); b["mobility"] = "trace"; b["trace_path"] = trace
    rows = sweep(pool, "E8-E1", "replicas_per_victim", [1, 2, 4, 6], runs, base=b)
    def f(o, v): o["event_grow_time"] = v; return o
    rows += sweep(pool, "E8-E2", "event_grow_time", [600, 300, 150], runs, base=b, cfg_fn=f)
    def g(o, v): o["attack"] = "naive"; o["replicas_per_victim"] = v; return o
    rows += sweep(pool, "E8-E4", "replicas_per_victim", [1, 2, 4, 6], runs, base=b, cfg_fn=g)
    return rows

def E2S(pool, runs):  # suppression vs trend with diurnal-period background (rho controlled)
    def f(o, v): o["event_grow_time"] = v; o["field_period"] = 86400.0; return o
    return sweep(pool, "E2S", "event_grow_time", [1200, 600, 300, 150, 75], runs,
                 protos=["NBHD_TRUST", "PROPOSED"], cfg_fn=f)

def E3S(pool, runs):  # mean-matching adversary: passes a neighbourhood-mean test by construction
    def f(o, v): o["attack"] = "mean_match"; o["field_period"] = 86400.0; o["replicas_per_victim"] = v; return o
    return sweep(pool, "E3S", "replicas_per_victim", [2, 4, 6], runs,
                 protos=["NBHD_TRUST", "PROPOSED"], cfg_fn=f)

def E1S(pool, runs):  # single malicious replica, no siblings: tests the *test*, not id pollution
    def f(o, v): o["attack"] = v; o["field_period"] = 86400.0; o["replicas_per_victim"] = 1; o["malicious_fraction"] = 1.0; return o
    return sweep(pool, "E1S", "attack", ["sem_suppress", "sem_displace", "mean_match"], runs,
                 protos=["NBHD_TRUST", "PROPOSED"], cfg_fn=f)

def E9(pool, runs):   # sensitivity of the proposed protocol
    rows = []
    rows += sweep(pool, "E9-kappa", "kappa", [1.5, 2.0, 2.5, 3.0, 4.0], runs, protos=["PROPOSED"])
    rows += sweep(pool, "E9-alpha", "alpha", [0.1, 0.2, 0.3, 0.5, 1.0], runs, protos=["PROPOSED"])
    rows += sweep(pool, "E9-delta", "delta", [30, 60, 120, 240, 480], runs, protos=["PROPOSED"])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=500)
    ap.add_argument("--cores", type=int, default=None)
    ap.add_argument("--only", default="E1,E2,E3,E4,E5,E6,E7,E8,E9")
    ap.add_argument("--trace", default=None, help="CSV t,node,x,y for E8")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    runs = 8 if a.quick else a.runs
    if a.quick: BASE["horizon"] = 1000
    out = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(out, exist_ok=True)
    print(f"runs={runs} cores={a.cores or mp.cpu_count()} out={out}/")
    t0 = time.time()
    with mp.Pool(a.cores) as pool:
        for name in a.only.split(","):
            print(f"== {name}")
            fn = globals()[name]
            rows = fn(pool, runs, a.trace) if name == "E8" else fn(pool, runs)
            if rows: save(rows, f"{out}/{name}.csv")
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
