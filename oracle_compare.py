"""Compare Python SIG_RANGE against the ver7 Java engine on Driver.java's
configuration: 200 nodes, 200x200 m, radius 32, 4 victims x 1 replica,
cThresh=1, all replicas quarantined on nonce mismatch (naive attack with
malicious_fraction=1), victims excluded as observers."""
import sys, numpy as np
from multiprocessing import Pool
from mwsnsim.sim import make_cfg, run

def one(a):
    speed, seed = a
    cfg = make_cfg(protocol="SIG_RANGE", attack="naive", n_nodes=200, area=200.0,
                   comm_range=32.0, speed=speed, n_victims=4, replicas_per_victim=1,
                   malicious_fraction=1.0, horizon=3000, mobility="java",
                   exclude_victims_as_observers=True, run_full_horizon=False,
                   field="smooth", nonce_gossip=True, claims_from_any=True, strict_threshold=True)
    r = run(cfg, seed); return (r["T_q"], r["T_q_id"])

if __name__ == "__main__":
    runs = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    for speed in (2, 4):
        with Pool() as p:
            tq = p.map(one, [(speed, s) for s in range(runs)])
        a=[x[0] for x in tq]; b=[x[1] for x in tq]
        print(f"speed={speed}  T_q(pid)={a} mean={np.mean([x for x in a if x])}")
        print(f"          T_q(id) ={b} mean={np.mean([x for x in b if x])}")
