import sys, json, numpy as np
from multiprocessing import Pool
from mwsnsim.sim import make_cfg, run
BASE = dict(n_nodes=200, area=200.0, comm_range=32.0, speed=3.0, horizon=1500,
            n_victims=1, replicas_per_victim=4, malicious_fraction=0.5,
            mobility="java", exclude_victims_as_observers=True, nonce_gossip=True,
            field="event", event_grow_time=300.0)
def one(a):
    proto, attack, seed = a
    return run(make_cfg(protocol=proto, attack=attack, **BASE), seed)
if __name__ == "__main__":
    runs = int(sys.argv[1]); attack = sys.argv[2]
    jobs = [(p, attack, s) for p in ["XED","SIG_RANGE","NBHD_TRUST","PROPOSED"] for s in range(runs)]
    with Pool() as pool: res = pool.map(one, jobs)
    print(f"attack={attack}  n=200 area=200 r=32 4 replicas (2 malicious)  runs={runs}")
    print(f"{'proto':11s} {'P_d':>5s} {'T_d':>6s} {'T_q':>6s} {'FPR':>6s} {'S':>5s} {'S_b':>5s} {'M(B)':>6s} {'C':>6s} {'E(mJ)':>6s}")
    for p in ["XED","SIG_RANGE","NBHD_TRUST","PROPOSED"]:
        r=[x for x in res if x["protocol"]==p]
        f=lambda k: np.mean([x[k] for x in r if x[k] is not None]) if any(x[k] is not None for x in r) else float('nan')
        print(f"{p:11s} {f('P_d'):5.2f} {f('T_d'):6.0f} {f('T_q'):6.0f} {f('FPR'):6.3f} {f('S'):5.2f} {f('S_b'):5.2f} {f('M'):6.0f} {f('C'):6.0f} {1e3*f('E'):6.1f}")
