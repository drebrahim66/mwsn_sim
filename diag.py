import sys, numpy as np
from multiprocessing import Pool
from mwsnsim.sim import make_cfg, run
import run_e1
def one(a):
    proto, attack, seed, over = a
    b=dict(run_e1.BASE); b.update(over)
    r=run(make_cfg(protocol=proto, attack=attack, **b), seed)
    return proto, over.get('tag',''), r
if __name__=="__main__":
    jobs=[("NBHD_TRUST","sem_suppress",s,{'tag':'nbhd-fixed'}) for s in range(2)]
    jobs+=[("SIG_RANGE","sem_suppress",s,{'tag':'nogossip','nonce_gossip':False}) for s in range(2)]
    jobs+=[("SIG_RANGE","none",s,{'tag':'noattack-gossip'}) for s in range(2)]
    with Pool() as p: res=p.map(one,jobs)
    for proto,tag,r in res:
        print(f"{proto:10s} {tag:16s} P_d={r['P_d']:.2f} T_d={r['T_d']} T_q={r['T_q']} FPR={r['FPR']:.3f} S={r['S']:.2f} S_b={r['S_b']:.2f}")
