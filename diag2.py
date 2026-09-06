import numpy as np
from multiprocessing import Pool
from mwsnsim.sim import make_cfg, run
import run_e1
def one(a):
    attack, seed, over = a
    b=dict(run_e1.BASE); b.update(over)
    r=run(make_cfg(protocol="PROPOSED", attack=attack, **b), seed)
    return attack, over.get('tag',''), r
if __name__=="__main__":
    jobs=[("none",s,{'tag':'base'}) for s in range(2)]
    jobs+=[("naive",s,{'tag':'base'}) for s in range(2)]
    jobs+=[("sem_suppress",s,{'tag':'alpha0.5','alpha':0.5}) for s in range(2)]
    with Pool() as p: res=p.map(one,jobs)
    for atk,tag,r in res:
        print(f"{atk:12s} {tag:9s} P_d={r['P_d']:.2f} T_d={r['T_d']} T_q={r['T_q']} FPR={r['FPR']:.4f} S={r['S']:.2f} S_b={r['S_b']:.2f} C={r['C']:.0f}")
