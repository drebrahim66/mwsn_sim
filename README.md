# MWSN semantic replica-quarantining simulator

Python re-implementation of the ver7 Java engine (validated: T_q within noise
of `OracleTest.java` on Driver.java's configuration), extended with a sensed
field, four protocols (XED, SIG_RANGE = ver7, NBHD_TRUST = Teng et al. 2026
data check, PROPOSED), attack classes, and the E1-E9 experiments.

    pip install numpy
    python3 -m mwsnsim.sim --protocol PROPOSED --attack sem_suppress --n 200   # one trial
    python3 oracle_compare.py 8                                                # Java calibration check
    python3 experiments.py --quick --only E1                                   # 8-run smoke test
    python3 experiments.py --runs 500 --cores 12                               # full evaluation
    python3 experiments.py --runs 500 --cores 12 --only E8 --trace trace.csv   # trace-driven (t,node,x,y)

Calibration notes (see comments in protocols.py / sim.py):
* ver7 curves were produced at 200 nodes / 200x200 m / r=32, with nonce gossip,
  claims from any Ql entry, strict threshold, and id-level quarantining.
* eps = 99.9th percentile of honest pairwise differences at contact distance
  (eps_frac = 0.32 of the field's dynamic range).
* NBHD_TRUST: local decisions (no gossip), >=5 observations before judging,
  A_th floored at 0.5*eps; cluster gap >= 0.2 required for a threshold.
* PROPOSED operating point: alpha=0.3, tau0=0.5, tau_q=0.2 (three failures).
