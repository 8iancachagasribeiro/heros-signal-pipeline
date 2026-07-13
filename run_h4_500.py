"""H4 frontier at 500 replicates. Chunked: python run_h4_500.py <sigma_b> <chunk> <n_chunks>"""
import warnings, sys, time; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import h4_frontier as H
from fastlrt import lrt_random_slope_fast
from calibration_fidelity_aliasing import simulate, within_r, even_days

N_SIMS = 500
DENS = [2,3,5,7,10,14,21,28]
RELS = [0.55,0.60,0.65,0.70,0.75,0.80,0.85]

def cell(rng, d, rel, sb, n=N_SIMS):
    both = c1 = c2 = 0
    for _ in range(n):
        Y, x, _ = simulate(rng, even_days(d,2), rel, sb)
        prop = float(np.nanmean(np.abs(within_r(Y,x)) > 0.20))
        p = lrt_random_slope_fast(Y,x)
        if not np.isfinite(p): continue
        a, b = prop > 0.50, p < 0.05
        c1 += a; c2 += b; both += (a and b)
    return both/n, c1/n, c2/n

if __name__ == "__main__":
    sb = float(sys.argv[1]); ch = int(sys.argv[2]); nch = int(sys.argv[3])
    cells = [(d,r) for d in DENS for r in RELS]
    mine = cells[ch::nch]
    rng = np.random.default_rng(7000 + ch)
    t0 = time.time(); rows = []
    for d, r in mine:
        rec, s, l = cell(rng, d, r, sb)
        rows.append(dict(sigma_b=sb, obs_per_cycle=d, reliability=r, n_cycles=2,
                         total_obs_per_subj=d*2, recovery=rec,
                         crit1_effectsize=s, crit2_lrt=l, n_sims=N_SIMS))
        print(f"  sb={sb} d={d:>2} rel={r:.2f} -> rec={rec:.3f} (size={s:.2f} lrt={l:.2f}) [{time.time()-t0:.0f}s]", flush=True)
    pd.DataFrame(rows).to_csv(f"/home/claude/repro/h4_500_sb{sb}_c{ch}.csv", index=False)
    print(f"[chunk {ch}/{nch} done, {len(rows)} cells, {time.time()-t0:.0f}s]")
