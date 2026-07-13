import warnings, sys, time; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import h4_frontier as H
from registered_test_power import phase_randomize, _corr, N_SUBJ, N_OBS, SPAN, R_X_REAL, R_Y_REAL

_xd = H.e2(np.linspace(0, H.CYCLE_LEN, 400, endpoint=False))

def sim(rng, sb, sigma_state, R_x, R_y):
    b = rng.normal(H.DA_OPT - H.K_GAIN*H._E2_MEAN, sb, N_SUBJ)
    offs = rng.uniform(0, H.CYCLE_LEN, N_SUBJ)
    Xs, Ys, r_true = [], [], []
    for i in range(N_SUBJ):
        days = np.sort(rng.choice(np.arange(SPAN), N_OBS, replace=False)).astype(float)
        xt = H.e2(days + offs[i])
        sig = H.inverted_u(b[i] + H.K_GAIN*xt)
        yt = sig + rng.normal(0, sigma_state, N_OBS)
        # TRUE coupling of this person: her own signal+state, densely, no instrument noise
        sig_d = H.inverted_u(b[i] + H.K_GAIN*_xd)
        yd = sig_d + rng.normal(0, sigma_state, len(_xd))
        r_true.append(_corr(_xd, yd))
        sdy = np.sqrt(max(yt.var(),1e-12)*(1-R_y)/R_y)
        sdx = np.sqrt(max(xt.var(),1e-12)*(1-R_x)/R_x)
        Ys.append(yt + rng.normal(0, sdy, N_OBS))
        Xs.append(xt + rng.normal(0, sdx, N_OBS))
    return Xs, Ys, np.array(r_true)

def surrogate_p(Xs, Ys, rng, B=100):
    S_obs = np.std([_corr(x,y) for x,y in zip(Xs,Ys)])
    S = np.array([np.std([_corr(phase_randomize(x,rng), y) for x,y in zip(Xs,Ys)]) for _ in range(B)])
    return (1 + int(np.sum(S >= S_obs))) / (B+1)

if __name__ == "__main__":
    states = [float(s) for s in sys.argv[1].split(",")]
    rng = np.random.default_rng(41); rows=[]; t0=time.time()
    for ss in states:
        for sb in [0.04, 0.075, 0.12, 0.20]:
            rej=0; med=[]
            for _ in range(25):
                Xs,Ys,rt = sim(rng, sb, ss, R_X_REAL, R_Y_REAL)
                rej += (surrogate_p(Xs,Ys,rng,B=100) < 0.05)
                med.append(np.median(np.abs(rt)))
            mr = float(np.mean(med))
            rows.append(dict(sigma_state=ss, sigma_b=sb, median_true_r=mr, power=rej/25))
            print(f"  SIGMA_STATE={ss:.3f} sigma_b={sb:.3f} -> median true |r|={mr:.3f}  power={rej/25:.2f}", flush=True)
    pd.DataFrame(rows).to_csv(f"/home/claude/sweep_{states[0]}.csv", index=False)
    print(f"[{time.time()-t0:.0f}s]")
