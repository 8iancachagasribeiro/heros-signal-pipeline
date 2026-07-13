"""
Fast ML likelihood-ratio test for between-person variance in the slope.

Balanced design (all subjects share the sampling schedule) => the marginal covariance
V = Z G Z' + sigma^2 I is IDENTICAL across subjects; invert once, not N times.

    y_ij = (b0 + u0_i) + (b1 + u1_i)*x_j + e_ij ,  u_i ~ N(0,G), e ~ N(0, sigma^2)

H0: random intercept only.   H1: random intercept + random slope.
Boundary => chi-bar-square null: 0.5*chi2(1) + 0.5*chi2(2).
Start values from method of moments (per-subject OLS), then L-BFGS-B.
"""
import numpy as np
from scipy import stats
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

_EPS = 1e-8

def _neg_ll(params, Y, X, random_slope):
    N, n = Y.shape
    sig2 = np.exp(2.0 * params[0])
    if random_slope:
        a, b, c = params[1], params[2], params[3]
        L = np.array([[a, 0.0], [b, c]])
        G = L @ L.T
    else:
        a = params[1]
        G = np.array([[a*a, 0.0], [0.0, 0.0]])
    V = X @ G @ X.T + sig2 * np.eye(n)
    try:
        cf = cho_factor(V, lower=True, check_finite=False)
        logdetV = 2.0 * np.sum(np.log(np.diag(cf[0])))
        ViX = cho_solve(cf, X, check_finite=False)          # V^-1 X
        A = X.T @ ViX
        ybar = Y.mean(axis=0)
        beta = np.linalg.solve(A, ViX.T @ ybar)
        R = Y - (X @ beta)[None, :]                          # (N,n)
        ViR = cho_solve(cf, R.T, check_finite=False)         # V^-1 R'  (n,N)
        quad = float(np.sum(R.T * ViR))
    except Exception:
        return 1e12
    if not np.isfinite(logdetV) or not np.isfinite(quad):
        return 1e12
    return 0.5 * (N*n*np.log(2*np.pi) + N*logdetV + quad)

def _mom_start(Y, X):
    """Method-of-moments start from per-subject OLS."""
    N, n = Y.shape
    XtXi = np.linalg.pinv(X.T @ X)
    B = Y @ (XtXi @ X.T).T                 # (N,2) per-subject [intercept, slope]
    R = Y - B @ X.T
    dof = max(n - 2, 1)
    s2 = float((R**2).sum() / (N*dof))     # residual variance
    se2 = s2 * np.diag(XtXi)               # sampling var of the OLS coefs
    g00 = max(B[:, 0].var(ddof=1) - se2[0], _EPS)
    g11 = max(B[:, 1].var(ddof=1) - se2[1], _EPS)
    return np.sqrt(max(s2, _EPS)), np.sqrt(g00), np.sqrt(g11)

def lrt_random_slope_fast(Y, x):
    """Y: (N,n) outcomes. x: (n,) predictor shared by all subjects. Returns p-value."""
    Y = np.asarray(Y, float); x = np.asarray(x, float)
    n = len(x)
    X = np.column_stack([np.ones(n), x])
    sd_e, sd0, sd1 = _mom_start(Y, X)
    ls = np.log(max(sd_e, 1e-4))

    try:
        r0 = minimize(_neg_ll, np.array([ls, sd0]), args=(Y, X, False),
                      method="L-BFGS-B", options=dict(maxiter=500))
        best1 = None
        for st in ([ls, sd0, 0.0, sd1], [ls, sd0, 0.0, sd1*3 + 1e-3], [ls, sd0*0.5, 0.0, 1e-3]):
            r = minimize(_neg_ll, np.array(st), args=(Y, X, True),
                         method="L-BFGS-B", options=dict(maxiter=500))
            if best1 is None or r.fun < best1.fun:
                best1 = r
    except Exception:
        return np.nan
    if r0.fun >= 1e11 or best1.fun >= 1e11:
        return np.nan
    stat = 2.0 * (r0.fun - best1.fun)
    if not np.isfinite(stat):
        return np.nan
    stat = max(stat, 0.0)
    return float(0.5*stats.chi2.sf(stat, 1) + 0.5*stats.chi2.sf(stat, 2))
