"""bounds_mc.py -- finite-sample critical values and p-values for the PSS bounds test.

Monte Carlo in the spirit of Pesaran, Shin and Smith (2001), Narayan (2005) and
Kripfganz and Schneider (2020), matched to this paper's specification:
Case III (unrestricted intercept, no trend), k = 7 level regressors, T = 294,
and 17 short-run regressors (as in the baseline NARDL).

Under the null of no level relationship the dependent variable is a random walk
independent of the regressors. The lower bound uses I(0) regressors (iid normal),
the upper bound I(1) regressors (independent random walks). For each draw we
compute the F statistic on the eight lagged-level coefficients and the t statistic
on the lagged dependent variable.

    python bounds_mc.py [reps]
"""
import sys, json, numpy as np
rng = np.random.default_rng(2026)
T, k, n_sr = 294, 7, 17
reps = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
F_obs, t_obs = 3.342, -4.179
burn = 50

def draw(integrated):
    n = T + burn + 1
    y = np.cumsum(rng.standard_normal(n))
    e = rng.standard_normal((n, k))
    x = np.cumsum(e, axis=0) if integrated else e
    dy = np.diff(y)[burn:]; ylag = y[burn:-1]
    xlag = x[burn:-1]; dx = np.diff(x, axis=0)[burn:]
    dylag = np.diff(y)[burn - 1:-1]
    extra = rng.standard_normal((T, n_sr - k - 1))   # remaining stationary short-run regressors
    X = np.column_stack([np.ones(T), ylag, xlag, dylag, dx, extra])
    return dy, X

def stats(dy, X):
    XtX = X.T @ X; b = np.linalg.solve(XtX, X.T @ dy); e = dy - X @ b
    df = len(dy) - X.shape[1]; s2 = e @ e / df; V = s2 * np.linalg.inv(XtX)
    idx = np.arange(1, 2 + k)                          # ylag and the k xlag columns
    bb = b[idx]; F = bb @ np.linalg.solve(V[np.ix_(idx, idx)], bb) / len(idx)
    t = b[1] / np.sqrt(V[1, 1])
    return F, t

R = {}
for name, integ in [("I0", False), ("I1", True)]:
    Fs, ts = np.empty(reps), np.empty(reps)
    for r in range(reps):
        Fs[r], ts[r] = stats(*draw(integ))
    R[name] = dict(F_cv={"10%": round(float(np.quantile(Fs, .90)), 3), "5%": round(float(np.quantile(Fs, .95)), 3), "1%": round(float(np.quantile(Fs, .99)), 3)},
                   t_cv={"10%": round(float(np.quantile(ts, .10)), 3), "5%": round(float(np.quantile(ts, .05)), 3), "1%": round(float(np.quantile(ts, .01)), 3)},
                   p_F=round(float((Fs >= F_obs).mean()), 4), p_t=round(float((ts <= t_obs).mean()), 4))
R["setup"] = dict(T=T, k=k, short_run_regressors=n_sr, reps=reps, case="III", F_obs=F_obs, t_obs=t_obs)
print(json.dumps(R, indent=1)); json.dump(R, open("results_bounds_mc.json", "w"), indent=1)
