"""replicate_revision.py -- additional results added in the November 2026 revision (Table 10, wild-bootstrap IRF/FEVD bands, joint Toda-Yamamoto system tests).
   python replicate_revision.py DF_with_globals.xlsx .
"""
import sys, json, itertools, warnings, numpy as np, pandas as pd, statsmodels.api as sm, scipy.stats as st
from statsmodels.tsa.api import VAR
warnings.filterwarnings("ignore"); rng = np.random.default_rng(2026); R = {}
f, out = sys.argv[1], sys.argv[2]
d = pd.read_excel(f).dropna(subset=["Date"]).set_index("Date"); d.index.freq = "MS"
d["M2"] = d.M2.astype(str).str.replace("T", "").astype(float)
D = pd.DataFrame({"oil": np.log(d.Oil), "x": np.log(d.Xrate), "pi": 100*np.log(d.Infl).diff(12), "IIPg": d.IIP*100,
                  "BY": d.BY, "m2": np.log(d.M2), "sm": np.log(d.Nifty), "spx": np.log(d.SP500), "dxy": np.log(d.DXY),
                  "vix": d.VIX})
do, dx = D.oil.diff(), D.x.diff()
op, on, xp, xn = do.clip(lower=0).fillna(0), do.clip(upper=0).fillna(0), dx.clip(lower=0).fillna(0), dx.clip(upper=0).fillna(0)

def build(po=3, px=2, pby=2, psm=1, extra=(), dummies=None):
    X = pd.DataFrame(index=D.index)
    for i in range(1, psm+1): X[f"dsm{i}"] = D.sm.diff().shift(i)
    for j in range(po): X[f"op{j}"] = op.shift(j); X[f"on{j}"] = on.shift(j)
    for j in range(px): X[f"xp{j}"] = xp.shift(j); X[f"xn{j}"] = xn.shift(j)
    for j in range(pby): X[f"dBY{j}"] = D.BY.diff().shift(j)
    X["dpi"] = D.pi.diff(); X["dIIP"] = D.IIPg.diff(); X["dm2"] = D.m2.diff()
    for e in extra: X["d"+e] = D[e].diff()
    for n, s in {"L_sm": D.sm, "L_op": op.cumsum(), "L_on": on.cumsum(), "L_xp": xp.cumsum(), "L_xn": xn.cumsum(),
                 "L_BY": D.BY, "L_pi": D.pi, "L_m2": D.m2}.items(): X[n] = s.shift(1)
    if dummies:
        for dt in dummies: X["D"+dt] = (X.index == pd.Timestamp(dt)).astype(float)
    return pd.concat([D.sm.diff().rename("y"), X], axis=1).dropna()

def fit(Z):
    Xm = sm.add_constant(Z.drop(columns="y")); m = sm.OLS(Z.y, Xm).fit(); h = m.get_robustcov_results("HAC", maxlags=4)
    return m, h, Xm

def summary(Z, po=3, px=2):
    m, h, Xm = fit(Z); b = m.params; lv = [c for c in Z if c.startswith("L_")]
    oj = ", ".join(f"op{j}=on{j}" for j in range(po))
    w = h.wald_test(oj, use_f=False, scalar=True)
    r = dict(N=int(m.nobs), R2=round(m.rsquared, 3), aic=round(m.aic, 2),
             on_sum=round(sum(b[f"on{j}"] for j in range(po)), 3), op_sum=round(sum(b[f"op{j}"] for j in range(po)), 3),
             xp0=round(b.xp0, 3), xn0=round(b.xn0, 3), xp_sum=round(sum(b[f"xp{j}"] for j in range(px)), 3),
             xn_sum=round(sum(b[f"xn{j}"] for j in range(px)), 3),
             oil_sym_chi2=round(float(w.statistic), 2), oil_sym_p=round(float(w.pvalue), 3),
             boundsF=round(float(m.f_test(" = ".join(lv)+" = 0").fvalue), 2), ecm=round(b.L_sm, 3),
             dIIP=round(b.dIIP, 5), dIIP_t=round(float(h.tvalues[list(Xm.columns).index("dIIP")]), 2))
    for c in Xm.columns:
        if c in ("ddxy", "dspx"): r[c] = round(b[c], 3); r[c+"_t"] = round(float(h.tvalues[list(Xm.columns).index(c)]), 2)
    return r

# 1. common estimation sample for the lag grid (start 2001:04 so every lag set fits)
grid = []
for po, px, pby, psm in itertools.product([1, 2, 3, 4], [1, 2, 3], [1, 2, 3], [1, 2]):
    Z = build(po, px, pby, psm)["2001-06":]; m, _, _ = fit(Z)
    grid.append(dict(po=po-1, px=px-1, pby=pby-1, psm=psm, aic=m.aic, bic=m.bic))
G = pd.DataFrame(grid).sort_values("aic"); R["lag_grid_top5_aic"] = G.head(5).round(2).to_dict("records")
R["lag_grid_top5_bic"] = G.sort_values("bic").head(5).round(2).to_dict("records")
base_row = G[(G.po == 2) & (G.px == 1) & (G.pby == 1) & (G.psm == 1)].iloc[0]
R["baseline_aic_rank"] = int((G.aic < base_row.aic).sum()) + 1; R["n_specs"] = len(G)

# 2. baseline and robustness rows
R["baseline"] = summary(build())
covid = ["2020-03-01", "2020-04-01", "2020-05-01", "2021-04-01", "2021-05-01"]
R["covid_dummies"] = summary(build(dummies=covid))
R["dxy"] = summary(build(extra=("dxy",)))
R["dxy_spx"] = summary(build(extra=("dxy", "spx")))
R["corr_dx_ddxy"] = round(float(D.x.diff().corr(D.dxy.diff())), 3)
# winsorised IIP growth at 1/99 pct
I0 = D.IIPg.copy(); lo, hi = I0.quantile([.01, .99]); D["IIPg"] = I0.clip(lo, hi)
R["iip_winsor"] = summary(build()); D["IIPg"] = I0
R["post2014"] = summary(build()["2014-11":])  # after diesel price deregulation (Oct 2014)

# 3. Toda-Yamamoto with a proper joint (cross-equation) Wald test for the system column
V = D[["oil", "x", "pi", "IIPg", "BY", "m2", "sm"]]["2001-01":].dropna(); cols = list(V.columns)
r3 = VAR(V).fit(3); P = r3.params; C = r3.cov_params(); ci = list(C.index)
def tyjoint(ca, effects):
    idx = [(f"L{l}.{ca}", ce) for ce in effects for l in (1, 2)]
    bb = np.array([P.loc[a, c] for a, c in idx]); ii = [ci.index(k) for k in idx]; Vv = C.values[np.ix_(ii, ii)]
    w = float(bb @ np.linalg.solve(Vv, bb)); return dict(chi2=round(w, 2), df=len(idx), p=round(float(1-st.chi2.cdf(w, len(idx))), 3))
R["TY_system_joint"] = {ca: tyjoint(ca, [c for c in cols if c != ca]) for ca in cols}
R["TY_sum_of_rows_check"] = {}
for ca in cols:
    s = sum(tyjoint(ca, [ce])["chi2"] for ce in cols if ce != ca); R["TY_sum_of_rows_check"][ca] = round(s, 2)

# 4. VAR(2): recursive-design wild bootstrap (Goncalves-Kilian 2004) for IRFs and FEVD at h=18
def irf_fevd(data, p=2, H=18):
    r = VAR(data).fit(p); k = cols.index("sm")
    return r.irf(H).orth_irfs[:, k, :], r.fevd(H).decomp[k][[0, 5, 11, 17], :]
base_irf, base_fevd = irf_fevd(V)
r2 = VAR(V).fit(2); U = r2.resid.values; c0 = r2.params.values[0]; A = list(r2.coefs); Y = V.values; T = len(Y); p = 2
irfs, fevds = [], []
for _ in range(500):
    eta = rng.choice([-1., 1.], size=(T-p, 1)); e = U*eta; y = list(Y[:p])
    for t in range(T-p): y.append(c0 + sum(A[q] @ y[-1-q] for q in range(p)) + e[t])
    try:
        i_, f_ = irf_fevd(pd.DataFrame(np.array(y), columns=cols)); irfs.append(i_); fevds.append(f_)
    except Exception: pass
irfs, fevds = np.array(irfs), np.array(fevds)
lo, hi = np.percentile(irfs, [5, 95], axis=0)
R["irf_wild"] = {c: dict(h0=round(float(base_irf[0, j])*100, 2), h6=round(float(base_irf[6, j])*100, 2),
                         h12=round(float(base_irf[12, j])*100, 2), h18=round(float(base_irf[18, j])*100, 2),
                         sig=[int(x) for x in range(19) if lo[x, j] > 0 or hi[x, j] < 0]) for j, c in enumerate(cols)}
flo, fhi = np.percentile(fevds, [5, 95], axis=0)
R["fevd_wild_ci"] = {c: {h: [round(float(base_fevd[i, j])*100, 1), round(float(flo[i, j])*100, 1), round(float(fhi[i, j])*100, 1)]
                         for i, h in enumerate(["h1", "h6", "h12", "h18"])} for j, c in enumerate(cols)}
R["n_boot_ok"] = len(irfs)
json.dump(R, open(out+"/results_robust.json", "w"), indent=1)

# ---- Table 10 and the post-2014 / post-2020 interaction tests
covid=["2020-03-01","2020-04-01","2020-05-01","2021-04-01","2021-05-01"]
I0=D.IIPg.copy(); lo,hi=I0.quantile([.01,.99])
specs=[("Baseline",lambda: build(),3,2),("COVID dummies",lambda: build(dummies=covid),3,2),("AIC lags",lambda: build(2,1,2,2),2,1),
       ("AIC+COVID",lambda: build(2,1,2,2,dummies=covid),2,1),("IIP wins",None,3,2),("Pre-2014",lambda: build()[:"2014-10"],3,2),("Post-2014+COVID",lambda: build(dummies=covid)["2014-11":],3,2)]
for lab,fn,po,px in specs:
    if fn is None:
        D["IIPg"]=I0.clip(lo,hi); Z=build(); D["IIPg"]=I0
    else: Z=fn()
    m,h,Xm=fit(Z); c=list(Xm.columns); b=m.params
    onl=[f"on{j}" for j in range(po)]; opn=[f"op{j}" for j in range(po)]
    w=h.wald_test(", ".join(f"op{j}=on{j}" for j in range(po)),use_f=False,scalar=True)
    s1=h.t_test("+".join(opn)+"=0"); s2=h.t_test("+".join(onl)+"=0")
    lv=[x for x in Z if x.startswith("L_")]
    def t(v): return round(float(h.tvalues[c.index(v)]),2)
    print(f"{lab:18s} op0 {b.op0:.3f} ({t('op0')}) on0 {b.on0:.3f} ({t('on0')}) opS {np.ravel(s1.effect)[0]:.3f} ({np.ravel(s1.tvalue)[0]:.2f}) onS {np.ravel(s2.effect)[0]:.3f} ({np.ravel(s2.tvalue)[0]:.2f}) symp {float(w.pvalue):.3f} xp0 {b.xp0:.3f} ({t('xp0')}) xn0 {b.xn0:.3f} ({t('xn0')}) F {float(m.f_test(' = '.join(lv)+' = 0').fvalue):.2f} R2 {m.rsquared:.3f} N {int(m.nobs)}")
def inter(Z,date,label):
    m,h,Xm=fit(Z); Xi=Xm.copy(); Dm=(Z.index>=date).astype(float); oc=[c for c in Xm.columns if c[:2] in("op","on")]
    for c in oc: Xi["p_"+c]=Xm[c].values*Dm
    Xi["p_const"]=Dm; hh=sm.OLS(Z.y,Xi).fit().get_robustcov_results("HAC",maxlags=4)
    t=hh.wald_test(", ".join(f"p_{c}=0" for c in oc),use_f=False,scalar=True); print(label,"chi2(6)=%.2f p=%.3f"%(float(t.statistic),float(t.pvalue)))
inter(build(),"2014-11-01","post2014 interact")
inter(build(dummies=covid),"2014-11-01","post2014 interact +covid")
inter(build(dummies=covid),"2020-01-01","post2020 interact +covid")
