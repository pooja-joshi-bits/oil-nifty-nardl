"""replicate_oilshocks.py -- structural oil-market shocks (Kilian 2009) and the Nifty 50.

    python replicate_oilshocks.py DF_with_globals.xlsx oil_market.csv OUTDIR

oil_market.csv: monthly columns date, prod (world crude oil production incl. lease
condensate, kb/d; EIA International Energy Statistics, series INTL.57-1-WORL-TBPD.M),
igrea (Kilian index of global real economic activity; FRED IGREA), cpi (US CPI-U;
FRED CPIAUCSL), wti (WTI spot, monthly average; FRED WTISPLC).

Step 1: VAR(24) in [100*dln prod, igrea, 100*ln(real oil price)], 1973:02-2025:07,
        recursive identification as in Kilian (2009): oil supply shock, aggregate
        demand shock, oil-specific demand shock.
Step 2: regress monthly Nifty returns (100*dln) on current and 12 lags of each shock
        (Kilian and Park 2009); cumulative responses with wild-bootstrap bands.
Step 3: contemporaneous regressions with controls, pre/post 2014 split.
Step 4: historical decomposition of the real oil price in three episodes.
"""
import sys, json, warnings, numpy as np, pandas as pd, statsmodels.api as sm
from statsmodels.tsa.api import VAR
warnings.filterwarnings("ignore"); rng = np.random.default_rng(2026)
fx, fo, out = sys.argv[1], sys.argv[2], sys.argv[3]; R = {}

om = pd.read_csv(fo, parse_dates=["date"]).set_index("date")
om["rpo"] = 100*np.log(om.wti/om.cpi)
Y = pd.DataFrame({"dprod": 100*np.log(om["prod"]).diff(), "rea": om.igrea, "rpo": om.rpo})["1973-02":"2025-07"].dropna()
p = 24
var = VAR(Y).fit(p, trend="c")
U = var.resid.values; S = np.cov(U.T, ddof=var.k_ar*3+1); P = np.linalg.cholesky(S)
E = pd.DataFrame(np.linalg.solve(P, U.T).T, index=var.resid.index, columns=["supply", "aggdemand", "oildemand"])
R["svar"] = dict(sample=[str(Y.index[0].date()), str(Y.index[-1].date())], lags=p, n=int(var.nobs))

# ---- historical decomposition of rpo (demeaned), contributions of each shock
ma = var.ma_rep(len(E)-1); Theta = np.array([m @ P for m in ma])  # h x 3 x 3
eps = E.values; T = len(eps); HD = np.zeros((T, 3))
for t in range(T):
    for k in range(3):
        HD[t, k] = np.sum(Theta[:t+1, 2, k] * eps[t::-1, k])
HD = pd.DataFrame(HD, index=E.index, columns=E.columns)
def episode(a, b):
    d = (HD.loc[b] - HD.loc[a]).round(1).to_dict(); d["actual_change_rpo"] = round(float(Y.rpo[b]-Y.rpo[a]), 1); return d
R["histdec"] = {"2008:07-2009:02": episode("2008-07-01", "2009-02-01"),
                "2014:06-2016:01": episode("2014-06-01", "2016-01-01"),
                "2019:12-2020:04": episode("2019-12-01", "2020-04-01"),
                "2020:04-2022:06": episode("2020-04-01", "2022-06-01")}

# ---- Nifty data
d = pd.read_excel(fx).dropna(subset=["Date"]).set_index("Date")
d["M2"] = d.M2.astype(str).str.replace("T", "").astype(float)
D = pd.DataFrame({"r": 100*np.log(d.Nifty).diff(), "dx": 100*np.log(d.Xrate).diff(), "spx": 100*np.log(d.SP500).diff(),
                  "dBY": d.BY.diff(), "dpi": (100*np.log(d.Infl).diff(12)).diff(), "dm2": 100*np.log(d.M2).diff()})
DD = D.join(E, how="inner")["2001-02":"2025-07"].dropna()
R["n_nifty"] = len(DD)
sh = ["supply", "aggdemand", "oildemand"]

# Step 2: Kilian-Park distributed lag, H=12
H = 12
def dl(df, H=H):
    X = pd.concat({f"{s}_{j}": df[s].shift(j) for s in sh for j in range(H+1)}, axis=1)
    Z = pd.concat([df.r, X], axis=1).dropna(); m = sm.OLS(Z.r, sm.add_constant(Z.drop(columns="r"))).fit()
    return m, Z
full = DD.join(E, rsuffix="_e", how="left")  # same
Efull = E["2000-01":"2025-07"]; base = pd.concat([D.r, Efull], axis=1)["1999-12":"2025-07"]
m, Z = dl(base["2000-02":"2025-07"].copy())
cum = {s: np.cumsum([m.params[f"{s}_{j}"] for j in range(H+1)]) for s in sh}
boots = {s: [] for s in sh}; Xm = sm.add_constant(Z.drop(columns="r")); fitted = m.fittedvalues.values; res = m.resid.values
for _ in range(1000):
    yb = fitted + res*rng.choice([-1., 1.], len(res)); bb = np.linalg.lstsq(Xm.values, yb, rcond=None)[0]; bb = pd.Series(bb, index=Xm.columns)
    for s in sh: boots[s].append(np.cumsum([bb[f"{s}_{j}"] for j in range(H+1)]))
R["kp_cumulative"] = {s: {"h0": round(cum[s][0], 2), "h3": round(cum[s][3], 2), "h6": round(cum[s][6], 2), "h12": round(cum[s][12], 2),
                          "lo90_h0": round(np.percentile(np.array(boots[s])[:, 0], 5), 2), "hi90_h0": round(np.percentile(np.array(boots[s])[:, 0], 95), 2),
                          "lo90_h12": round(np.percentile(np.array(boots[s])[:, 12], 5), 2), "hi90_h12": round(np.percentile(np.array(boots[s])[:, 12], 95), 2)} for s in sh}
R["kp_N"] = int(m.nobs); R["kp_R2"] = round(m.rsquared, 3)
# R2 share by shock group (joint F-tests)
for s in sh:
    t = m.f_test(", ".join(f"{s}_{j}=0" for j in range(H+1))); R.setdefault("kp_joint_p", {})[s] = round(float(t.pvalue), 4)

# figure
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
names = {"supply": "Oil supply shock", "aggdemand": "Aggregate demand shock", "oildemand": "Oil-specific demand shock"}
fig, ax = plt.subplots(1, 3, figsize=(9, 2.7), sharey=True)
for a, s in zip(ax, sh):
    B = np.array(boots[s]); lo, hi = np.percentile(B, [5, 95], axis=0)
    a.fill_between(range(H+1), lo, hi, color="0.82"); a.plot(cum[s], "k"); a.axhline(0, color="k", lw=.5)
    a.set_title(names[s], fontsize=9); a.set_xlabel("months", fontsize=8); a.tick_params(labelsize=7)
ax[0].set_ylabel("cumulative Nifty return (%)", fontsize=8); fig.tight_layout(); fig.savefig(out+"/fig_oilshocks.pdf")

# Step 3: contemporaneous regressions with HAC SEs
def contemp(df, extra=(), label=""):
    X = sm.add_constant(df[list(sh)+list(extra)]); mm = sm.OLS(df.r, X).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
    return {"N": int(mm.nobs), "R2": round(mm.rsquared, 3), **{f"{v}": [round(mm.params[v], 3), round(mm.tvalues[v], 2)] for v in list(sh)+list(extra)}}
R["contemp"] = {"no_controls": contemp(DD),
                "with_fx_macro": contemp(DD, ("dx", "dBY", "dpi", "dm2")),
                "with_fx_macro_spx": contemp(DD, ("dx", "dBY", "dpi", "dm2", "spx")),
                "pre2014": contemp(DD[:"2014-10"], ("dx", "dBY", "dpi", "dm2")),
                "post2014": contemp(DD["2014-11":], ("dx", "dBY", "dpi", "dm2")),
                "post2014_excl_covid": contemp(DD["2014-11":].drop(pd.to_datetime(["2020-03-01", "2020-04-01", "2020-05-01"])), ("dx", "dBY", "dpi", "dm2"))}
# break test on the three shock coefficients at 2014:11 (with controls)
X = sm.add_constant(DD[list(sh)+["dx", "dBY", "dpi", "dm2"]]).copy(); post = (DD.index >= "2014-11-01").astype(float)
for s in sh: X["p_"+s] = DD[s]*post
X["p_c"] = post; mm = sm.OLS(DD.r, X).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
t = mm.wald_test(", ".join(f"p_{s}=0" for s in sh), use_f=False, scalar=True); R["break2014_chi2_3"] = [round(float(t.statistic), 2), round(float(t.pvalue), 3)]
# correlation of shocks with WTI log change used in the paper
R["corr_dloil_shocks"] = {s: round(float(np.corrcoef(np.log(d.Oil).diff().reindex(DD.index), DD[s])[0, 1]), 3) for s in sh}
json.dump(R, open(out+"/results_oilshocks.json", "w"), indent=1); print(json.dumps(R, indent=1))

# ---- pgfplots version of the figure (for Overleaf; no binary upload needed)
lines = [r"% fig_oilshocks.tex -- generated by replicate_oilshocks.py",
         r"\begin{tikzpicture}",
         r"\begin{groupplot}[group style={group size=3 by 1, horizontal sep=8mm, y descriptions at=edge left},",
         r"  width=0.34\linewidth, height=4.2cm, xmin=0, xmax=12, xtick={0,3,6,9,12}, ymin=-4, ymax=4,",
         r"  xlabel={months}, tick label style={font=\scriptsize}, label style={font=\scriptsize}, title style={font=\small}]"]
for i, s in enumerate(sh):
    B = np.array(boots[s]); lo, hi = np.percentile(B, [5, 95], axis=0)
    opt = r"[title={%s}%s]" % (names[s], r", ylabel={cumulative Nifty return (\%)}" if i == 0 else "")
    lines.append(r"\nextgroupplot" + opt)
    band = " ".join(f"({h},{hi[h]:.3f})" for h in range(H+1)) + " " + " ".join(f"({h},{lo[h]:.3f})" for h in range(H, -1, -1))
    lines.append(r"\addplot[fill=gray!30, draw=none] coordinates {" + band + r"} -- cycle;")
    lines.append(r"\addplot[black, thick, mark=none] coordinates {" + " ".join(f"({h},{cum[s][h]:.3f})" for h in range(H+1)) + "};")
    lines.append(r"\addplot[black, thin, mark=none] coordinates {(0,0) (12,0)};")
lines += [r"\end{groupplot}", r"\end{tikzpicture}"]
open(out+"/fig_oilshocks.tex", "w").write("\n".join(lines)+"\n")
