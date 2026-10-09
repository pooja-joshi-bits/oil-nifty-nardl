"""replicate.py -- reproduces every table and figure in the paper.

    python replicate.py            # uses DF_with_globals.xlsx in this folder
    python replicate.py DATA.xlsx

Writes results_main.json, results_global.json and the four figure PDFs.

Variable definitions: pi = 100 * 12-month log change of the WPI index;
IIPg = IIP column x 100 (year-on-year growth, percent); all price series in
logs. NARDL sample 2001:02-2025:07 (N=294); VAR sample 2001:01-2025:07.

Part 1 - main results (Tables 2-7, Figures 2-5).
Part 2 - global controls and the Brent benchmark check (Tables 8-9).
"""
import sys,json,warnings,numpy as np,pandas as pd,statsmodels.api as sm,scipy.stats as st
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller,kpss
from statsmodels.stats.diagnostic import acorr_breusch_godfrey,het_breuschpagan,linear_reset,recursive_olsresiduals
from statsmodels.stats.stattools import jarque_bera
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
warnings.filterwarnings("ignore"); rng=np.random.default_rng(2026); R={}
f=sys.argv[1] if len(sys.argv)>1 else "DF_with_globals.xlsx"
d=pd.read_excel(f).dropna(subset=["Date"]).set_index("Date"); d.index.freq="MS"
d["M2"]=d.M2.astype(str).str.replace("T","").astype(float)
D=pd.DataFrame({"oil":np.log(d.Oil),"x":np.log(d.Xrate),"pi":100*np.log(d.Infl).diff(12),"IIPg":d.IIP*100,"BY":d.BY,"m2":np.log(d.M2),"sm":np.log(d.Nifty),"cpi":np.log(d.Infl)})
DF=D.copy()  # full history kept so that lags and partial sums use 2000 data
# ---- unit roots
ur={}
for c in ["sm","oil","x","m2","BY","pi","IIPg"]:
    s=D[c]["2001-01":].dropna(); a=adfuller(s,regression="c",autolag="AIC"); k=kpss(s,regression="c",nlags="auto")
    a1=adfuller(s.diff().dropna(),regression="c",autolag="AIC"); k1=kpss(s.diff().dropna(),regression="c",nlags="auto")
    ur[c]=dict(adf=a[0],adf_p=a[1],kpss=k[0],kpss_p=k[1],dadf=a1[0],dadf_p=a1[1],dkpss=k1[0],dkpss_p=k1[1])
R["unitroot"]=ur
# ---- NARDL
do,dx=D.oil.diff(),D.x.diff(); op,on,xp,xn=do.clip(lower=0).fillna(0),do.clip(upper=0).fillna(0),dx.clip(lower=0).fillna(0),dx.clip(upper=0).fillna(0)
def build(levels=True):
    X=pd.DataFrame({"dsm1":D.sm.diff().shift(1)})
    for j in range(3): X[f"op{j}"]=op.shift(j); X[f"on{j}"]=on.shift(j)
    for j in range(2): X[f"xp{j}"]=xp.shift(j); X[f"xn{j}"]=xn.shift(j)
    X["dBY0"]=D.BY.diff(); X["dBY1"]=D.BY.diff().shift(1); X["dpi"]=D.pi.diff(); X["dIIP"]=D.IIPg.diff(); X["dm2"]=D.m2.diff()
    if levels:
        for n,s in {"L_sm":D.sm,"L_op":op.cumsum(),"L_on":on.cumsum(),"L_xp":xp.cumsum(),"L_xn":xn.cumsum(),"L_BY":D.BY,"L_pi":D.pi,"L_m2":D.m2}.items(): X[n]=s.shift(1)
    Z=pd.concat([D.sm.diff().rename("y"),X],axis=1).dropna()
    return Z
Z=build(); Xm=sm.add_constant(Z.drop(columns="y")); m=sm.OLS(Z.y,Xm).fit(); h=m.get_robustcov_results("HAC",maxlags=4)
R["nardl"]=dict(N=int(m.nobs),first=str(Z.index[0].date()),R2=m.rsquared,coef={k:dict(b=float(b),se=float(s),t=float(t),p=float(p)) for k,b,s,t,p in zip(Xm.columns,m.params,h.bse,h.tvalues,h.pvalues)})
lv=[c for c in Z if c.startswith("L_")]
R["bounds_F"]=float(m.f_test(" = ".join(lv)+" = 0").fvalue); R["bounds_t"]=float(m.tvalues["L_sm"])
R["bounds_F_HAC"]=float(h.wald_test(" = ".join(lv)+" = 0",use_f=True,scalar=True).statistic)
Z93=Z["2001-03":]; m93=sm.OLS(Z93.y,sm.add_constant(Z93.drop(columns="y"))).fit(); R["bounds_F_N293"]=float(m93.f_test(" = ".join(lv)+" = 0").fvalue)
def wald(r): t=h.wald_test(r,use_f=False,scalar=True); return dict(chi2=float(t.statistic),df=int(np.atleast_1d(t.df_denom)[0]) if False else r.count("=") if "," in r else 1,p=float(t.pvalue))
W={"oil_joint":"op0=on0, op1=on1, op2=on2","oil_sum":"op0+op1+op2=on0+on1+on2","oil_contemp":"op0=on0","oil_LR":"L_op=L_on","fx_joint":"xp0=xn0, xp1=xn1","fx_sum":"xp0+xp1=xn0+xn1","fx_contemp":"xp0=xn0","fx_LR":"L_xp=L_xn"}
R["wald"]={k:wald(v) for k,v in W.items()}
for k,n in [("oil_joint",3),("fx_joint",2)]: R["wald"][k]["df"]=n
b=m.params; R["sums"]={"op":float(b.op0+b.op1+b.op2),"on":float(b.on0+b.on1+b.on2),"xp":float(b.xp0+b.xp1),"xn":float(b.xn0+b.xn1)}
R["LR"]={k:float(-b[k]/b.L_sm) for k in lv if k!="L_sm"}; R["ecm"]=float(b.L_sm)
bg=acorr_breusch_godfrey(m,nlags=4); bp=het_breuschpagan(m.resid,Xm); rs=linear_reset(m,power=2,use_f=True); jb=jarque_bera(m.resid)
R["diag"]=dict(BG_p=float(bg[1]),BP_p=float(bp[1]),RESET_p=float(rs.pvalue),JB_p=float(jb[1]))
# short-run only
Z0=build(False); m0=sm.OLS(Z0.y,sm.add_constant(Z0.drop(columns="y"))).fit(); b0=m0.params
R["sr_only"]={"op":float(b0.op0+b0.op1+b0.op2),"on":float(b0.on0+b0.on1+b0.on2),"xp":float(b0.xp0+b0.xp1),"xn":float(b0.xn0+b0.xn1)}
# wild bootstrap (Rademacher, 999) for joint oil symmetry + key t's
def hacW(mod): return float(mod.get_robustcov_results("HAC",maxlags=4).wald_test(W["oil_joint"],use_f=False,scalar=True).statistic)
Xr=Xm.copy(); 
for j in range(3): Xr[f"o{j}"]=Xr[f"op{j}"]+Xr[f"on{j}"]
Xr=Xr.drop(columns=[f"op{j}" for j in range(3)]+[f"on{j}" for j in range(3)]); mr=sm.OLS(Z.y,Xr).fit()
w0=hacW(m); cnt=0; B=999
for i in range(B):
    ys=mr.fittedvalues+mr.resid*rng.choice([-1,1],len(Z)); cnt+= hacW(sm.OLS(ys,Xm).fit())>=w0
R["wild_oil_joint_p"]=(cnt+1)/(B+1)

# ---- sup-Wald (Andrews) on the ten SR oil/FX coefficients, 15% trimming
sr=[c for c in Xm.columns if c[:2] in ("op","on","xp","xn")]; n_=len(Z); best=(0,None)
for tb in range(int(.15*n_),int(.85*n_)):
    Dm=(np.arange(n_)>=tb).astype(float); Xi=Xm.copy()
    for c in sr: Xi["b_"+c]=Xm[c].values*Dm
    w_=float(sm.OLS(Z.y,Xi).fit().wald_test(", ".join(f"b_{c}=0" for c in sr),use_f=False,scalar=True).statistic)  # homoskedastic Wald, as in Andrews (1993)
    if w_>best[0]: best=(w_,str(Z.index[tb].date()))
R["supwald"]=dict(stat=best[0],date=best[1],q=len(sr))
# ---- post-2020 oil interactions
Dm=(Z.index>="2020-01-01").astype(float); Xi=Xm.copy(); oc=[c for c in sr if c[0]=="o"]
for c in oc: Xi["p_"+c]=Xm[c].values*Dm
Xi["p_const"]=Dm; hh=sm.OLS(Z.y,Xi).fit().get_robustcov_results("HAC",maxlags=4); t_=hh.wald_test(", ".join(f"p_{c}=0" for c in oc),use_f=False,scalar=True)
R["post2020"]=dict(chi2=float(t_.statistic),p=float(t_.pvalue),coefs={c:dict(b=float(hh.params[list(Xi.columns).index("p_"+c)]),p=float(hh.pvalues[list(Xi.columns).index("p_"+c)])) for c in oc})
# ---- lagged-only FX
Xl=Xm.drop(columns=["xp0","xn0"]).copy(); Xl["xp2"]=xp.shift(2).reindex(Z.index); Xl["xn2"]=xn.shift(2).reindex(Z.index); ml=sm.OLS(Z.y,Xl).fit(); hl=ml.get_robustcov_results("HAC",maxlags=4)
R["fx_lag_only"]=dict(R2=float(ml.rsquared),p={c:float(hl.pvalues[list(Xl.columns).index(c)]) for c in ["xp1","xn1","xp2","xn2"]},oil_sums=[float(ml.params[["op0","op1","op2"]].sum()),float(ml.params[["on0","on1","on2"]].sum())])
# ---- reverse TY handled in matrix below
# CUSUM / CUSUMsq
rr=recursive_olsresiduals(m,skip=len(Xm.columns)+1); w=rr[3] if False else None
rec=rr[0] if False else None
resid_rec=rr[1] if False else None
rresid, rparams, rypred, rresid_std, rresid_scaled, rcusum, rcusumci = rr
k=Xm.shape[1]; wres=rresid_scaled[k+1:] if len(rresid_scaled)==len(Z) else rresid_scaled; wres=wres[~np.isnan(wres)]
n=len(wres); cs=np.cumsum(wres)/wres.std(ddof=1); css=np.cumsum(wres**2)/np.sum(wres**2); tt=np.arange(1,n+1)
a=0.948; cb=a*np.sqrt(n)+2*a*tt/np.sqrt(n); c0=1.358/np.sqrt(n/2-1)  # 5% band, Edgerton-Wells approximation
dev=np.abs(css-tt/n); R["cusum"]=dict(cusum_breach=bool((np.abs(cs)>cb).any()),cusumsq_maxdev=float(dev.max()),cusumsq_band=float(c0),cusumsq_breach=bool(dev.max()>c0))
ix=Z.index[-n:]; bi=ix[dev>c0]; R["cusum"]["breach_from_to"]=[str(bi[0].date()),str(bi[-1].date())] if len(bi) else None
fig,ax=plt.subplots(1,2,figsize=(9,2.8)); ax[0].plot(ix,cs,"k"); ax[0].plot(ix,cb,"r--",lw=.8); ax[0].plot(ix,-cb,"r--",lw=.8); ax[0].set_title("CUSUM",fontsize=9)
ax[1].plot(ix,css,"k"); ax[1].plot(ix,tt/n+c0,"r--",lw=.8); ax[1].plot(ix,tt/n-c0,"r--",lw=.8); ax[1].set_title("CUSUM of squares",fontsize=9)
for a_ in ax: a_.tick_params(labelsize=8)
fig.tight_layout(); fig.savefig("fig_cusum.pdf")
# dynamic multipliers (h=0..12) for oil +/-
def dynmult(pos,H=12):
    phi=b.dsm1; th=b.L_sm; lvl=b.L_op if pos else b.L_on; sr=[b[f"{'op' if pos else 'on'}{j}"] for j in range(3)]
    y=np.zeros(H+3); dy=np.zeros(H+3)  # unit permanent change in partial sum at t=0
    for t in range(H+1):
        i=t+2; dxv=[1 if t-j==0 else 0 for j in range(3)]
        dy[i]=phi*dy[i-1]+sum(s*v for s,v in zip(sr,dxv))+th*y[i-1]+lvl*(1 if t>=1 else 0); y[i]=y[i-1]+dy[i]
    return y[2:]
mp,mn=dynmult(True),dynmult(False); R["dynmult"]={"pos":mp.tolist(),"neg":mn.tolist()}
fig,ax=plt.subplots(figsize=(5,2.8)); ax.plot(mp,"k-",label="1% oil price increase"); ax.plot(mn,"k--",label="1% oil price decrease (sign reversed)"); ax.plot(mp-mn,"0.5",lw=.8,label="difference"); ax.axhline(0,color="k",lw=.5)
ax.set_xlabel("months",fontsize=8); ax.set_ylabel("cumulative effect on ln SM (%)",fontsize=8); ax.legend(fontsize=7); ax.tick_params(labelsize=8); fig.tight_layout(); fig.savefig("fig_multipliers.pdf")
# ---- VAR
V=D.drop(columns="cpi")["2001-01":].dropna(); last=["oil","x","pi","IIPg","BY","m2","sm"]; first=["sm","oil","x","BY","pi","IIPg","m2"]; Hs=[0,2,5,11,14,17]
def fevd(cols): t=VAR(V[cols]).fit(2).fevd(18).decomp[cols.index("sm")]; return {c:[round(float(v)*100,1) for v in t[Hs,i]] for i,c in enumerate(cols)}
R["fevd_last"]=fevd(last); R["fevd_first"]=fevd(first)
r=VAR(V[last]).fit(2); S=r.sigma_u.values; ma=r.ma_rep(17); i=last.index("sm"); g=np.zeros((18,7))
for hh in range(18):
    num=np.array([sum((ma[l][i,:]@S[:,j])**2 for l in range(hh+1))/S[j,j] for j in range(7)]); g[hh]=num/num.sum()
R["gfevd"]={c:[round(float(v)*100,1) for v in g[Hs,j]] for j,c in enumerate(last)}
def irf_boot(data,cols,H=18,B=500,p=2):
    r=VAR(data[cols]).fit(p); k=cols.index("sm"); base=r.irf(H).orth_irfs[:,k,:]; Y=data[cols].values; U=r.resid.values; c=r.params.values[0]; A=list(r.coefs); T=len(Y); dr=[]
    for _ in range(B):
        e=U[rng.integers(0,len(U),T-p)]; e=e-e.mean(0); y=list(Y[:p])
        for t in range(T-p): y.append(c+sum(A[q]@y[-1-q] for q in range(p))+e[t])
        try: dr.append(VAR(pd.DataFrame(np.array(y),columns=cols)).fit(p).irf(H).orth_irfs[:,k,:])
        except Exception: pass
    q=np.percentile(np.array(dr),[5,95],axis=0); return base,q[0],q[1]
def summ(b_,lo,hi): return {c:dict(h0=float(b_[0,j]),h6=float(b_[6,j]),h12=float(b_[12,j]),h18=float(b_[18,j]),sig=[int(x) for x in range(19) if lo[x,j]>0 or hi[x,j]<0]) for j,c in enumerate(last)}
full=irf_boot(V,last); R["irf_full"]=summ(*full); pre=irf_boot(V[:"2019-12"],last); post=irf_boot(V["2020-01":],last); R["irf_pre"]=summ(*pre); R["irf_post"]=summ(*post); R["n_pre_post"]=[len(V[:"2019-12"]),len(V["2020-01":])]
def panel(ax,res,j,t): b_,lo,hi=res; ax.fill_between(range(19),lo[:,j]*100,hi[:,j]*100,color="0.82"); ax.plot(b_[:,j]*100,"k"); ax.axhline(0,color="k",lw=.5); ax.set_title(t,fontsize=9); ax.tick_params(labelsize=7)
fig,ax=plt.subplots(2,3,figsize=(9,4.6)); names={"oil":"Oil price","x":"Exchange rate (depreciation)","pi":"Inflation","IIPg":"IIP growth","BY":"Bond yield","m2":"M2"}
for a_,(c,t) in zip(ax.ravel(),names.items()): panel(a_,full,last.index(c),t+" shock")
for a_ in ax[1]: a_.set_xlabel("months",fontsize=8)
ax[0,0].set_ylabel("ln SM (%)",fontsize=8); ax[1,0].set_ylabel("ln SM (%)",fontsize=8); fig.tight_layout(); fig.savefig("fig_irf.pdf")
fig,ax=plt.subplots(1,2,figsize=(7,2.6)); panel(ax[0],pre,0,"Pre-COVID: oil price shock"); panel(ax[1],post,0,"Post-COVID: oil price shock"); [a_.set_xlabel("months",fontsize=8) for a_ in ax]; fig.tight_layout(); fig.savefig("fig_irf_covid.pdf")
# ---- Toda-Yamamoto full matrix
r3=VAR(V[last]).fit(3); P=r3.params; cov=r3.cov_params(); ci=list(cov.index)
def ty(ca,ce):
    idx=[(f"L{l}.{ca}",ce) for l in (1,2)]; bb=np.array([P.loc[a,c] for a,c in idx]); ii=[ci.index(k) for k in idx]; Vv=cov.values[np.ix_(ii,ii)]; return float(bb@np.linalg.solve(Vv,bb))
TY={}
for ca in last:
    row={ce:dict(chi2=ty(ca,ce),p=float(1-st.chi2.cdf(ty(ca,ce),2))) for ce in last if ce!=ca}; tot=sum(v["chi2"] for v in row.values()); row["SYSTEM"]=dict(chi2=tot,p=float(1-st.chi2.cdf(tot,12))); TY[ca]=row
R["TY"]=TY


json.dump(R, open("results_main.json", "w"), indent=1)
print("Part 1 done -> results_main.json")


# ==========================================================================
# Part 2: global controls (S&P 500, VIX) and the Brent benchmark check
# ==========================================================================

import json,numpy as np,pandas as pd,statsmodels.api as sm,warnings; warnings.filterwarnings("ignore")
d=pd.read_excel(f).dropna(subset=["Date"]).set_index("Date"); d.index.freq="MS"
d["M2"]=d.M2.astype(str).str.replace("T","").astype(float)
D=pd.DataFrame({"oil":np.log(d.Oil),"brent":np.log(d.OilAlt),"x":np.log(d.Xrate),"pi":100*np.log(d.Infl).diff(12),
                "IIPg":d.IIP*100,"BY":d.BY,"m2":np.log(d.M2),"sm":np.log(d.Nifty),"spx":np.log(d.SP500),"vix":d.VIX,"dxy":np.log(d.DXY)})
do,dx=D.oil.diff(),D.x.diff(); op,on,xp,xn=do.clip(lower=0),do.clip(upper=0),dx.clip(lower=0),dx.clip(upper=0)
def build(extra=None,oilvar="oil"):
    o=D[oilvar]; do_=o.diff(); op_,on_=do_.clip(lower=0),do_.clip(upper=0)
    X=pd.DataFrame({"dsm1":D.sm.diff().shift(1)})
    for j in range(3): X[f"op{j}"]=op_.shift(j); X[f"on{j}"]=on_.shift(j)
    for j in range(2): X[f"xp{j}"]=xp.shift(j); X[f"xn{j}"]=xn.shift(j)
    X["dBY0"]=D.BY.diff(); X["dBY1"]=D.BY.diff().shift(1); X["dpi"]=D.pi.diff(); X["dIIP"]=D.IIPg.diff(); X["dm2"]=D.m2.diff()
    if extra in ("spx","both"): X["dspx"]=D.spx.diff()
    if extra in ("vix","both"): X["dvix"]=D.vix.diff()
    for n,s in {"L_sm":D.sm,"L_op":op_.cumsum(),"L_on":on_.cumsum(),"L_xp":xp.cumsum(),"L_xn":xn.cumsum(),"L_BY":D.BY,"L_pi":D.pi,"L_m2":D.m2}.items(): X[n]=s.shift(1)
    if extra in ("spx","both"): X["L_spx"]=D.spx.shift(1)
    Z=pd.concat([D.sm.diff().rename("y"),X],axis=1).dropna()
    return Z
R={}
def fitrow(Z):
    m=sm.OLS(Z.y,sm.add_constant(Z.drop(columns="y"))).fit(); h=m.get_robustcov_results("HAC",maxlags=4); b=m.params
    idx=list(b.index)
    row=dict(N=int(m.nobs),R2=round(m.rsquared,3),on_sum=round(b.on0+b.on1+b.on2,4),op_sum=round(b.op0+b.op1+b.op2,4),
              xp_sum=round(b.xp0+b.xp1,4),xn_sum=round(b.xn0+b.xn1,4),
              on0=round(b.on0,4),on0_t=round(h.tvalues[idx.index("on0")],2))
    for extra_c in ["dspx","dvix","L_spx"]:
        if extra_c in idx: row[extra_c]=round(b[extra_c],4); row[extra_c+"_t"]=round(h.tvalues[idx.index(extra_c)],2)
    wj=h.wald_test("op0=on0, op1=on1, op2=on2",use_f=False,scalar=True)
    row["oil_joint_chi2"]=round(float(wj.statistic),2); row["oil_joint_p"]=round(float(wj.pvalue),3)
    ws=h.wald_test("op0+op1+op2=on0+on1+on2",use_f=False,scalar=True)
    row["oil_sum_chi2"]=round(float(ws.statistic),2); row["oil_sum_p"]=round(float(ws.pvalue),3)
    return row
R["baseline"]=fitrow(build(None))
R["spx"]=fitrow(build("spx"))
R["vix"]=fitrow(build("vix"))
R["both"]=fitrow(build("both"))
R["brent"]=fitrow(build(None,oilvar="brent"))
R["corr_doil_dspx"]=round(do.corr(D.spx.diff()),3)
R["corr_doil_dvix"]=round(do.corr(D.vix.diff()),3)
R["corr_wti_brent"]=round(do.corr(D.brent.diff()),3)
from statsmodels.tsa.api import VAR
V=D.drop(columns=["brent","dxy"]).dropna()
last=["oil","x","spx","vix","pi","IIPg","BY","m2","sm"]
r=VAR(V[last]).fit(2); t=r.fevd(18).decomp[last.index("sm")]
R["fevd_globals_h18"]={c:round(float(t[17,i])*100,1) for i,c in enumerate(last)}
json.dump(R, open("results_global.json", "w"), indent=1)
print("Part 2 done -> results_global.json")


# ==========================================================================
# Part 3: wild-bootstrap p-value for the sup-Wald break test (Section 5.6)
# ==========================================================================
rng=np.random.default_rng(2026)
d=pd.read_excel(f).dropna(subset=["Date"]).set_index("Date"); d["M2"]=d.M2.astype(str).str.replace("T","").astype(float)
D=pd.DataFrame({"oil":np.log(d.Oil),"x":np.log(d.Xrate),"pi":100*np.log(d.Infl).diff(12),"IIPg":d.IIP*100,"BY":d.BY,"m2":np.log(d.M2),"sm":np.log(d.Nifty)})
do,dx=D.oil.diff(),D.x.diff(); op,on,xp,xn=do.clip(lower=0),do.clip(upper=0),dx.clip(lower=0),dx.clip(upper=0)
X=pd.DataFrame({"dsm1":D.sm.diff().shift(1)})
for j in range(3): X[f"op{j}"]=op.shift(j); X[f"on{j}"]=on.shift(j)
for j in range(2): X[f"xp{j}"]=xp.shift(j); X[f"xn{j}"]=xn.shift(j)
X["dBY0"]=D.BY.diff(); X["dBY1"]=D.BY.diff().shift(1); X["dpi"]=D.pi.diff(); X["dIIP"]=D.IIPg.diff(); X["dm2"]=D.m2.diff()
for n_,s_ in {"L_sm":D.sm,"L_op":op.cumsum(),"L_on":on.cumsum(),"L_xp":xp.cumsum(),"L_xn":xn.cumsum(),"L_BY":D.BY,"L_pi":D.pi,"L_m2":D.m2}.items(): X[n_]=s_.shift(1)
Z=pd.concat([D.sm.diff().rename("y"),X],axis=1).dropna(); Xm=sm.add_constant(Z.drop(columns="y")); y=Z.y.values
sr=[c for c in Xm.columns if c[:2] in("op","on","xp","xn")]; n=len(Z); lo,hi=int(.15*n),int(.85*n)
Xv=Xm.values; k=Xv.shape[1]; sr_idx=[list(Xm.columns).index(c) for c in sr]
def supwald(yv):
    best=0.0
    for tb in range(lo,hi):
        Dm=(np.arange(n)>=tb).astype(float)
        Xi=np.hstack([Xv,Xv[:,sr_idx]*Dm[:,None]])
        beta,res,_,_=np.linalg.lstsq(Xi,yv,rcond=None); e=yv-Xi@beta; s2=e@e/(n-Xi.shape[1])
        XtXi=np.linalg.inv(Xi.T@Xi); b=beta[k:]; V=s2*XtXi[k:,k:]
        w=b@np.linalg.solve(V,b); best=max(best,w)
    return best
m0=sm.OLS(y,Xv).fit(); w0=supwald(y); print("sup-W observed",round(w0,2))
B=999; cnt=0
for i in range(B):
    ys=m0.fittedvalues+m0.resid*rng.choice([-1,1],n)   # null imposed: no break
    cnt+= supwald(ys)>=w0
R3={"supwald":round(w0,2),"boot_p":round((cnt+1)/(B+1),3)}; json.dump(R3,open("results_supwald.json","w")); print("Part 3 done -> results_supwald.json", R3)
