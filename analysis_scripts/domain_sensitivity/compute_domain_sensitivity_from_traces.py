#!/usr/bin/env python3
from __future__ import annotations
import argparse, gc, json, re, shutil, tempfile, time, zipfile
from pathlib import Path
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

DOMAINS={"narrow":(0.02,0.98),"nominal":(0.0,1.0),"broad":(-0.05,1.05)}
SIM_RE=re.compile(r"sim(\d+)\.nc$")
AREA_VARS=["y0","ymax","K_A","h","sigma0","rho_A","sigma_lab","sigma_batch","nu"]

def flat(idata,name):
    v=np.asarray(idata.posterior[name].values,dtype=float)
    return v.reshape(-1,*v.shape[2:])

def theta_mask(idata,ymin,ymax):
    a={n:flat(idata,n).reshape(-1) for n in AREA_VARS}
    ok=np.ones(len(a["y0"]),dtype=bool)
    ok&=a["y0"]>=ymin; ok&=a["ymax"]>=0; ok&=a["K_A"]>0; ok&=a["h"]>0
    ok&=a["sigma0"]>0; ok&=a["rho_A"]>=0; ok&=a["sigma_lab"]>0; ok&=a["sigma_batch"]>0; ok&=a["nu"]>2
    ok&=(a["y0"]+a["ymax"])<=ymax
    return ok

def pp_array(idata):
    if "posterior_predictive" not in idata.groups() or "y_obs" not in idata.posterior_predictive:
        raise KeyError("posterior_predictive/y_obs ausente")
    v=np.asarray(idata.posterior_predictive["y_obs"].values,dtype=float)
    return v.reshape(-1,v.shape[-1])

def stats(s):
    x=pd.to_numeric(s,errors="coerce").dropna()
    return {"n":len(x),"median":x.median() if len(x) else np.nan,"q1":x.quantile(.25) if len(x) else np.nan,"q3":x.quantile(.75) if len(x) else np.nan,"minimum":x.min() if len(x) else np.nan,"maximum":x.max() if len(x) else np.nan}

def process(zip_path,workflow,raw_path,existing,temp):
    rows=existing.to_dict("records") if not existing.empty else []
    done={(int(r["simulation_id"]),str(r["workflow"]),str(r["domain"])) for r in rows}
    with zipfile.ZipFile(zip_path) as zf:
        members=sorted(n for n in zf.namelist() if n.lower().endswith(".nc"))
        bar=tqdm(members,desc=f"Traces {workflow}",unit="trace")
        for member in bar:
            m=SIM_RE.search(Path(member).name)
            if not m: raise RuntimeError(f"simulation_id não identificado: {member}")
            sim=int(m.group(1)); needed=[d for d in DOMAINS if (sim,workflow,d) not in done]
            if not needed: continue
            bar.set_postfix(sim=sim)
            local=temp/f"{workflow}_{sim}.nc"
            with zf.open(member) as src, local.open("wb") as dst: shutil.copyfileobj(src,dst,16*1024*1024)
            idata=az.from_netcdf(local); pp=pp_array(idata)
            chains=int(idata.posterior.sizes.get("chain",-1)); draws=int(idata.posterior.sizes.get("draw",-1))
            for domain in needed:
                ymin,ymax=DOMAINS[domain]; predok=(pp>=ymin)&(pp<=ymax)
                t=j=np.nan
                if workflow=="area":
                    tok=theta_mask(idata,ymin,ymax); t=float(tok.mean()); nd=min(len(tok),predok.shape[0]); j=float(np.mean(tok[:nd,None]&predok[:nd,:]))
                rows.append({"simulation_id":sim,"workflow":workflow,"domain":domain,"y_min":ymin,"y_max":ymax,"chains":chains,"draws_per_chain":draws,"posterior_draws_total":chains*draws,"n_obs":pp.shape[1],"pvs_theta":t,"pvs_pred":float(predok.mean()),"pvs_joint":j,"trace_member":member})
                done.add((sim,workflow,domain))
            pd.DataFrame(rows).sort_values(["workflow","simulation_id","domain"]).to_csv(raw_path,index=False)
            del pp,idata; gc.collect(); local.unlink(missing_ok=True)
    return pd.DataFrame(rows).sort_values(["workflow","simulation_id","domain"])

def summaries(raw,metrics,out):
    rows=[]
    for (w,d),g in raw.groupby(["workflow","domain"]):
        r={"workflow":w,"domain":d,"y_min":g.y_min.iloc[0],"y_max":g.y_max.iloc[0],"n_runs":g.simulation_id.nunique()}
        for met in ["pvs_theta","pvs_pred","pvs_joint"]:
            for k,v in stats(g[met]).items(): r[f"{met}_{k}"]=v
        rows.append(r)
    pd.DataFrame(rows).sort_values(["workflow","domain"]).to_csv(out/"tables/domain_sensitivity_summary.csv",index=False)
    nom=raw[raw.domain=="nominal"][["simulation_id","workflow","pvs_theta","pvs_pred","pvs_joint"]].rename(columns={m:f"{m}_nominal" for m in ["pvs_theta","pvs_pred","pvs_joint"]})
    delta=raw.merge(nom,on=["simulation_id","workflow"],how="left")
    for m in ["pvs_theta","pvs_pred","pvs_joint"]: delta[f"delta_{m}_vs_nominal"]=delta[m]-delta[f"{m}_nominal"]
    delta.to_csv(out/"tables/domain_sensitivity_delta_raw.csv",index=False)
    ds=[]
    for (w,d),g in delta.groupby(["workflow","domain"]):
        r={"workflow":w,"domain":d,"n_runs":g.simulation_id.nunique()}
        for m in ["delta_pvs_theta_vs_nominal","delta_pvs_pred_vs_nominal","delta_pvs_joint_vs_nominal"]:
            for k,v in stats(g[m]).items(): r[f"{m}_{k}"]=v
        ds.append(r)
    pd.DataFrame(ds).to_csv(out/"tables/domain_sensitivity_delta_summary.csv",index=False)
    val=[]; lookup={"mass":"Bayes_hierarchical_mass_MCMC","area":"PVS_aware_area_MCMC"}
    for w,model in lookup.items():
        a=raw[(raw.workflow==w)&(raw.domain=="nominal")]; b=metrics[metrics.model_name==model]
        z=a.merge(b,on="simulation_id",suffixes=("_recomputed","_archived"))
        for m in ["pvs_theta","pvs_pred","pvs_joint"]:
            if w=="mass" and m in {"pvs_theta","pvs_joint"}: continue
            diff=(pd.to_numeric(z[f"{m}_recomputed"],errors="coerce")-pd.to_numeric(z[f"{m}_archived"],errors="coerce")).abs().dropna()
            val.append({"workflow":w,"metric":m,"n_compared":len(diff),"max_abs_difference":diff.max() if len(diff) else np.nan,"median_abs_difference":diff.median() if len(diff) else np.nan,"count_gt_1e-12":int((diff>1e-12).sum()),"count_gt_1e-8":int((diff>1e-8).sum())})
    pd.DataFrame(val).to_csv(out/"tables/nominal_recalculation_validation.csv",index=False)

def figures(raw,out):
    order=["narrow","nominal","broad"]
    vals=[]; labels=[]
    for w in ["mass","area"]:
        for d in order: vals.append(raw[(raw.workflow==w)&(raw.domain==d)].pvs_pred.dropna().to_numpy()); labels.append(f"{w}\n{d}")
    fig,ax=plt.subplots(figsize=(9,5)); ax.boxplot(vals,tick_labels=labels,showfliers=False); ax.set_ylabel("Predictive PVS"); ax.set_title("Predictive-domain sensitivity from archived traces"); fig.tight_layout(); fig.savefig(out/"figures/domain_sensitivity_pvs_pred.png",dpi=300); plt.close(fig)
    area=raw[raw.workflow=="area"]; vals=[]; labels=[]
    for m in ["pvs_theta","pvs_joint"]:
        for d in order: vals.append(area[area.domain==d][m].dropna().to_numpy()); labels.append(f"{m}\n{d}")
    fig,ax=plt.subplots(figsize=(9,5)); ax.boxplot(vals,tick_labels=labels,showfliers=False); ax.set_ylabel("PVS"); ax.set_title("Area-based parameter and joint PVS sensitivity"); fig.tight_layout(); fig.savefig(out/"figures/domain_sensitivity_area_theta_joint.png",dpi=300); plt.close(fig)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--mass-zip",type=Path,required=True); ap.add_argument("--pvs-zip",type=Path,required=True); ap.add_argument("--metrics",type=Path,required=True); ap.add_argument("--output",type=Path,required=True); ap.add_argument("--state-file",type=Path,required=True); a=ap.parse_args()
    t=time.time(); out=a.output; (out/"tables").mkdir(parents=True,exist_ok=True); (out/"figures").mkdir(parents=True,exist_ok=True); a.state_file.parent.mkdir(parents=True,exist_ok=True)
    rawp=out/"tables/domain_sensitivity_raw.csv"; existing=pd.read_csv(rawp) if rawp.exists() else pd.DataFrame(); metrics=pd.read_csv(a.metrics)
    with tempfile.TemporaryDirectory(prefix="nano_domain_") as td:
        a.state_file.write_text("Processing mass traces\n"); raw=process(a.mass_zip,"mass",rawp,existing,Path(td))
        a.state_file.write_text("Processing area traces\n"); raw=process(a.pvs_zip,"area",rawp,raw,Path(td))
    summaries(raw,metrics,out); figures(raw,out)
    raw.groupby(["workflow","domain"]).simulation_id.nunique().rename("n_simulations").reset_index().to_csv(out/"tables/domain_sensitivity_completeness.csv",index=False)
    (out/"domain_sensitivity_metadata.json").write_text(json.dumps({"method":"recalculation from archived ArviZ NetCDF traces","domains":DOMAINS,"mass_pvs_theta":"NA","mass_pvs_joint":"NA","elapsed_seconds":time.time()-t},indent=2),encoding="utf-8")
    a.state_file.write_text("Domain sensitivity completed\n"); print("Concluído:",out)
if __name__=="__main__": main()
