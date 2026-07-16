#!/usr/bin/env python3
from __future__ import annotations
import argparse, gc, json, re, shutil, sys, time, zipfile
from pathlib import Path
import arviz as az
import numpy as np
import pandas as pd

DATA_RE=re.compile(r"data_.*_sim(\d+)\.csv$")
TRACE_RE=re.compile(r"sim(\d+)\.nc$")
MODES=["nominal","narrow","broad","mildly_misspecified"]
ALT=["narrow","broad","mildly_misspecified"]
PARAMS=["y0","ymax","K_A","h","sigma0","rho_A","sigma_lab","sigma_batch","nu"]

def save(records,path,sortcols):
    df=pd.DataFrame(records)
    if len(df): df=df.sort_values(sortcols)
    path.parent.mkdir(parents=True,exist_ok=True); df.to_csv(path,index=False)

def param_rows(idata,sim,mode,source):
    out=[]
    for name in PARAMS:
        if name not in idata.posterior: continue
        x=np.asarray(idata.posterior[name].values,dtype=float).reshape(-1)
        out.append({"simulation_id":sim,"prior_mode":mode,"parameter":name,"posterior_median":float(np.median(x)),"posterior_q1":float(np.quantile(x,.25)),"posterior_q3":float(np.quantile(x,.75)),"posterior_mean":float(np.mean(x)),"source":source})
    return out

def map_members(zf,regex,suffix=None):
    out={}
    for n in zf.namelist():
        if suffix and not n.lower().endswith(suffix): continue
        m=regex.search(Path(n).name)
        if m: out[int(m.group(1))]=n
    return out

def load_idata(zf,member,temp):
    target=temp/Path(member).name
    with zf.open(member) as src, target.open("wb") as dst: shutil.copyfileobj(src,dst,16*1024*1024)
    return az.from_netcdf(target),target

def summary(metrics,params,out):
    metcols=["rmse","mae","ppc_coverage","pvs_theta","pvs_pred","pvs_joint","n_divergences","rhat_max","ess_bulk_min","ess_tail_min","bfmi_min","runtime_seconds"]
    rows=[]
    for mode,g in metrics.groupby("prior_mode"):
        r={"prior_mode":mode,"n_runs":g.simulation_id.nunique()}
        for m in metcols:
            x=pd.to_numeric(g[m],errors="coerce").dropna(); r[f"{m}_median"]=x.median() if len(x) else np.nan; r[f"{m}_q1"]=x.quantile(.25) if len(x) else np.nan; r[f"{m}_q3"]=x.quantile(.75) if len(x) else np.nan
        rows.append(r)
    pd.DataFrame(rows).sort_values("prior_mode").to_csv(out/"tables/prior_sensitivity_summary.csv",index=False)
    pr=[]
    for (mode,p),g in params.groupby(["prior_mode","parameter"]):
        x=pd.to_numeric(g.posterior_median,errors="coerce").dropna(); pr.append({"prior_mode":mode,"parameter":p,"n_runs":g.simulation_id.nunique(),"median_of_posterior_medians":x.median() if len(x) else np.nan,"q1_of_posterior_medians":x.quantile(.25) if len(x) else np.nan,"q3_of_posterior_medians":x.quantile(.75) if len(x) else np.nan})
    pd.DataFrame(pr).sort_values(["parameter","prior_mode"]).to_csv(out/"tables/prior_parameter_shift_summary.csv",index=False)
    nom=metrics[metrics.prior_mode=="nominal"][["simulation_id"]+metcols].rename(columns={m:f"{m}_nominal" for m in metcols})
    d=metrics.merge(nom,on="simulation_id",how="left")
    for m in metcols: d[f"delta_{m}_vs_nominal"]=pd.to_numeric(d[m],errors="coerce")-pd.to_numeric(d[f"{m}_nominal"],errors="coerce")
    d.to_csv(out/"tables/prior_sensitivity_delta_raw.csv",index=False)
    dr=[]
    for mode,g in d.groupby("prior_mode"):
        r={"prior_mode":mode,"n_runs":g.simulation_id.nunique()}
        for m in metcols:
            x=pd.to_numeric(g[f"delta_{m}_vs_nominal"],errors="coerce").dropna(); r[f"delta_{m}_median"]=x.median() if len(x) else np.nan; r[f"delta_{m}_q1"]=x.quantile(.25) if len(x) else np.nan; r[f"delta_{m}_q3"]=x.quantile(.75) if len(x) else np.nan
        dr.append(r)
    pd.DataFrame(dr).sort_values("prior_mode").to_csv(out/"tables/prior_sensitivity_delta_summary.csv",index=False)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--repo",type=Path,required=True); ap.add_argument("--config",type=Path,required=True); ap.add_argument("--metrics",type=Path,required=True); ap.add_argument("--synthetic-zip",type=Path,required=True); ap.add_argument("--nominal-pvs-zip",type=Path,required=True); ap.add_argument("--output",type=Path,required=True); ap.add_argument("--n-sim",type=int,default=30); ap.add_argument("--selection-seed",type=int,default=20260716); ap.add_argument("--state-file",type=Path,required=True); ap.add_argument("--cores",type=int,default=4); a=ap.parse_args()
    sys.path.insert(0,str(a.repo))
    from nanostatistics_stress_test.config import DGPConfig,InferenceConfig,ScenarioConfig
    from nanostatistics_stress_test.models_bayesian_pvs import fit_pvs_pymc
    cfg=json.loads(a.config.read_text()); dgp=DGPConfig(**cfg["dgp"]); ik=dict(cfg["inference"]); ik["progressbar"]=False; ik["save_traces"]=True; ik["cores"]=a.cores; inf=InferenceConfig(**ik)
    out=a.output
    for s in ["tables","traces","logs","temp"]: (out/s).mkdir(parents=True,exist_ok=True)
    mp=out/"tables/prior_sensitivity_metrics_raw.csv"; pp=out/"tables/prior_parameter_summaries_raw.csv"
    mrec=pd.read_csv(mp).to_dict("records") if mp.exists() else []; prec=pd.read_csv(pp).to_dict("records") if pp.exists() else []
    done={(int(r["simulation_id"]),str(r["prior_mode"])) for r in mrec}; pdone={(int(r["simulation_id"]),str(r["prior_mode"])) for r in prec}
    archived=pd.read_csv(a.metrics); archived=archived[archived.model_name=="PVS_aware_area_MCMC"]
    with zipfile.ZipFile(a.synthetic_zip) as dz, zipfile.ZipFile(a.nominal_pvs_zip) as tz:
        dm=map_members(dz,DATA_RE); tm=map_members(tz,TRACE_RE,".nc"); common=sorted(set(dm)&set(tm))
        if len(common)<a.n_sim: raise RuntimeError(f"Somente {len(common)} pares disponíveis")
        rng=np.random.default_rng(a.selection_seed); selected=sorted(map(int,rng.choice(common,size=a.n_sim,replace=False)))
        (out/"selected_simulation_ids.json").write_text(json.dumps({"selection_seed":a.selection_seed,"n_sim":a.n_sim,"simulation_ids":selected},indent=2))
        total=a.n_sim*4; job=0
        for sim in selected:
            with dz.open(dm[sim]) as fh: df=pd.read_csv(fh)
            job+=1; a.state_file.write_text(f"Job {job}/{total}: nominal sim {sim}\n")
            if (sim,"nominal") not in done:
                r=archived[archived.simulation_id==sim]
                if len(r)!=1: raise RuntimeError(f"Métrica nominal não única: sim {sim}")
                row=r.iloc[0].to_dict(); row["prior_mode"]="nominal"; row["source"]="archived_nominal"; mrec.append(row); done.add((sim,"nominal")); save(mrec,mp,["simulation_id","prior_mode"])
            if (sim,"nominal") not in pdone:
                idata,target=load_idata(tz,tm[sim],out/"temp"); prec.extend(param_rows(idata,sim,"nominal","archived_nominal")); pdone.add((sim,"nominal")); save(prec,pp,["simulation_id","prior_mode","parameter"]); del idata; target.unlink(missing_ok=True); gc.collect()
            for mi,mode in enumerate(ALT,1):
                job+=1; a.state_file.write_text(f"Job {job}/{total}: prior {mode}, sim {sim}\n")
                if (sim,mode) in done and (sim,mode) in pdone: print(f"[SKIP] sim={sim} prior={mode}",flush=True); continue
                print(f"[START] {job}/{total} sim={sim} prior={mode}",flush=True)
                sc=ScenarioConfig(scenario_id=f"prior_{mode}",description=f"Targeted prior sensitivity: {mode}",n_replicates=dgp.n_replicates,nu=dgp.nu,rho_A=dgp.rho_A,h=dgp.h,K_A=dgp.K_A,prior_mode=mode,y_phys_mode="nominal")
                tp=out/"traces"/f"trace_pvs_prior_{mode}_sim{sim}.nc"; seed=int(inf.random_seed)+100000*sim+10000*mi+2000; st=time.time()
                row,idata=fit_pvs_pymc(df,dgp,inf,sc,sim,seed,tp); row["prior_mode"]=mode; row["source"]="new_targeted_fit"; row["targeted_fit_seed"]=seed; row["wallclock_seconds_stage8"]=time.time()-st
                mrec=[r for r in mrec if not (int(r["simulation_id"])==sim and str(r["prior_mode"])==mode)]; mrec.append(row); done.add((sim,mode)); save(mrec,mp,["simulation_id","prior_mode"])
                if idata is not None:
                    prec=[r for r in prec if not (int(r["simulation_id"])==sim and str(r["prior_mode"])==mode)]; prec.extend(param_rows(idata,sim,mode,"new_targeted_fit")); pdone.add((sim,mode)); save(prec,pp,["simulation_id","prior_mode","parameter"]); del idata; gc.collect()
                print(f"[DONE] sim={sim} prior={mode} runtime={row['wallclock_seconds_stage8']:.1f}s flag={row.get('diagnostic_flag')}",flush=True)
    summary(pd.DataFrame(mrec),pd.DataFrame(prec),out); a.state_file.write_text("Prior sensitivity pilot completed\n"); print("Concluído",flush=True)
if __name__=="__main__": main()
