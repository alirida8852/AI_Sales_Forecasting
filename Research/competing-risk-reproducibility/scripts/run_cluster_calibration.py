#!/usr/bin/env python3
"""Cluster bootstrap uncertainty and IPCW calibration analysis for held-out CIF predictions."""
from __future__ import annotations
import argparse, sys, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from run_reference_pipeline import load_seed, censoring_km, censor_survival

def ipcw_vectors(test, cause, horizon, km, floor=0.05):
    a=pd.to_datetime(test["as_of_at"],utc=True,errors="coerce")
    close=pd.to_datetime(test["final_close_at"],utc=True,errors="coerce")
    cutoff=pd.to_datetime(test["observation_cutoff_at"],utc=True,errors="coerce")
    close_days=(close-a).dt.total_seconds().to_numpy()/86400
    cutoff_days=(cutoff-a).dt.total_seconds().to_numpy()/86400
    outcome=test["final_outcome"].astype(str).str.lower().to_numpy()
    observed=np.isfinite(close_days)&(close_days>=0)&(close_days<=cutoff_days)
    event_h=observed&(close_days<=horizon)
    event_free=(~event_h)&(cutoff_days>=horizon)
    y=((outcome==cause)&event_h).astype(float)
    w=np.zeros(len(test),dtype=float)
    for i in np.where(event_h)[0]:
        w[i]=1/max(censor_survival(km,float(close_days[i]),left_limit=True),floor)
    for i in np.where(event_free)[0]:
        w[i]=1/max(censor_survival(km,float(horizon)),floor)
    return y,w,(event_h|event_free)

def weighted_calibration(y,p,w):
    valid=(w>0)&np.isfinite(p)&np.isfinite(y)
    y=np.asarray(y)[valid]; p=np.asarray(p)[valid]; w=np.asarray(w)[valid]
    if len(y)<10 or len(np.unique(y))<2: return (np.nan,np.nan)
    z=np.log(np.clip(p,1e-6,1-1e-6)/(1-np.clip(p,1e-6,1-1e-6)))
    def objective(beta):
        eta=beta[0]+beta[1]*z
        # stable weighted logistic negative log likelihood
        loss=np.logaddexp(0,eta)-y*eta
        return float(np.sum(w*loss))
    res=minimize(objective,np.array([0.0,1.0]),method="BFGS")
    if not res.success and not np.isfinite(res.fun): return (np.nan,np.nan)
    return float(res.x[0]),float(res.x[1])

def weighted_brier(y,p,w,mult=None):
    if mult is not None:
        return float(np.sum(w*mult*(y-p)**2)/np.sum(mult))
    return float(np.sum(w*(y-p)**2)/len(y))

def bootstrap_cluster(y,p,w,clusters,n_boot,rng):
    labels=pd.Series(clusters).fillna("__MISSING__").astype(str).to_numpy()
    uniq, inv=np.unique(labels,return_inverse=True)
    n=len(uniq)
    vals=[]
    for _ in range(n_boot):
        draw=rng.integers(0,n,size=n)
        counts=np.bincount(draw,minlength=n)
        mult=counts[inv]
        if mult.sum()==0: continue
        vals.append(weighted_brier(y,p,w,mult))
    if not vals: return (np.nan,np.nan,np.nan)
    a=np.asarray(vals)
    return float(np.mean(a)),float(np.quantile(a,.025)),float(np.quantile(a,.975))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data-dir",type=Path,required=True)
    ap.add_argument("--predictions",type=Path,required=True)
    ap.add_argument("--output-dir",type=Path,required=True)
    ap.add_argument("--bootstrap",type=int,default=200)
    ap.add_argument("--random-state",type=int,default=314159)
    args=ap.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    pred=pd.read_csv(args.predictions)
    all_rows=[]
    for sd in sorted(args.data_dir.glob("seed_*")):
        seed=int(sd.name.split("_")[-1])
        frame=load_seed(sd)
        pseed=pred[pred.seed==seed]
        for landmark in ["day_0","day_30","day_60"]:
            train=frame[(frame.landmark==landmark)&(frame.scoring_eligible==True)&
                        (frame.split=="within_company_train_dev")].drop_duplicates("opportunity_id")
            km=censoring_km(train)
            for partition in ["within_company_future_test","unseen_company_test"]:
                sub=pseed[(pseed.landmark==landmark)&
                          (pseed.prediction_test_partition==partition)]
                for horizon in [90,180,365]:
                    ph=sub[sub.horizon_days==horizon].copy()
                    # split_manifest group_id is the account-level cluster identifier.
                    ph["account_id"]=ph["group_id"]
                    for cause,col in [("won","predicted_cif_won"),("lost","predicted_cif_lost")]:
                        y,w,known=ipcw_vectors(ph,cause,horizon,km)
                        pp=ph[col].to_numpy()
                        brier=weighted_brier(y,pp,w)
                        ci={}
                        for cluster in ["account_id","company_id"]:
                            rng=np.random.default_rng(args.random_state+seed+landmark.__len__()+horizon+(0 if cause=="won" else 100)+(0 if cluster=="account_id" else 1000))
                            mean,lo,hi=bootstrap_cluster(y,pp,w,ph[cluster],args.bootstrap,rng)
                            ci[cluster]=(mean,lo,hi)
                        intercept,slope=weighted_calibration(y,pp,w)
                        all_rows.append({
                          "seed":seed,"landmark":landmark,"test":partition,
                          "horizon_days":horizon,"cause":cause,"n":len(ph),
                          "n_known":int(known.sum()),"ipcw_brier":brier,
                          "account_bootstrap_mean":ci["account_id"][0],
                          "account_ci_2_5":ci["account_id"][1],"account_ci_97_5":ci["account_id"][2],
                          "company_bootstrap_mean":ci["company_id"][0],
                          "company_ci_2_5":ci["company_id"][1],"company_ci_97_5":ci["company_id"][2],
                          "calibration_intercept":intercept,"calibration_slope":slope,
                          "mean_predicted_cif":float(pp.mean()),
                          "observed_cause_rate_weighted":float(np.sum(w*y)/np.sum(w)) if w.sum() else np.nan,
                          "mean_positive_ipcw_weight":float(w[w>0].mean()) if np.any(w>0) else np.nan,
                          "max_ipcw_weight":float(w.max()) if len(w) else np.nan
                        })
    out=pd.DataFrame(all_rows)
    out.to_csv(args.output_dir/"cluster_bootstrap_calibration.csv",index=False)
    # Seed-aggregated summary; bootstrap intervals are per-seed cluster CIs, not pooled CIs.
    summary=out.groupby(["landmark","test","horizon_days","cause"]).agg(
      mean_ipcw_brier=("ipcw_brier","mean"),sd_ipcw_brier=("ipcw_brier","std"),
      mean_calibration_intercept=("calibration_intercept","mean"),
      mean_calibration_slope=("calibration_slope","mean"),
      mean_predicted_cif=("mean_predicted_cif","mean"),
      mean_observed_cause_rate_weighted=("observed_cause_rate_weighted","mean"),
      mean_account_ci_width=("account_ci_97_5",lambda x: np.nan),
    ).reset_index()
    summary.to_csv(args.output_dir/"cluster_bootstrap_calibration_summary.csv",index=False)
    meta={"bootstrap_replicates_per_cell":args.bootstrap,"cluster_types":["account_id","company_id"],
          "seed_count":3,"landmarks":3,"test_partitions":2,"horizons":[90,180,365],
          "causes":["won","lost"],"cells":len(out),
          "caveat":"Intervals are computed separately within each seed and test cell. Company bootstrap has only 18 company clusters in unseen-company tests and should be interpreted as sensitivity analysis. Calibration estimates are weighted point estimates, not bootstrap intervals."}
    (args.output_dir/"analysis_metadata.json").write_text(json.dumps(meta,indent=2))
    print(out.head(12).to_string(index=False))
    print(f"\nWrote {len(out)} cells to {args.output_dir}")

if __name__=="__main__": main()
