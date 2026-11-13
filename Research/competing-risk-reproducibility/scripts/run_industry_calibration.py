#!/usr/bin/env python3
"""Industry-stratified IPCW reliability bins and calibration summaries."""
import argparse,sys
from pathlib import Path
import numpy as np,pandas as pd
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from run_reference_pipeline import load_seed,censoring_km,censor_survival
from run_cluster_calibration import ipcw_vectors,weighted_calibration

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument("--data-dir",type=Path,required=True)
 ap.add_argument("--predictions",type=Path,required=True)
 ap.add_argument("--output-dir",type=Path,required=True)
 ap.add_argument("--bins",type=int,default=10)
 args=ap.parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True)
 pred=pd.read_csv(args.predictions)
 rows=[]; bins_out=[]
 for sd in sorted(args.data_dir.glob("seed_*")):
  seed=int(sd.name.split("_")[-1]); frame=load_seed(sd)
  company_ind=frame[["company_id","industry"]].drop_duplicates()
  ps=pred[pred.seed==seed].merge(company_ind,on="company_id",how="left",validate="many_to_one")
  for lm in ["day_0","day_30","day_60"]:
   tr=frame[(frame.landmark==lm)&(frame.scoring_eligible==True)&(frame.split=="within_company_train_dev")].drop_duplicates("opportunity_id")
   km=censoring_km(tr)
   for part in ["within_company_future_test","unseen_company_test"]:
    sub=ps[(ps.landmark==lm)&(ps.prediction_test_partition==part)]
    for h in [90,180,365]:
     ph=sub[sub.horizon_days==h]
     for cause,col in [("won","predicted_cif_won"),("lost","predicted_cif_lost")]:
      for industry,g in ph.groupby("industry"):
       y,w,known=ipcw_vectors(g,cause,h,km); p=g[col].to_numpy()
       intercept,slope=weighted_calibration(y,p,w)
       rows.append({"seed":seed,"landmark":lm,"test":part,"horizon_days":h,
                    "cause":cause,"industry":industry,"n":len(g),"n_known":int(known.sum()),
                    "ipcw_brier":float(np.sum(w*(y-p)**2)/len(g)),
                    "calibration_intercept":intercept,"calibration_slope":slope,
                    "mean_predicted_cif":float(p.mean()),
                    "weighted_observed_rate":float(np.sum(w*y)/np.sum(w)) if w.sum() else np.nan})
       # Weighted quantile bins; ties can reduce actual bin count.
       order=np.argsort(p); cum=np.cumsum(w[order]); total=cum[-1] if len(cum) else 0
       if total<=0: continue
       # Assign equal-count rows by rank, then calculate IPCW-weighted summaries.
       rank=np.empty(len(g),dtype=int); rank[order]=np.minimum((np.arange(len(g))*args.bins)//max(len(g),1),args.bins-1)
       for b in range(args.bins):
        ix=rank==b
        if not ix.any(): continue
        sw=w[ix].sum()
        bins_out.append({"seed":seed,"landmark":lm,"test":part,"horizon_days":h,
                         "cause":cause,"industry":industry,"bin":b+1,"n":int(ix.sum()),
                         "mean_predicted":float(np.average(p[ix],weights=np.ones(ix.sum()))),
                         "weighted_observed":float(np.sum(w[ix]*y[ix])/sw) if sw else np.nan,
                         "weight_sum":float(sw)})
 pd.DataFrame(rows).to_csv(args.output_dir/"industry_calibration_metrics.csv",index=False)
 pd.DataFrame(bins_out).to_csv(args.output_dir/"industry_reliability_bins.csv",index=False)
 print("industry metric cells",len(rows),"reliability rows",len(bins_out))
if __name__=="__main__": main()
