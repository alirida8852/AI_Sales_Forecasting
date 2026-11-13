#!/usr/bin/env python3
"""Training-only group-held-out calibration experiment for competing-risk CIFs."""
import argparse,sys,json
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from scipy.special import logit
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from run_reference_pipeline import (load_seed,build_person_period,make_model,NUMERIC,CATEGORICAL,
    predict_cif,censoring_km,censor_survival)

def ipcw(test,cause,h,km,floor=.05):
 a=test.as_of_at_dt; close=test.final_close_at_dt; cutoff=test.observation_cutoff_at_dt
 cd=(close-a).dt.total_seconds().to_numpy()/86400
 td=(cutoff-a).dt.total_seconds().to_numpy()/86400
 outcome=test.final_outcome.astype(str).str.lower().to_numpy()
 ev=np.isfinite(cd)&(cd>=0)&(cd<=td)&(cd<=h)
 free=(~ev)&(td>=h); known=ev|free
 y=((outcome==cause)&ev).astype(float); w=np.zeros(len(test))
 for i in np.where(ev)[0]: w[i]=1/max(censor_survival(km,float(cd[i]),left_limit=True),floor)
 for i in np.where(free)[0]: w[i]=1/max(censor_survival(km,float(h)),floor)
 return y,w,known

def fit_calibrator(method,p,y,w):
 valid=(w>0)&np.isfinite(p)
 p=np.clip(p[valid],1e-6,1-1e-6); y=y[valid]; w=w[valid]
 if len(p)<30 or len(np.unique(y))<2: return None
 if method=="sigmoid":
  z=logit(p).reshape(-1,1)
  m=LogisticRegression(C=1e6,max_iter=2000)
  m.fit(z,y,sample_weight=w)
  return ("sigmoid",m)
 if method=="isotonic":
  m=IsotonicRegression(out_of_bounds="clip")
  m.fit(p,y,sample_weight=w)
  return ("isotonic",m)
 return None

def apply_calibrator(cal,p):
 if cal is None: return np.asarray(p)
 kind,m=cal
 p=np.clip(np.asarray(p),1e-6,1-1e-6)
 if kind=="sigmoid": return m.predict_proba(logit(p).reshape(-1,1))[:,1]
 return m.predict(p)

def score(test,cause,h,p,km):
 y,w,known=ipcw(test,cause,h,km)
 return float(np.sum(w*(y-p)**2)/len(test)),int(known.sum())

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument("--data-dir",type=Path,required=True)
 ap.add_argument("--predictions",type=Path,required=True)
 ap.add_argument("--output-dir",type=Path,required=True)
 ap.add_argument("--calibration-fraction",type=float,default=.20)
 ap.add_argument("--random-state",type=int,default=202611)
 args=ap.parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True)
 prior=pd.read_csv(args.predictions)
 metric_rows=[]; split_rows=[]; calrows=[]; eval_rows=[]
 for sd in sorted(args.data_dir.glob("seed_*")):
  seed=int(sd.name.split("_")[-1]); df=load_seed(sd)
  for lm in ["day_0","day_30","day_60"]:
   base=df[(df.landmark==lm)&(df.scoring_eligible==True)&(df.split=="within_company_train_dev")].copy()
   # Group split: all snapshots/opportunities belonging to one account stay together.
   groups=base.drop_duplicates("group_id")[["group_id"]].copy()
   gss=GroupShuffleSplit(n_splits=1,test_size=args.calibration_fraction,random_state=args.random_state+seed%1000+int(lm[-1]))
   fit_idx,cal_idx=next(gss.split(groups,groups=groups.group_id))
   fit_groups=set(groups.iloc[fit_idx].group_id); cal_groups=set(groups.iloc[cal_idx].group_id)
   fit=base[base.group_id.isin(fit_groups)].copy()
   cal=base[base.group_id.isin(cal_groups)].copy()
   if fit.group_id.isin(cal_groups).any(): raise RuntimeError("Account group leakage between fit and calibration.")
   pp=build_person_period(df[df.group_id.isin(fit_groups)],lm)
   active=[c for c in NUMERIC if c in pp and pp[c].notna().any()]
   model=make_model(active)
   model.fit(pp[active+CATEGORICAL+["interval_index"]],pp.target)
   fit_opp=fit.drop_duplicates("opportunity_id")
   km=censoring_km(fit_opp)
   cal=cal.drop_duplicates("opportunity_id")
   cp=predict_cif(model,cal,active)
   test=prior[(prior.seed==seed)&(prior.landmark==lm)]
   # Generate test predictions from the SAME fit-only model used to train calibrators.
   # This avoids applying a calibrator learned from a reduced-fit model to predictions
   # produced by the earlier full-train reference model.
   test_feature_rows={}
   for part, split_name in [("within_company_future_test","within_company_future_test"),
                            ("unseen_company_test","unseen_company_test")]:
    test_feature_rows[part]=df[(df.landmark==lm)&(df.scoring_eligible==True)&(df.split==split_name)].copy()
   # For each cause/horizon, fit on calibration partition and assess test predictions.
   split_rows.append({"seed":seed,"landmark":lm,"fit_opportunities":fit.opportunity_id.nunique(),
      "calibration_opportunities":cal.opportunity_id.nunique(),"fit_account_groups":len(fit_groups),
      "calibration_account_groups":len(cal_groups),"group_overlap":len(fit_groups&cal_groups)})
   for h,(cw,cl,cs) in cp.items():
    for cause,pc in [("won",cw),("lost",cl)]:
     y,w,known=ipcw(cal,cause,h,km)
     calfit={}
     for method in ["sigmoid","isotonic"]:
      fitted=fit_calibrator(method,pc,y,w); calfit[method]=fitted
      calrows.append({"seed":seed,"landmark":lm,"horizon_days":h,"cause":cause,
         "method":method,"n_calibration":len(cal),"n_known":int(known.sum()),
         "weight_sum":float(w.sum()),"fitted":fitted is not None})
     # Original reference held-out predictions; only calibration mapping changes.
     for part in ["within_company_future_test","unseen_company_test"]:
      sub=test_feature_rows[part].copy()
      test_cif=predict_cif(model,sub,active)[h]
      original=test_cif[0] if cause=="won" else test_cif[1]
      # Reference metric recomputed with fit-only censoring KM for consistent evaluation.
      for method in ["none","sigmoid","isotonic"]:
       calobj=None if method=="none" else calfit[method]
       calibrated=apply_calibrator(calobj,original)
       ytest,wtest,knowntest=ipcw(sub,cause,h,km)
       brier,nknown=score(sub,cause,h,calibrated,km)
       for k in range(len(sub)):
        eval_rows.append({"seed":seed,"landmark":lm,"test":part,"horizon_days":h,
          "cause":cause,"method":method,"opportunity_id":sub.opportunity_id.iloc[k],
          "group_id":sub.group_id.iloc[k],"prediction":float(calibrated[k]),
          "target":float(ytest[k]),"ipcw_weight":float(wtest[k])})
       metric_rows.append({"seed":seed,"landmark":lm,"test":part,"horizon_days":h,
          "cause":cause,"method":method,"n_test":len(sub),"n_known":nknown,
          "ipcw_brier":brier,"mean_prediction":float(np.mean(calibrated)),
          "cif_sum_exceed_rate":np.nan})
   # CIF joint probability check using separately calibrated cause mappings.
   for h,(cw,cl,cs) in cp.items():
    for part in ["within_company_future_test","unseen_company_test"]:
     sub=test_feature_rows[part]
     test_cif=predict_cif(model,sub,active)[h]
     pw=test_cif[0]; pl=test_cif[1]
     # mappings were fit for corresponding cause above; retain by fitting again from cal rows
     y,w,_=ipcw(cal,"won",h,km); sw=fit_calibrator("sigmoid",cw,y,w); iw=fit_calibrator("isotonic",cw,y,w)
     y,w,_=ipcw(cal,"lost",h,km); sl=fit_calibrator("sigmoid",cl,y,w); il=fit_calibrator("isotonic",cl,y,w)
     for method,a,b in [("none",pw,pl),("sigmoid",apply_calibrator(sw,pw),apply_calibrator(sl,pl)),
                        ("isotonic",apply_calibrator(iw,pw),apply_calibrator(il,pl))]:
      ix=(metric_rows and next((i for i,r in enumerate(metric_rows) if r["seed"]==seed and r["landmark"]==lm and r["test"]==part and r["horizon_days"]==h and r["cause"]=="won" and r["method"]==method),None))
      if ix is not None: metric_rows[ix]["cif_sum_exceed_rate"]=float(np.mean(a+b>1+1e-10))
 pd.DataFrame(metric_rows).to_csv(args.output_dir/"calibration_test_metrics.csv",index=False)
 pd.DataFrame(eval_rows).to_csv(args.output_dir/"calibration_test_predictions.csv",index=False)
 pd.DataFrame(split_rows).to_csv(args.output_dir/"calibration_group_splits.csv",index=False)
 pd.DataFrame(calrows).to_csv(args.output_dir/"calibrator_fit_diagnostics.csv",index=False)
 m=pd.DataFrame(metric_rows)
 summary=m.groupby(["landmark","test","horizon_days","cause","method"]).agg(
  mean_ipcw_brier=("ipcw_brier","mean"),sd_ipcw_brier=("ipcw_brier","std"),
  mean_cif_sum_exceed_rate=("cif_sum_exceed_rate","mean"),
  mean_prediction=("mean_prediction","mean")).reset_index()
 summary.to_csv(args.output_dir/"calibration_test_summary.csv",index=False)
 print(summary.to_string(index=False))
if __name__=="__main__": main()
