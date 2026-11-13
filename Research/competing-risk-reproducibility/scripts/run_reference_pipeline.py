#!/usr/bin/env python3
"""
Reference reproduction pipeline for personalized discrete-time competing-risk forecasting.

This is a transparent reimplementation of the documented methodology, NOT a byte-for-byte
recovery of the historical experiment code (which was not present in the result archives).

Input: extracted Synthetic Sales Simulator v0.4 directory containing seed_*/ tables.
Output: per-seed metrics and run metadata. No original repository files are touched.
"""
from __future__ import annotations
import argparse, json, platform, hashlib
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

HORIZONS = [90, 180, 365]
INTERVAL_EDGES = list(range(0, 361, 30)) + [365]
LANDMARKS = {"day_0": 0, "day_30": 30, "day_60": 60}
NUMERIC = [
    "fit_alignment_score", "business_need_score", "budget_readiness_score",
    "competitive_intensity_score", "purchase_urgency_score", "rep_experience_months",
    "discount_requested_pct", "initial_amount", "contract_term_months", "amount_as_of",
    "age_days", "activity_count_to_date", "call_count_to_date", "email_count_to_date",
    "meeting_count_to_date", "demo_count_to_date", "response_rate_to_date",
    "days_since_last_activity", "stage_transition_count_to_date",
    "buyer_stakeholders_identified", "buyer_stakeholders_engaged",
    "economic_buyers_identified",
]
CATEGORICAL = ["initial_source", "initial_segment", "currency", "stage_as_of", "industry"]

def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""): h.update(block)
    return h.hexdigest()

def parse_utc(s):
    return pd.to_datetime(s, utc=True, errors="coerce")

def load_seed(seed_dir: Path):
    feat=pd.read_csv(seed_dir/"point_in_time_features.csv")
    opp=pd.read_csv(seed_dir/"opportunities.csv")
    split=pd.read_csv(seed_dir/"split_manifest.csv")
    companies=pd.read_csv(seed_dir/"companies.csv")[["company_id","industry"]]
    feat=feat.merge(opp[["opportunity_id","created_at","final_outcome","final_close_at",
                         "observation_cutoff_at","is_censored"]],on="opportunity_id",how="left",
                    validate="many_to_one",suffixes=("","_opp"))
    feat=feat.merge(companies,on="company_id",how="left",validate="many_to_one")
    feat=feat.merge(split[["opportunity_id","split","group_id"]],on="opportunity_id",
                    how="left",validate="many_to_one")
    for c in ["as_of_at","created_at","final_close_at","observation_cutoff_at"]:
        feat[c+"_dt"]=parse_utc(feat[c])
    if feat["split"].isna().any(): raise ValueError("Some feature rows have no split assignment.")
    if feat.duplicated(["opportunity_id","landmark"]).any():
        raise ValueError("Duplicate opportunity/landmark feature rows.")
    return feat

def build_person_period(df: pd.DataFrame, landmark: str):
    """Build interval rows. Event interval is included; censored intervals only if fully observed."""
    x=df[(df.landmark==landmark) & (df.scoring_eligible==True)].copy()
    x=x[x.split=="within_company_train_dev"].copy()
    rows=[]
    edges=INTERVAL_EDGES
    for r in x.itertuples(index=False):
        asof=getattr(r,"as_of_at_dt")
        close=getattr(r,"final_close_at_dt")
        cutoff=getattr(r,"observation_cutoff_at_dt")
        outcome=getattr(r,"final_outcome")
        if pd.isna(asof) or pd.isna(cutoff): continue
        end_days=(close-asof).total_seconds()/86400 if pd.notna(close) else np.inf
        censor_days=(cutoff-asof).total_seconds()/86400
        if end_days < 0: continue
        event=outcome in ("won","lost") and end_days <= 365
        followup=min(end_days,censor_days,365) if event else min(censor_days,365)
        for j,(lo,hi) in enumerate(zip(edges[:-1],edges[1:])):
            if event and lo <= end_days <= hi:
                row=make_row(r,j,"won" if outcome=="won" else "lost")
                rows.append(row); break
            if hi <= followup:
                rows.append(make_row(r,j,"none"))
            else:
                break
    return pd.DataFrame(rows)

def make_row(r, interval, target):
    d={c:getattr(r,c,None) for c in NUMERIC+CATEGORICAL}
    d["interval_index"]=interval
    d["target"]=target
    return d

def make_model(active_numeric):
    numeric_pipe=Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler())])
    categorical_pipe=Pipeline([("impute",SimpleImputer(strategy="most_frequent")),
                               ("onehot",OneHotEncoder(handle_unknown="ignore"))])
    prep=ColumnTransformer([("num",numeric_pipe,active_numeric),
                            ("cat",categorical_pipe,CATEGORICAL),
                            ("interval",OneHotEncoder(handle_unknown="ignore"),["interval_index"])])
    return Pipeline([("prep",prep),("model",LogisticRegression(
        max_iter=1500, C=1.0, solver="lbfgs"))])

def predict_cif(model, feature_rows, active_numeric):
    """Return arrays of cumulative incidence at the configured horizons."""
    x=feature_rows[active_numeric+CATEGORICAL].copy()
    surv=np.ones(len(x)); cif_w=np.zeros(len(x)); cif_l=np.zeros(len(x))
    out={}
    for j,(lo,hi) in enumerate(zip(INTERVAL_EDGES[:-1],INTERVAL_EDGES[1:])):
        z=x.copy(); z["interval_index"]=j
        p=model.predict_proba(z)
        classes=list(model.named_steps["model"].classes_)
        pnone=p[:,classes.index("none")] if "none" in classes else np.zeros(len(x))
        pw=p[:,classes.index("won")] if "won" in classes else np.zeros(len(x))
        pl=p[:,classes.index("lost")] if "lost" in classes else np.zeros(len(x))
        cif_w += surv*pw; cif_l += surv*pl
        surv *= pnone
        for h in HORIZONS:
            if hi==h: out[h]=(cif_w.copy(),cif_l.copy(),surv.copy())
    return out


def censoring_km(reference_rows):
    """Kaplan-Meier estimate of G(t)=P(C>=t), treating sales events as censored."""
    a=reference_rows["as_of_at_dt"]
    close=reference_rows["final_close_at_dt"]
    cutoff=reference_rows["observation_cutoff_at_dt"]
    close_days=(close-a).dt.total_seconds()/86400
    cutoff_days=(cutoff-a).dt.total_seconds()/86400
    observed_event=close.notna() & (close_days >= 0) & (close_days <= cutoff_days)
    censor_time=np.where(observed_event, close_days, cutoff_days)
    censor_event=(~observed_event).astype(int)
    valid=np.isfinite(censor_time) & (censor_time >= 0)
    times=np.asarray(censor_time[valid],dtype=float)
    events=np.asarray(censor_event[valid],dtype=int)
    unique=np.sort(np.unique(times[events==1]))
    surv=1.0
    km_times=[]; km_before=[]; km_after=[]
    for t in unique:
        risk=int(np.sum(times >= t))
        d=int(np.sum((times == t) & (events == 1)))
        before=surv
        if risk:
            surv *= (1.0-d/risk)
        km_times.append(float(t)); km_before.append(float(before)); km_after.append(float(surv))
    return {"times":np.asarray(km_times),"before":np.asarray(km_before),
            "after":np.asarray(km_after),"n":int(valid.sum())}

def censor_survival(km, t, left_limit=False):
    """Evaluate G(t); left_limit=True returns G(t-) for observed event cases."""
    times=km["times"]
    side="left" if left_limit else "right"
    idx=np.searchsorted(times,t,side=side)-1
    return 1.0 if idx < 0 else float(km["after"][idx])

def ipcw_competing_brier(test, prediction, cause, horizon, km, g_floor=0.05):
    """Cause-specific cumulative-incidence Brier score with train-estimated censoring KM."""
    a=test["as_of_at_dt"]
    close=test["final_close_at_dt"]
    cutoff=test["observation_cutoff_at_dt"]
    close_days=(close-a).dt.total_seconds().to_numpy()/86400
    cutoff_days=(cutoff-a).dt.total_seconds().to_numpy()/86400
    outcome=test["final_outcome"].astype(str).str.lower().to_numpy()
    event_observed=np.isfinite(close_days) & (close_days >= 0) & (close_days <= cutoff_days)
    event_by_h=(event_observed & (close_days <= horizon))
    event_free_h=(~event_by_h) & (cutoff_days >= horizon)
    known=event_by_h | event_free_h
    y=((outcome == cause) & event_by_h).astype(float)
    weights=np.zeros(len(test),dtype=float)
    for i in np.where(event_by_h)[0]:
        weights[i]=1.0/max(censor_survival(km,float(close_days[i]),left_limit=True),g_floor)
    for i in np.where(event_free_h)[0]:
        weights[i]=1.0/max(censor_survival(km,float(horizon)),g_floor)
    score=float(np.sum(weights*(y-np.asarray(prediction))**2)/len(test))
    return {"ipcw_brier":score,"n_test":int(len(test)),"n_known":int(known.sum()),
            "n_censored_before_horizon":int((~known).sum()),
            "max_ipcw_weight":float(weights.max()),"mean_ipcw_weight":float(weights[weights>0].mean()) if np.any(weights>0) else None,
            "g_at_horizon":float(censor_survival(km,float(horizon)))}

def evaluate_landmark(df, landmark, output_dir):
    train=df[(df.landmark==landmark)&(df.scoring_eligible==True)&
             (df.split=="within_company_train_dev")].copy()
    train_pp=build_person_period(df,landmark)
    train_opp=train.drop_duplicates("opportunity_id").copy()
    km=censoring_km(train_opp)
    if train_pp.empty: raise RuntimeError(f"No person-period training rows for {landmark}")
    active_numeric=[c for c in NUMERIC if train_pp[c].notna().any()]
    model=make_model(active_numeric)
    model.fit(train_pp[active_numeric+CATEGORICAL+["interval_index"]],train_pp["target"])
    metrics=[]
    prediction_frames=[]
    for test_name in ["within_company_future_test","unseen_company_test"]:
        test=df[(df.landmark==landmark)&(df.scoring_eligible==True)&
                (df.split==test_name)].copy()
        if test.empty: continue
        preds=predict_cif(model,test,active_numeric)
        base_cols=["opportunity_id","company_id","account_id","group_id","split",
                   "landmark","as_of_at","final_outcome","final_close_at",
                   "observation_cutoff_at","is_censored"]
        base=test[[c for c in base_cols if c in test.columns]].copy()
        base["prediction_test_partition"]=test_name
        base["prediction_model"]="multinomial_discrete_time_logistic_reference_v1"
        for h,(pw,pl,ps) in preds.items():
            pred_frame=base.copy()
            pred_frame["horizon_days"]=h
            pred_frame["predicted_cif_won"]=pw
            pred_frame["predicted_cif_lost"]=pl
            pred_frame["predicted_event_free_probability"]=ps
            pred_frame["cif_sum"]=pw+pl
            pred_frame["prediction_timestamp_utc"]=datetime.now(timezone.utc).isoformat()
            prediction_frames.append(pred_frame)
            for cause,pred in [("won",pw),("lost",pl)]:
                score=ipcw_competing_brier(test,cause=cause,prediction=pred,horizon=h,km=km)
                metrics.append({
                    "landmark":landmark,"test":test_name,"horizon_days":h,"cause":cause,
                    **score,
                    "mean_predicted_cif":float(np.mean(pred)),
                })
            # Mathematical invariants for every opportunity at each horizon.
            if np.any(pw < -1e-10) or np.any(pl < -1e-10):
                raise AssertionError("Negative cumulative incidence.")
            if np.any(pw > 1+1e-10) or np.any(pl > 1+1e-10) or np.any(pw+pl > 1+1e-8):
                raise AssertionError("Cumulative incidence probability bounds violated.")
    predictions=pd.concat(prediction_frames,ignore_index=True) if prediction_frames else pd.DataFrame()
    return model,pd.DataFrame(metrics),len(train_pp),predictions

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data-dir",type=Path,required=True,
                    help="Directory containing seed_20261121 etc.")
    ap.add_argument("--output-dir",type=Path,default=Path("repro_output"))
    ap.add_argument("--seed",type=int,default=None,help="Optional single seed.")
    ap.add_argument("--landmarks",nargs="+",choices=list(LANDMARKS),default=list(LANDMARKS))
    args=ap.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    seed_dirs=sorted(args.data_dir.glob("seed_*"))
    if args.seed is not None: seed_dirs=[p for p in seed_dirs if p.name==f"seed_{args.seed}"]
    all_metrics=[]; all_predictions=[]; run_meta=[]
    for sd in seed_dirs:
        seed=int(sd.name.split("_")[-1])
        frame=load_seed(sd)
        for landmark in args.landmarks:
            model,metrics,npp,predictions=evaluate_landmark(frame,landmark,args.output_dir)
            metrics.insert(0,"seed",seed); all_metrics.append(metrics)
            if not predictions.empty:
                predictions.insert(0,"seed",seed)
                all_predictions.append(predictions)
            run_meta.append({"seed":seed,"landmark":landmark,"person_period_rows":npp,
                             "train_opportunities":int(((frame.landmark==landmark)&
                              (frame.split=="within_company_train_dev")&
                              (frame.scoring_eligible==True)).sum())})
    result=pd.concat(all_metrics,ignore_index=True) if all_metrics else pd.DataFrame()
    result.to_csv(args.output_dir/"reference_metrics.csv",index=False)
    prediction_table=pd.concat(all_predictions,ignore_index=True) if all_predictions else pd.DataFrame()
    if not prediction_table.empty:
        if prediction_table.duplicated(["seed","opportunity_id","landmark","horizon_days"]).any():
            raise AssertionError("Duplicate prediction keys.")
        if not np.allclose(prediction_table["cif_sum"],
                           prediction_table["predicted_cif_won"]+prediction_table["predicted_cif_lost"]):
            raise AssertionError("Stored CIF sum does not match cause-specific CIFs.")
        prediction_table.to_csv(args.output_dir/"opportunity_predictions.csv",index=False)
    metadata={"created_utc":datetime.now(timezone.utc).isoformat(),
              "python":platform.python_version(),"numpy":np.__version__,
              "pandas":pd.__version__,"scikit_learn":sklearn.__version__,
              "pipeline_status":"reference_reimplementation_not_exact_historical_reproduction",
              "interval_edges_days":INTERVAL_EDGES,"horizons_days":HORIZONS,
              "model":"sklearn multinomial LogisticRegression C=1.0, max_iter=1500",
              "prediction_rows":int(len(prediction_table)),
              "prediction_key":["seed","opportunity_id","landmark","horizon_days"],
              "prediction_file":"opportunity_predictions.csv",
              "run_details":run_meta}
    (args.output_dir/"run_metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    print(result.to_string(index=False))
    print(f"\nWrote {args.output_dir/'reference_metrics.csv'}")

if __name__=="__main__": main()
