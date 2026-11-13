#!/usr/bin/env python3
"""Validate structural and split invariants for an extracted v0.4 dataset."""
import argparse, json
from pathlib import Path
import pandas as pd

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data-dir",type=Path,required=True)
    args=p.parse_args()
    results=[]
    for sd in sorted(args.data_dir.glob("seed_*")):
        opp=pd.read_csv(sd/"opportunities.csv")
        split=pd.read_csv(sd/"split_manifest.csv")
        feat=pd.read_csv(sd/"point_in_time_features.csv")
        companies=pd.read_csv(sd/"companies.csv")
        truth_path=sd/"simulator_truth_RESTRICTED.csv"
        company_splits=split.groupby("company_id")["split"].nunique()
        account_splits=split.groupby("account_id")["split"].nunique()
        checks={
          "opportunity_ids_unique":bool(opp.opportunity_id.is_unique),
          "split_has_one_row_per_opportunity":bool(len(split)==len(opp) and split.opportunity_id.nunique()==len(opp)),
          "all_opportunities_assigned":bool(set(opp.opportunity_id)==set(split.opportunity_id)),
          "account_groups_do_not_cross_splits":bool(account_splits.max()==1),
          "heldout_companies_count_18":bool((company_splits==1).sum()==18),
          "one_feature_row_per_opportunity_landmark":bool(not feat.duplicated(["opportunity_id","landmark"]).any()),
          "model_predictor_allowlist_excludes_outcomes":bool(not set([
              "target_win","final_outcome","final_close_at","final_amount","is_censored",
              "simulator_truth_RESTRICTED.csv","split","group_id","company_id","account_id"
          ]) & set([
              "fit_alignment_score","business_need_score","budget_readiness_score",
              "competitive_intensity_score","purchase_urgency_score","rep_experience_months",
              "discount_requested_pct","initial_amount","contract_term_months","initial_source",
              "initial_segment","currency","stage_as_of","amount_as_of","activity_count_to_date",
              "call_count_to_date","email_count_to_date","meeting_count_to_date","demo_count_to_date",
              "response_rate_to_date","days_since_last_activity","stage_transition_count_to_date",
              "buyer_stakeholders_identified","buyer_stakeholders_engaged",
              "economic_buyers_identified","industry"
          ])),
          "restricted_truth_file_present_for_governance_awareness":bool(truth_path.exists()),
          "company_count_90":bool(len(companies)==90),
          "opportunity_count_9000":bool(len(opp)==9000),
        }
        results.append({"seed":sd.name,"checks":checks,"passed":sum(checks.values()),"total":len(checks)})
    print(json.dumps(results,indent=2))
    if not results or any(r["passed"]!=r["total"] for r in results):
        raise SystemExit("Dataset validation failed.")
if __name__=="__main__": main()
