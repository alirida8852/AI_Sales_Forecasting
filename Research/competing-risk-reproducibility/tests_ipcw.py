import unittest
import numpy as np
import pandas as pd
from scripts.run_reference_pipeline import censoring_km, censor_survival, ipcw_competing_brier

class IPCWTests(unittest.TestCase):
    def test_km_no_censoring(self):
        df=pd.DataFrame({
          "as_of_at_dt":pd.to_datetime(["2024-01-01","2024-01-01"],utc=True),
          "final_close_at_dt":pd.to_datetime(["2024-02-01","2024-03-01"],utc=True),
          "observation_cutoff_at_dt":pd.to_datetime(["2025-01-01","2025-01-01"],utc=True)})
        km=censoring_km(df)
        self.assertAlmostEqual(censor_survival(km,90),1.0)

    def test_km_censor_tie_and_left_limit(self):
        df=pd.DataFrame({
          "as_of_at_dt":pd.to_datetime(["2024-01-01"]*4,utc=True),
          "final_close_at_dt":pd.to_datetime(["2024-02-01","2024-02-01",None,None],utc=True),
          "observation_cutoff_at_dt":pd.to_datetime(["2024-06-01"]*4,utc=True)})
        # two observed events at day 31, two censored at day 152; no censoring at day 31
        km=censoring_km(df)
        self.assertAlmostEqual(censor_survival(km,31,left_limit=True),1.0)
        self.assertAlmostEqual(censor_survival(km,31),1.0)

    def test_ipcw_known_cases_and_censoring(self):
        asof=pd.to_datetime(["2024-01-01"]*3,utc=True)
        test=pd.DataFrame({
          "as_of_at_dt":asof,
          "final_close_at_dt":pd.to_datetime(["2024-02-01","2024-03-01",None],utc=True),
          "observation_cutoff_at_dt":pd.to_datetime(["2025-01-01","2025-01-01","2024-02-15"],utc=True),
          "final_outcome":["won","lost","open"]})
        km={"times":np.array([45.]),"before":np.array([1.]),"after":np.array([2/3]),"n":3}
        out=ipcw_competing_brier(test,np.array([.8,.2,.1]),"won",90,km)
        self.assertEqual(out["n_known"],2)
        self.assertEqual(out["n_censored_before_horizon"],1)
        self.assertAlmostEqual(out["ipcw_brier"],((1-.8)**2+(.2)**2*1.5)/3)

if __name__=="__main__": unittest.main()
