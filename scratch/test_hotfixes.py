import unittest
import pandas as pd
from app import load_synthetic_hospital_data
from flow_view import render_patient_flow
from staff_resources_view import render_staff_resources
from bottleneck_engine import evaluate_staff_workload_imbalance, StaffDepartmentWorkload, StaffRebalanceRecommendation
from forecast_engine import generate_historical_arrival_dataset

class TestHotfixes(unittest.TestCase):
    def setUp(self):
        self.summary, self.df_arrivals, self.df_departments, self.df_resources = load_synthetic_hospital_data()

    def test_flow_history_columns(self):
        df_history = generate_historical_arrival_dataset(start_date="2026-08-18", num_days=7, seed=42)
        self.assertIn("timestamp", df_history.columns)
        self.assertIn("total_arrivals", df_history.columns)
        recent_hist = df_history.iloc[-24:].copy()
        # Verify timestamps can be safely retrieved
        self.assertTrue("timestamp" in recent_hist.columns or "datetime" in recent_hist.columns)
        timestamps = recent_hist["timestamp"].tolist()
        self.assertEqual(len(timestamps), 24)

    def test_staff_workload_attributes(self):
        workloads, recs = evaluate_staff_workload_imbalance(self.df_departments, self.summary)
        for w in workloads:
            # Both direct and property access should work
            self.assertTrue(hasattr(w, "workload_per_staff"))
            self.assertTrue(hasattr(w, "workload_to_staff_ratio"))
            self.assertEqual(w.workload_to_staff_ratio, w.workload_per_staff)
            self.assertEqual(w.staff_count, w.staff_on_duty)
            self.assertEqual(w.occupied_beds, w.occupied)
            self.assertEqual(w.capacity_beds, w.capacity)
            self.assertEqual(w.acuity_multiplier, w.acuity_weight)

        for r in recs:
            self.assertTrue(hasattr(r, "staff_to_move"))
            self.assertTrue(hasattr(r, "nurses_to_rebalance"))
            self.assertEqual(r.nurses_to_rebalance, r.staff_to_move)
            self.assertEqual(r.rationale, r.clinical_advisory)
            self.assertEqual(r.qualification_match, r.staff_qualification)

if __name__ == "__main__":
    unittest.main()
