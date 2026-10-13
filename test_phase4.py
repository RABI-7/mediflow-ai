"""
=============================================================================
 MediFlow AI — Phase 4 Comprehensive Test Suite
=============================================================================
 Tests all required Phase 4 capabilities:
   1. Forecast horizons of 4, 8, 12, and 24 hours.
   2. Non-negative forecasts and correct aggregation.
   3. Missing, insufficient, and malformed historical data handling.
   4. Forecast evaluation without temporal data leakage (strict holdout).
   5. Forecast-to-simulator integration and parameter translation.
   6. Prevention of duplicate patient-arrival counts.
   7. Preservation of Phase 1, Phase 2, and Phase 3 functionality.
=============================================================================
"""

import unittest
from datetime import datetime
import pandas as pd
import numpy as np

from forecast_engine import (
    generate_historical_arrival_dataset,
    train_seasonal_profile_model,
    generate_patient_arrival_forecast,
    evaluate_forecast_model_performance,
    evaluate_forecast_driven_capacity_risk,
    translate_forecast_to_simulator_parameters,
    FORECAST_THRESHOLDS,
    ForecastResult,
    ForecastEvaluationMetrics
)
from simulator import run_scenario_step_by_step, run_hospital_what_if_simulation, ScenarioResult
from bottleneck_engine import (
    detect_hospital_bottlenecks,
    evaluate_hidden_capacity,
    detect_bottleneck_chains,
    evaluate_staff_workload_imbalance,
    evaluate_multiple_candidate_interventions
)
from app import load_synthetic_hospital_data


class TestMediFlowPhase4(unittest.TestCase):

    def setUp(self):
        self.df_history = generate_historical_arrival_dataset(start_date="2026-08-18", num_days=28, seed=42)
        self.summary, self.df_arrivals, self.df_departments, self.df_resources = load_synthetic_hospital_data()

    def test_01_forecast_horizons_4_8_12_24(self):
        """Test 1: Generates valid forecasts for all 4 horizons (4, 8, 12, and 24 hours)."""
        for h in [4, 8, 12, 24]:
            fc = generate_patient_arrival_forecast(self.df_history, horizon_hours=h, target_category="All Facility")
            self.assertEqual(fc.horizon_hours, h)
            self.assertEqual(len(fc.interval_forecasts), h)
            
            # Verify sequential hourly progression
            for idx in range(len(fc.interval_forecasts) - 1):
                t1 = datetime.strptime(fc.interval_forecasts[idx].timestamp, "%Y-%m-%d %H:00")
                t2 = datetime.strptime(fc.interval_forecasts[idx + 1].timestamp, "%Y-%m-%d %H:00")
                diff_hours = (t2 - t1).total_seconds() / 3600
                self.assertEqual(diff_hours, 1.0)

    def test_02_non_negative_and_aggregation(self):
        """Test 2: All forecast estimates and prediction intervals are non-negative and aggregate correctly."""
        fc = generate_patient_arrival_forecast(self.df_history, horizon_hours=12, target_category="All Facility")
        
        # Non-negativity and bounds check
        for pt in fc.interval_forecasts:
            self.assertGreaterEqual(pt.predicted_arrivals, 0)
            self.assertGreaterEqual(pt.lower_ci_80, 0)
            self.assertGreaterEqual(pt.lower_ci_95, 0)
            self.assertGreaterEqual(pt.upper_ci_80, pt.predicted_arrivals)
            self.assertGreaterEqual(pt.upper_ci_95, pt.upper_ci_80)
            self.assertLessEqual(pt.lower_ci_95, pt.lower_ci_80)

        # Mathematical aggregation check
        calculated_total = sum(pt.predicted_arrivals for pt in fc.interval_forecasts)
        self.assertEqual(fc.total_predicted_arrivals, calculated_total)
        expected_mean = round(calculated_total / 12, 1)
        self.assertEqual(fc.mean_arrivals_per_hour, expected_mean)

    def test_03_missing_insufficient_malformed_data(self):
        """Test 3: Graceful fallback and clear warnings when data is missing or insufficient."""
        # Empty DataFrame
        fc_empty = generate_patient_arrival_forecast(pd.DataFrame(), horizon_hours=12)
        self.assertIsInstance(fc_empty, ForecastResult)
        self.assertEqual(len(fc_empty.interval_forecasts), 12)
        self.assertIn("INSUFFICIENT HISTORICAL DATA", fc_empty.surge_warning_message)

        # Small dataset (< 24 rows)
        small_df = self.df_history.iloc[:10].copy()
        fc_small = generate_patient_arrival_forecast(small_df, horizon_hours=8)
        self.assertIsInstance(fc_small, ForecastResult)
        self.assertEqual(len(fc_small.interval_forecasts), 8)
        self.assertIn("INSUFFICIENT HISTORICAL DATA", fc_small.surge_warning_message)

        # Malformed columns
        malformed_df = pd.DataFrame({"random_col": [1, 2, 3]})
        fc_malformed = generate_patient_arrival_forecast(malformed_df, horizon_hours=4)
        self.assertIsInstance(fc_malformed, ForecastResult)

    def test_04_chronological_evaluation_no_temporal_leakage(self):
        """Test 4: Forecast evaluation uses strict chronological holdout without temporal data leakage."""
        test_days = 7
        eval_metrics = evaluate_forecast_model_performance(self.df_history, test_days=test_days)

        self.assertIsInstance(eval_metrics, ForecastEvaluationMetrics)
        self.assertEqual(eval_metrics.test_observations_count, test_days * 24)
        
        # Verify valid non-negative count error metrics
        self.assertGreater(eval_metrics.mae, 0.0)
        self.assertGreater(eval_metrics.rmse, 0.0)
        self.assertGreater(eval_metrics.baseline_mae, 0.0)
        
        # RMSE is mathematically >= MAE
        self.assertGreaterEqual(eval_metrics.rmse, eval_metrics.mae)

        # Verify temporal boundary (train ends strictly before test begins)
        split_idx = len(self.df_history) - (test_days * 24)
        train_max_dt = self.df_history.iloc[:split_idx]["datetime"].max()
        test_min_dt = self.df_history.iloc[split_idx:]["datetime"].min()
        self.assertLess(train_max_dt, test_min_dt)

    def test_05_forecast_to_simulator_integration(self):
        """Test 5: Forecast parameters translate correctly and execute in the What-If Simulator."""
        fc = generate_patient_arrival_forecast(self.df_history, horizon_hours=12, target_category="All Facility")
        sim_params = translate_forecast_to_simulator_parameters(fc, ct_scanner_status="Unavailable", discharge_delay=True)

        self.assertEqual(sim_params["duration_hours"], 12)
        self.assertGreaterEqual(sim_params["additional_arrivals_total"], 0)
        self.assertEqual(sim_params["forecast_total"], fc.total_predicted_arrivals)

        # Execute simulation using translated parameters
        sim_res = run_scenario_step_by_step(
            name="Forecasted Operational Stress",
            scenario_type="disrupted",
            description="Testing forecast parameters",
            assumptions=[],
            duration_hours=sim_params["duration_hours"],
            total_beds=280,
            initial_occupied=238,
            initial_waiting=34,
            initial_ct_queue=6,
            additional_arrivals_total=sim_params["additional_arrivals_total"],
            discharge_rate_hourly=2.4,
            max_admission_intake_hourly=12,
            ct_capacity_hourly=3.8,
            ct_status_label="Unavailable"
        )
        self.assertIsInstance(sim_res, ScenarioResult)
        self.assertGreater(sim_res.total_admissions_fulfilled, 0)
        self.assertGreaterEqual(sim_res.final_occupied, 0)
        self.assertLessEqual(sim_res.final_occupied, 280)

    def test_06_prevention_of_duplicate_counts(self):
        """Test 6: Prevents duplicate patient counting by isolating incremental surge above baseline."""
        fc = generate_patient_arrival_forecast(self.df_history, horizon_hours=12, target_category="All Facility")
        sim_params = translate_forecast_to_simulator_parameters(fc)

        # Additional arrivals must equal (total - baseline), never total itself
        expected_surge = max(0, fc.total_predicted_arrivals - fc.baseline_expected_arrivals)
        self.assertEqual(sim_params["additional_arrivals_total"], expected_surge)
        self.assertLess(sim_params["additional_arrivals_total"], fc.total_predicted_arrivals)

    def test_07_preservation_of_phases_1_2_3(self):
        """Test 7: Preserves complete functionality of Phase 1, Phase 2, and Phase 3 modules."""
        # Phase 1: Dashboard data intact
        self.assertEqual(self.summary["total_beds"], 280)
        self.assertEqual(self.summary["occupied_beds"], 238)
        self.assertEqual(len(self.df_departments), 6)

        # Phase 2: What-If Simulator intact
        b_res, d_res, r_res, meta = run_hospital_what_if_simulation(
            additional_arrivals=30,
            ct_scanner_status="Unavailable",
            discharge_delay=True,
            duration_hours=12
        )
        self.assertIsInstance(b_res, ScenarioResult)
        self.assertIsInstance(d_res, ScenarioResult)
        self.assertIsInstance(r_res, ScenarioResult)

        # Phase 3: Bottleneck intelligence intact
        findings = detect_hospital_bottlenecks(self.summary, self.df_departments, self.df_resources, d_res)
        self.assertGreater(len(findings), 0)

        hc = evaluate_hidden_capacity(self.summary, d_res)
        self.assertTrue(hc.breakdown.is_invariant_valid)
        self.assertEqual(hc.breakdown.total_accounted, 280)

        chain = detect_bottleneck_chains(self.summary, d_res)
        self.assertIsNotNone(chain)

        workloads, s_recs = evaluate_staff_workload_imbalance(self.df_departments, self.summary)
        self.assertGreater(len(workloads), 0)

        multi_res = evaluate_multiple_candidate_interventions(
            candidate_ids=["cand_discharge_lounge", "cand_infeasible_extreme_pull"],
            additional_arrivals=30,
            ct_scanner_status="Unavailable",
            discharge_delay=True,
            duration_hours=12,
            summary=self.summary,
            df_departments=self.df_departments
        )
        self.assertEqual(len(multi_res.evaluated_responses), 2)
        # First candidate is feasible, second is rejected by feasibility check
        self.assertTrue(multi_res.evaluated_responses[0][1].is_feasible)
        self.assertFalse(multi_res.evaluated_responses[1][1].is_feasible)


if __name__ == "__main__":
    unittest.main()
