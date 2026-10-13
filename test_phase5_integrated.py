"""
=============================================================================
 MediFlow AI — Phase 5 Integrated Decision Intelligence Test Suite
=============================================================================
 Tests:
 1. Occupancy and capacity calculations (division-by-zero, negative counts).
 2. Scenario comparisons using identical baselines and time horizons.
 3. Feasibility constraints (staff floors, physical licensed caps, equipment).
 4. Bed recovery conditions and hidden capacity partition invariants.
 5. Staff availability, cross-unit qualifications, and floor preservation.
 6. Forecast integration and duplicate-arrival prevention.
 7. Recommendation explanations, trade-offs, and evidence-quality labels.
 8. Defensive handling of missing, negative, or invalid input data.
 9. Human review governance, status transitions, and audit records.
 10. End-to-end 7-step pipeline execution and integrity.
=============================================================================
"""

import unittest
import pandas as pd
import numpy as np

from decision_engine import (
    validate_hospital_state,
    run_integrated_decision_pipeline,
    ReviewStatus,
    HumanReviewRecord,
    IntegratedDecisionPipelineResult
)
from simulator import run_scenario_step_by_step, run_hospital_what_if_simulation
from forecast_engine import (
    generate_historical_arrival_dataset,
    generate_patient_arrival_forecast,
    translate_forecast_to_simulator_parameters
)
from bottleneck_engine import (
    evaluate_hidden_capacity,
    detect_bottleneck_chains,
    evaluate_staff_workload_imbalance,
    THRESHOLDS
)
from app import load_synthetic_hospital_data


class TestPhase5Integrated(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary, cls.df_arrivals, cls.df_departments, cls.df_resources = load_synthetic_hospital_data()

    def test_01_occupancy_and_capacity_calculations(self):
        """Test occupancy calculation and division-by-zero protection."""
        # Standard case
        occ_pct = round((self.summary["occupied_beds"] / max(1, self.summary["total_beds"])) * 100, 1)
        self.assertEqual(occ_pct, 85.0)

        # Zero total beds (division-by-zero guard)
        zero_summary = self.summary.copy()
        zero_summary["total_beds"] = 0
        safe_pct = round((zero_summary["occupied_beds"] / max(1, zero_summary["total_beds"])) * 100, 1)
        self.assertGreaterEqual(safe_pct, 0.0)

        # Negative count guards
        sim_res = run_scenario_step_by_step(
            name="Test Non-negative",
            scenario_type="baseline",
            description="",
            assumptions=[],
            duration_hours=4,
            total_beds=280,
            initial_occupied=238,
            initial_waiting=34,
            initial_ct_queue=6,
            additional_arrivals_total=0,
            discharge_rate_hourly=10.0,
            max_admission_intake_hourly=12,
            ct_capacity_hourly=8.0,
            ct_status_label="Available"
        )
        self.assertGreaterEqual(sim_res.final_occupied, 0)
        self.assertGreaterEqual(sim_res.final_available, 0)
        self.assertGreaterEqual(sim_res.final_waiting, 0)
        self.assertGreaterEqual(sim_res.final_ct_queue, 0)

    def test_02_scenario_comparisons_using_same_baseline(self):
        """Verify that baseline, disrupted, and response options share exact initial conditions."""
        res = run_integrated_decision_pipeline(
            summary=self.summary,
            df_departments=self.df_departments,
            df_resources=self.df_resources,
            additional_arrivals=30,
            ct_scanner_status="Unavailable",
            discharge_delay=True,
            duration_hours=12
        )
        # Check identical initial conditions
        self.assertEqual(res.sim_baseline.initial_occupied, res.sim_disrupted.initial_occupied)
        self.assertEqual(res.sim_baseline.initial_waiting, res.sim_disrupted.initial_waiting)

        for ev in res.ranked_evaluations:
            if ev.is_feasible and ev.sim_result:
                self.assertEqual(ev.sim_result.initial_occupied, res.sim_disrupted.initial_occupied)
                self.assertEqual(ev.sim_result.initial_waiting, res.sim_disrupted.initial_waiting)
                # Verify delta consistency
                self.assertEqual(ev.impact.delta_waiting, ev.sim_result.final_waiting - res.sim_disrupted.final_waiting)
                self.assertEqual(ev.impact.delta_available, ev.sim_result.final_available - res.sim_disrupted.final_available)

    def test_03_feasibility_constraints(self):
        """Verify strict enforcement of staffing floors and licensed bed limits."""
        # 1. Licensed physical bed cap
        invalid_summary = self.summary.copy()
        invalid_summary["total_beds"] = 350  # Exceeds 295 cap
        valid, errs = validate_hospital_state(invalid_summary, self.df_departments, self.df_resources)
        self.assertFalse(valid)
        self.assertTrue(any("exceeds physical licensed envelope" in e for e in errs))

        # 2. Staffing floors (Pediatrics statutory floor = 3)
        peds = self.df_departments[self.df_departments["Department"] == "Pediatrics"].iloc[0]
        self.assertEqual(peds["MinStaffFloor"], 3)
        self.assertGreaterEqual(peds["Staff"], peds["MinStaffFloor"])

    def test_04_bed_recovery_conditions_and_invariants(self):
        """Verify bed state partition invariant: sum of states == total physical capacity."""
        hc = evaluate_hidden_capacity(self.summary)
        breakdown = hc.breakdown
        self.assertTrue(breakdown.is_invariant_valid, "Bed states must sum to total physical capacity.")
        self.assertEqual(breakdown.total_accounted, breakdown.total_physical_capacity)

        # Never count unready beds as ready available
        self.assertLess(breakdown.ready_available_beds, breakdown.theoretical_unoccupied)
        self.assertEqual(breakdown.recoverable_capacity, (
            breakdown.awaiting_discharge_coordination +
            breakdown.awaiting_cleaning_evs +
            breakdown.awaiting_readiness_verification
        ))

    def test_05_staff_availability_and_qualifications(self):
        """Test acuity-weighted workload calculations and cross-unit eligibility."""
        workloads, rebalances = evaluate_staff_workload_imbalance(self.df_departments, self.summary)
        self.assertEqual(len(workloads), 6)

        for w in workloads:
            expected_units = round(w.occupied * w.acuity_weight, 1)
            self.assertEqual(w.workload_units, expected_units)
            self.assertGreaterEqual(w.available_float_staff, 0)
            self.assertGreaterEqual(w.staff_on_duty, w.min_staff_floor)

        for r in rebalances:
            self.assertTrue(r.is_floor_respected, "Rebalance must not strip donor below statutory floor.")
            self.assertGreaterEqual(r.donor_remaining_staff, r.donor_min_floor)

    def test_06_forecast_integration_and_duplicate_prevention(self):
        """Verify that arrival forecasts isolate incremental surge without double-counting."""
        df_hist = generate_historical_arrival_dataset(start_date="2026-08-18", num_days=28, seed=42)
        fc = generate_patient_arrival_forecast(df_hist, horizon_hours=12)

        # Non-negative counts
        self.assertGreater(fc.total_predicted_arrivals, 0)
        self.assertGreaterEqual(fc.incremental_surge_above_baseline, 0)

        # Duplicate-arrival prevention: incremental surge is strictly the delta above nominal baseline
        expected_surge = max(0, fc.total_predicted_arrivals - fc.baseline_expected_arrivals)
        self.assertEqual(fc.incremental_surge_above_baseline, expected_surge)

        # Check uncertainty bounds ordering: lower_95 <= lower_80 <= point <= upper_80 <= upper_95
        for pt in fc.interval_forecasts:
            self.assertLessEqual(pt.lower_ci_95, pt.lower_ci_80)
            self.assertLessEqual(pt.lower_ci_80, pt.predicted_arrivals)
            self.assertLessEqual(pt.predicted_arrivals, pt.upper_ci_80)
            self.assertLessEqual(pt.upper_ci_80, pt.upper_ci_95)

    def test_07_recommendation_explanations_and_evidence_labels(self):
        """Verify explainability fields, trade-off matrix, and evidence quality labels."""
        res = run_integrated_decision_pipeline(
            summary=self.summary,
            df_departments=self.df_departments,
            df_resources=self.df_resources,
            additional_arrivals=40,
            ct_scanner_status="Unavailable",
            discharge_delay=True,
            duration_hours=12
        )
        for s in res.available_solutions:
            self.assertIn(s.evidence_quality, ["Strong Model Evidence", "Partial Model Evidence", "Insufficient Evidence"])
            self.assertTrue(len(s.target_bottleneck) > 0)
            self.assertTrue(len(s.operational_rationale) > 0)
            self.assertTrue(len(s.assumptions) > 0)
            self.assertTrue(len(s.required_human_approval) > 0)
            self.assertTrue(len(s.clinical_priority_safeguard) > 0)
            self.assertTrue(len(s.limitations_or_missing_data) > 0)

        # Verify trade-offs generated for feasible solutions
        self.assertGreater(len(res.trade_offs), 0)
        for to in res.trade_offs:
            self.assertTrue(len(to.primary_gain) > 0)
            self.assertTrue(len(to.resources_consumed) > 0)
            self.assertTrue(len(to.potential_trade_off) > 0)
            self.assertTrue(len(to.mitigation_strategy) > 0)

    def test_08_missing_or_invalid_input_data(self):
        """Verify robust error reporting when invalid or missing data is supplied."""
        # Empty summary
        valid, errs = validate_hospital_state({}, self.df_departments, self.df_resources)
        self.assertFalse(valid)

        # Negative occupied beds
        bad_summary = self.summary.copy()
        bad_summary["occupied_beds"] = -10
        valid, errs = validate_hospital_state(bad_summary, self.df_departments, self.df_resources)
        self.assertFalse(valid)
        self.assertTrue(any("cannot be negative" in e for e in errs))

        # Occupied exceeds total
        overflow_summary = self.summary.copy()
        overflow_summary["occupied_beds"] = 300
        overflow_summary["total_beds"] = 280
        valid, errs = validate_hospital_state(overflow_summary, self.df_departments, self.df_resources)
        self.assertFalse(valid)

    def test_09_human_review_governance_and_audit_trail(self):
        """Verify human review records and advisory governance."""
        res = run_integrated_decision_pipeline(
            summary=self.summary,
            df_departments=self.df_departments,
            df_resources=self.df_resources,
            additional_arrivals=30,
            ct_scanner_status="Available",
            discharge_delay=False,
            duration_hours=12
        )
        self.assertGreater(len(res.human_reviews), 0)
        for sid, rec in res.human_reviews.items():
            self.assertIn(rec.status, ReviewStatus.ALL_STATUSES)
            self.assertTrue(rec.safeguard_notice_acknowledged)
            self.assertIn("ADVISORY DECISION SUPPORT", rec.advisory_disclaimer)

        # Update a review record
        first_sid = list(res.human_reviews.keys())[0]
        res.human_reviews[first_sid].status = ReviewStatus.APPROVED_FOR_CONSIDERATION
        res.human_reviews[first_sid].review_notes = "Approved by CMO for simulation testing."
        self.assertEqual(res.human_reviews[first_sid].status, ReviewStatus.APPROVED_FOR_CONSIDERATION)

    def test_10_end_to_end_pipeline_execution(self):
        """Verify complete execution of 7-step decision pipeline."""
        df_hist = generate_historical_arrival_dataset(start_date="2026-08-18", num_days=28, seed=42)
        fc = generate_patient_arrival_forecast(df_hist, horizon_hours=12)

        res = run_integrated_decision_pipeline(
            summary=self.summary,
            df_departments=self.df_departments,
            df_resources=self.df_resources,
            additional_arrivals=fc.incremental_surge_above_baseline,
            ct_scanner_status="Unavailable",
            discharge_delay=True,
            duration_hours=12,
            forecast=fc
        )

        # Step 1: Monitor
        self.assertEqual(res.total_beds, 280)
        self.assertGreater(res.active_occupancy_pct, 0.0)

        # Step 2: Detect
        self.assertGreater(len(res.bottlenecks), 0)
        self.assertIsNotNone(res.hidden_capacity)
        self.assertIsNotNone(res.bottleneck_chains)

        # Step 3: Forecast
        self.assertIsNotNone(res.forecast)
        self.assertGreater(res.forecast.total_predicted_arrivals, 0)

        # Step 4 & 5: Solutions & Sim
        self.assertGreater(len(res.available_solutions), 0)
        self.assertGreater(len(res.ranked_evaluations), 0)

        # Step 6: Trade-offs & Best option
        self.assertIsNotNone(res.best_feasible_option)
        self.assertGreater(len(res.trade_offs), 0)

        # Step 7: Governance
        self.assertGreater(len(res.human_reviews), 0)


if __name__ == "__main__":
    unittest.main()
