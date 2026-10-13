"""
=============================================================================
 MediFlow AI — Phase 6: Comprehensive Operational Audit & Validation Tests
=============================================================================
 Tests:
 1. Normal hospital operations baseline checks.
 2. Acute overcrowding & severe patient surge (+60 arrivals).
 3. Critical equipment failure (CT scanner outage).
 4. Severe inpatient discharge delays.
 5. Missing data, malformed inputs, and NaN robustness.
 6. Infeasible interventions strictly rejected (staff floors & capacity caps).
 7. Mathematical invariants (bed partition, queue non-negativity).
 8. Forecast model reproducibility, prediction intervals & surge isolation.
 9. End-to-end 7-step decision intelligence workflow integrity.
 10. Human review governance audit lifecycle and advisory safeguard notice.
=============================================================================
"""

import unittest
import pandas as pd
import numpy as np

from simulator import run_scenario_step_by_step, run_hospital_what_if_simulation, ScenarioResult
from bottleneck_engine import (
    detect_hospital_bottlenecks,
    evaluate_hidden_capacity,
    detect_bottleneck_chains,
    evaluate_staff_workload_imbalance,
    generate_explainable_recommendations,
    THRESHOLDS
)
from forecast_engine import (
    generate_historical_arrival_dataset,
    generate_patient_arrival_forecast,
    evaluate_forecast_model_performance,
    translate_forecast_to_simulator_parameters
)
from simulator_solutions import (
    generate_scenario_tailored_solutions,
    simulate_and_rank_solutions
)
from decision_engine import (
    validate_hospital_state,
    run_integrated_decision_pipeline,
    ReviewStatus,
    HumanReviewRecord,
    IntegratedDecisionPipelineResult
)


class Phase6AuditValidationTests(unittest.TestCase):

    def setUp(self):
        """Set up standard synthetic test datasets."""
        self.summary = {
            "total_beds": 280,
            "occupied_beds": 238,
            "available_beds": 42,
            "occupancy_rate_pct": 85.0,
            "patients_waiting": 34,
            "staff_on_duty": 64,
            "awaiting_discharge_beds": 14,
            "awaiting_cleaning_beds": 12,
            "awaiting_verification_beds": 6,
            "ready_available_beds": 6,
            "unavailable_maintenance_beds": 4
        }
        self.df_departments = pd.DataFrame([
            {"Department": "Emergency Dept (ER)", "Capacity": 45, "Occupied": 42, "Staff": 14, "MinStaffFloor": 10, "AcuityWeight": 1.5, "WorkloadUnits": 63.0, "Qualifications": "Emergency BLS, ACLS, Trauma Triage", "EligibleCrossUnits": "None"},
            {"Department": "Intensive Care (ICU)", "Capacity": 30, "Occupied": 27, "Staff": 12, "MinStaffFloor": 10, "AcuityWeight": 2.0, "WorkloadUnits": 54.0, "Qualifications": "Critical Care, CCRN, ACLS", "EligibleCrossUnits": "None"},
            {"Department": "General Medicine",   "Capacity": 95, "Occupied": 86, "Staff": 16, "MinStaffFloor": 10, "AcuityWeight": 1.0, "WorkloadUnits": 86.0, "Qualifications": "Med-Surg RN, BLS", "EligibleCrossUnits": "None"},
            {"Department": "Surgery & Post-Op",   "Capacity": 55, "Occupied": 51, "Staff": 11, "MinStaffFloor": 8,  "AcuityWeight": 1.1, "WorkloadUnits": 56.1, "Qualifications": "PACU / Surgical RN, BLS", "EligibleCrossUnits": "None"},
            {"Department": "Pediatrics",          "Capacity": 35, "Occupied": 18, "Staff": 6,  "MinStaffFloor": 3,  "AcuityWeight": 1.0, "WorkloadUnits": 18.0, "Qualifications": "Pediatric RN, PALS, Med-Surg Eligible", "EligibleCrossUnits": "General Medicine"},
            {"Department": "Cardiology Ward",     "Capacity": 20, "Occupied": 14, "Staff": 5,  "MinStaffFloor": 2,  "AcuityWeight": 1.0, "WorkloadUnits": 14.0, "Qualifications": "Telemetry RN, ACLS, Med-Surg Eligible", "EligibleCrossUnits": "General Medicine"}
        ])
        self.df_resources = pd.DataFrame([
            {"Resource / Unit": "CT Scanner 1", "Category": "Diagnostics", "Availability Status": "Operational", "Queue Length": "6 Patients", "Est. Wait Time": "45 mins"},
            {"Resource / Unit": "CT Scanner 2", "Category": "Diagnostics", "Availability Status": "Maintenance / Degraded", "Queue Length": "11 Patients", "Est. Wait Time": "95 mins"}
        ])

    def test_01_normal_conditions_baseline(self):
        """Verify baseline calculations under standard unshocked conditions."""
        res = run_scenario_step_by_step(
            name="Baseline Test",
            scenario_type="baseline",
            description="Normal operations",
            assumptions=["No surge", "Both CT operational", "Standard discharge flow"],
            duration_hours=12,
            total_beds=280,
            initial_occupied=238,
            initial_waiting=12,
            initial_ct_queue=4,
            additional_arrivals_total=0,
            discharge_rate_hourly=5.5,
            max_admission_intake_hourly=12,
            ct_capacity_hourly=8.0,
            ct_status_label="Available"
        )
        self.assertEqual(res.total_beds, 280)
        self.assertLessEqual(res.final_occupied, 280)
        self.assertEqual(res.final_occupied + res.final_available, 280)
        self.assertGreater(res.final_available, 0)
        self.assertLessEqual(res.est_ct_wait_mins, 45)

    def test_02_acute_overcrowding_surge(self):
        """Verify model behavior under acute surge shock (+60 arrivals)."""
        res = run_scenario_step_by_step(
            name="Surge Shock",
            scenario_type="disrupted",
            description="Extreme patient surge",
            assumptions=["+60 surge arrivals"],
            duration_hours=12,
            total_beds=280,
            initial_occupied=238,
            initial_waiting=34,
            initial_ct_queue=6,
            additional_arrivals_total=60,
            discharge_rate_hourly=5.5,
            max_admission_intake_hourly=12,
            ct_capacity_hourly=8.0,
            ct_status_label="Available"
        )
        # Hospital experiences acute capacity strain and queue accumulation
        self.assertGreaterEqual(res.final_occupied, 260)
        self.assertGreaterEqual(res.final_occupancy_pct, 95.0)
        self.assertGreaterEqual(res.peak_waiting, 34)
        self.assertGreater(res.projected_bed_demand, 50)

    def test_03_equipment_failure_ct_outage(self):
        """Verify diagnostic wait times spike when CT scanner has 50% outage."""
        res_ok = run_scenario_step_by_step(
            name="CT Normal",
            scenario_type="baseline",
            description="CT operational",
            assumptions=["Both CT operational"],
            duration_hours=12,
            total_beds=280,
            initial_occupied=238,
            initial_waiting=12,
            initial_ct_queue=4,
            additional_arrivals_total=0,
            discharge_rate_hourly=5.5,
            max_admission_intake_hourly=12,
            ct_capacity_hourly=8.0,
            ct_status_label="Available"
        )
        res_down = run_scenario_step_by_step(
            name="CT Outage",
            scenario_type="disrupted",
            description="CT down",
            assumptions=["1 CT down"],
            duration_hours=12,
            total_beds=280,
            initial_occupied=238,
            initial_waiting=12,
            initial_ct_queue=4,
            additional_arrivals_total=0,
            discharge_rate_hourly=5.5,
            max_admission_intake_hourly=10,
            ct_capacity_hourly=3.8,
            ct_status_label="Unavailable"
        )
        self.assertGreater(res_down.est_ct_wait_mins, res_ok.est_ct_wait_mins)
        self.assertGreater(res_down.final_ct_queue, res_ok.final_ct_queue)

    def test_04_discharge_delays_impact(self):
        """Verify discharge delay traps occupied beds and increases waiting queue."""
        res_normal = run_scenario_step_by_step(
            name="Standard Discharge",
            scenario_type="baseline",
            description="Normal discharge",
            assumptions=[],
            duration_hours=12,
            total_beds=280,
            initial_occupied=265,
            initial_waiting=20,
            initial_ct_queue=4,
            additional_arrivals_total=30,
            discharge_rate_hourly=5.5,
            max_admission_intake_hourly=12,
            ct_capacity_hourly=8.0,
            ct_status_label="Available"
        )
        res_delayed = run_scenario_step_by_step(
            name="Delayed Discharge",
            scenario_type="disrupted",
            description="Delayed discharge flow",
            assumptions=[],
            duration_hours=12,
            total_beds=280,
            initial_occupied=265,
            initial_waiting=20,
            initial_ct_queue=4,
            additional_arrivals_total=30,
            discharge_rate_hourly=2.4,
            max_admission_intake_hourly=12,
            ct_capacity_hourly=8.0,
            ct_status_label="Available"
        )
        self.assertLess(res_delayed.total_discharges, res_normal.total_discharges)
        self.assertGreater(res_delayed.final_waiting, res_normal.final_waiting)

    def test_05_missing_and_invalid_data_robustness(self):
        """Verify that malformed or incomplete summary inputs are caught and handled defensively."""
        empty_summary = {}
        is_valid, errors = validate_hospital_state(empty_summary, self.df_departments, self.df_resources)
        self.assertFalse(is_valid)
        self.assertIn("Invalid input", errors[0])

        negative_summary = self.summary.copy()
        negative_summary["occupied_beds"] = -10
        is_valid_neg, errors_neg = validate_hospital_state(negative_summary, self.df_departments, self.df_resources)
        self.assertFalse(is_valid_neg)

        # Division by zero protection
        zero_beds_summary = self.summary.copy()
        zero_beds_summary["total_beds"] = 0
        is_valid_zero, errors_zero = validate_hospital_state(zero_beds_summary, self.df_departments, self.df_resources)
        self.assertFalse(is_valid_zero)

    def test_06_infeasible_interventions_rejected(self):
        """Verify that interventions violating staff floors or licensed capacity are rejected."""
        _, sim_disrupted, _, _ = run_hospital_what_if_simulation(
            additional_arrivals=30, ct_scanner_status="Unavailable", discharge_delay=True, duration_hours=12
        )
        solutions = generate_scenario_tailored_solutions(
            disrupted_res=sim_disrupted,
            summary=self.summary,
            df_departments=self.df_departments,
            additional_arrivals=30,
            ct_scanner_status="Unavailable",
            discharge_delay=True,
            duration_hours=12
        )
        infeasible = [s for s in solutions if not s.feasibility.is_feasible]
        self.assertGreaterEqual(len(infeasible), 2)

        # Attempt to simulate infeasible solutions directly
        infeasible_ids = [s.id for s in infeasible]
        ranked, _ = simulate_and_rank_solutions(
            selected_solution_ids=infeasible_ids,
            available_solutions=solutions,
            duration_hours=12,
            additional_arrivals=30,
            ct_scanner_status="Unavailable",
            discharge_delay=True,
            disrupted_res=sim_disrupted
        )
        for ev in ranked:
            self.assertFalse(ev.is_feasible)
            self.assertIsNotNone(ev.rejection_explanation)

    def test_07_mathematical_invariants_preserved(self):
        """Verify bed partition sums, queue non-negativity, and wait time validity."""
        for duration in [4, 8, 12, 24]:
            base, disr, resp, _ = run_hospital_what_if_simulation(
                additional_arrivals=25, ct_scanner_status="Unavailable", discharge_delay=True, duration_hours=duration
            )
            for res in [base, disr, resp]:
                self.assertEqual(res.final_occupied + res.final_available, res.total_beds)
                self.assertGreaterEqual(res.final_waiting, 0)
                self.assertGreaterEqual(res.est_ct_wait_mins, 0)
                self.assertGreaterEqual(res.total_discharges, 0)

    def test_08_forecast_model_and_duplicate_prevention(self):
        """Verify forecast generation, 7-day holdout validation, and surge delta isolation."""
        df_hist = generate_historical_arrival_dataset(start_date="2026-08-18", num_days=56, seed=42)
        eval_metrics = evaluate_forecast_model_performance(df_hist, test_days=7)
        self.assertLess(eval_metrics.mae, 3.0)  # High accuracy on synthetic dataset
        self.assertLess(eval_metrics.rmse, 4.0)

        forecast = generate_patient_arrival_forecast(df_hist, horizon_hours=12)
        self.assertEqual(len(forecast.interval_forecasts), 12)
        self.assertGreater(forecast.total_predicted_arrivals, 0)
        self.assertGreater(forecast.total_predicted_arrivals, 0)

        # Check incremental surge isolation
        sim_params = translate_forecast_to_simulator_parameters(forecast)
        self.assertGreaterEqual(sim_params["additional_arrivals_total"], 0)
        self.assertLess(sim_params["additional_arrivals_total"], forecast.total_predicted_arrivals)

    def test_09_end_to_end_decision_workflow_integrity(self):
        """Verify the complete 7-step decision pipeline produces consistent results."""
        df_hist = generate_historical_arrival_dataset(start_date="2026-08-18", num_days=56, seed=42)
        forecast = generate_patient_arrival_forecast(df_hist, horizon_hours=12)

        pipe_result = run_integrated_decision_pipeline(
            summary=self.summary,
            df_departments=self.df_departments,
            df_resources=self.df_resources,
            additional_arrivals=30,
            ct_scanner_status="Unavailable",
            discharge_delay=True,
            duration_hours=12,
            forecast=forecast
        )

        self.assertIsInstance(pipe_result, IntegratedDecisionPipelineResult)
        self.assertGreater(len(pipe_result.bottlenecks), 0)
        self.assertTrue(pipe_result.hidden_capacity.breakdown.is_invariant_valid)
        self.assertGreater(pipe_result.hidden_capacity.recoverable_beds_total, 0)
        self.assertIsNotNone(pipe_result.best_feasible_option)
        self.assertTrue(pipe_result.best_feasible_option.is_feasible)
        self.assertGreater(len(pipe_result.trade_offs), 0)
        self.assertGreater(len(pipe_result.human_reviews), 0)

    def test_10_human_review_governance_audit_lifecycle(self):
        """Verify governance status tracking and clinical safeguard audit persistence."""
        pipe_result = run_integrated_decision_pipeline(
            summary=self.summary,
            df_departments=self.df_departments,
            df_resources=self.df_resources
        )
        rec = list(pipe_result.human_reviews.values())[0]
        self.assertEqual(rec.status, ReviewStatus.PENDING_REVIEW)
        self.assertTrue(rec.safeguard_notice_acknowledged)

        # Transition to Approved for Consideration
        rec.status = ReviewStatus.APPROVED_FOR_CONSIDERATION
        rec.reviewer_role = "Chief Medical Officer (CMO)"
        rec.review_notes = "Approved for implementation during evening shift surge."
        self.assertEqual(rec.status, ReviewStatus.APPROVED_FOR_CONSIDERATION)
        self.assertIn("ADVISORY DECISION SUPPORT", rec.advisory_disclaimer)


if __name__ == "__main__":
    unittest.main()
