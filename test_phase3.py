"""
=============================================================================
 MediFlow AI — Phase 3 Comprehensive Test Suite
=============================================================================
 Tests all 12 required test conditions:
   1. Normal operating conditions
   2. High bed occupancy
   3. Beds awaiting cleaning or readiness verification
   4. Delayed discharges
   5. CT scanner unavailability
   6. Multiple simultaneous disruptions
   7. Staff workload imbalance
   8. A feasible intervention
   9. An infeasible intervention
   10. Missing or incomplete model data
   11. Consistency of baseline and disrupted comparisons
   12. Preservation of the existing dashboard and What-If Simulator

 Verifies:
   - Invariants: sum of bed states == total licensed beds (no double-counting)
   - Never count a bed as ready until all readiness conditions are satisfied
   - Feasibility engine rejects violations with explicit explanations
   - Calculations are deterministic and mathematically grounded
=============================================================================
"""

import unittest
import pandas as pd
from simulator import run_scenario_step_by_step, run_hospital_what_if_simulation, ScenarioResult
from bottleneck_engine import (
    THRESHOLDS,
    detect_hospital_bottlenecks,
    evaluate_hidden_capacity,
    detect_bottleneck_chains,
    evaluate_staff_workload_imbalance,
    get_available_candidate_interventions,
    evaluate_intervention_feasibility,
    evaluate_multiple_candidate_interventions,
    generate_explainable_recommendations
)
from app import load_synthetic_hospital_data


class TestMediFlowPhase3(unittest.TestCase):

    def setUp(self):
        self.summary, self.df_arrivals, self.df_departments, self.df_resources = load_synthetic_hospital_data()

    def test_01_normal_operating_conditions(self):
        """Test 1: Normal operating conditions without active disruptions."""
        sim_norm = run_scenario_step_by_step(
            name="Normal Baseline",
            scenario_type="baseline",
            description="Nominal flow",
            assumptions=[],
            duration_hours=12,
            total_beds=280,
            initial_occupied=238,
            initial_waiting=34,
            initial_ct_queue=6,
            additional_arrivals_total=0,
            discharge_rate_hourly=5.5,
            max_admission_intake_hourly=12,
            ct_capacity_hourly=8.0,
            ct_status_label="Available"
        )
        findings = detect_hospital_bottlenecks(self.summary, self.df_departments, self.df_resources, sim_norm)
        self.assertIsNotNone(findings)
        
        # Bed occupancy should remain in reasonable range
        self.assertLess(sim_norm.final_occupancy_pct, 95.0)
        
        # Bed state invariant check
        hc = evaluate_hidden_capacity(self.summary, sim_norm)
        self.assertTrue(hc.breakdown.is_invariant_valid)
        self.assertEqual(hc.breakdown.total_accounted, 280)

    def test_02_high_bed_occupancy(self):
        """Test 2: High bed occupancy triggers critical capacity alarm."""
        sim_surge = run_scenario_step_by_step(
            name="High Surge",
            scenario_type="disrupted",
            description="Surge arrivals",
            assumptions=["+60 arrivals"],
            duration_hours=12,
            total_beds=280,
            initial_occupied=238,
            initial_waiting=34,
            initial_ct_queue=6,
            additional_arrivals_total=60,
            discharge_rate_hourly=2.0,  # restricted discharges
            max_admission_intake_hourly=12,
            ct_capacity_hourly=8.0,
            ct_status_label="Available"
        )
        findings = detect_hospital_bottlenecks(self.summary, self.df_departments, self.df_resources, sim_surge)
        bed_finding = next((f for f in findings if f.category == "Bed Capacity"), None)
        self.assertIsNotNone(bed_finding)
        self.assertIn(bed_finding.severity, ["CRITICAL", "WARNING"])
        self.assertGreaterEqual(sim_surge.final_occupancy_pct, THRESHOLDS["BED_OCCUPANCY"]["warning"])

    def test_03_beds_awaiting_cleaning_or_readiness_verification(self):
        """Test 3: Beds awaiting cleaning or verification are never counted as ready available."""
        hc = evaluate_hidden_capacity(self.summary)
        b = hc.breakdown

        # Invariant: partition sum must strictly equal total physical beds
        self.assertEqual(b.total_accounted, b.total_physical_capacity)
        self.assertTrue(b.is_invariant_valid)

        # Readiness rule: ready_available_beds MUST NOT include cleaning or verification beds
        self.assertGreater(b.awaiting_cleaning_evs, 0)
        self.assertGreater(b.awaiting_readiness_verification, 0)
        self.assertLess(b.ready_available_beds, hc.theoretical_unoccupied)

        # Recoverable capacity must strictly equal sum of transitional states
        expected_recoverable = (
            b.awaiting_discharge_coordination +
            b.awaiting_cleaning_evs +
            b.awaiting_readiness_verification
        )
        self.assertEqual(hc.recoverable_beds_total, expected_recoverable)

        # Ordered recovery actions must include EVS and Verification protocols
        action_titles = [a.title for a in hc.ordered_actions]
        self.assertTrue(any("Verification" in t for t in action_titles))
        self.assertTrue(any("EVS" in t for t in action_titles))

    def test_04_delayed_discharges(self):
        """Test 4: Delayed patient discharges trigger discharge bottleneck finding."""
        sim_delayed_disch = run_scenario_step_by_step(
            name="Discharge Delay",
            scenario_type="disrupted",
            description="Stalled handoffs",
            assumptions=["Discharge delay: ON"],
            duration_hours=12,
            total_beds=280,
            initial_occupied=238,
            initial_waiting=34,
            initial_ct_queue=6,
            additional_arrivals_total=0,
            discharge_rate_hourly=2.2,  # Delayed discharge rate
            max_admission_intake_hourly=12,
            ct_capacity_hourly=8.0,
            ct_status_label="Available"
        )
        findings = detect_hospital_bottlenecks(self.summary, self.df_departments, self.df_resources, sim_delayed_disch)
        disch_finding = next((f for f in findings if f.category == "Discharge Flow"), None)
        self.assertIsNotNone(disch_finding)
        self.assertEqual(disch_finding.severity, "CRITICAL")
        self.assertIn("Discharge", disch_finding.title)

    def test_05_ct_scanner_unavailability(self):
        """Test 5: CT scanner unavailability triggers diagnostic backlog finding and chain link."""
        sim_ct_down = run_scenario_step_by_step(
            name="CT Outage",
            scenario_type="disrupted",
            description="CT 2 Down",
            assumptions=["CT status: Unavailable"],
            duration_hours=12,
            total_beds=280,
            initial_occupied=238,
            initial_waiting=34,
            initial_ct_queue=6,
            additional_arrivals_total=0,
            discharge_rate_hourly=5.5,
            max_admission_intake_hourly=10,
            ct_capacity_hourly=3.8,  # Degraded 1-scanner capacity
            ct_status_label="Unavailable"
        )
        findings = detect_hospital_bottlenecks(self.summary, self.df_departments, self.df_resources, sim_ct_down)
        diag_finding = next((f for f in findings if f.category == "Diagnostic Imaging"), None)
        self.assertIsNotNone(diag_finding)
        self.assertEqual(diag_finding.severity, "CRITICAL")
        self.assertGreaterEqual(sim_ct_down.est_ct_wait_mins, THRESHOLDS["CT_WAIT_TIME_MINS"]["critical"])

    def test_06_multiple_simultaneous_disruptions(self):
        """Test 6: Multiple simultaneous disruptions trigger confirmed multi-step dependency chain."""
        sim_multi = run_scenario_step_by_step(
            name="Multiple Disruptions",
            scenario_type="disrupted",
            description="Surge + CT outage + Discharge delay",
            assumptions=["+40 arrivals", "CT status: Unavailable", "Discharge delay: ON"],
            duration_hours=12,
            total_beds=280,
            initial_occupied=238,
            initial_waiting=34,
            initial_ct_queue=6,
            additional_arrivals_total=40,
            discharge_rate_hourly=2.4,
            max_admission_intake_hourly=10,
            ct_capacity_hourly=3.8,
            ct_status_label="Unavailable"
        )
        chain_res = detect_bottleneck_chains(self.summary, sim_multi)
        self.assertTrue(chain_res.is_chain_active)
        self.assertIn("CT", chain_res.initiating_bottleneck)
        self.assertGreater(len(chain_res.affected_resources), 2)

    def test_07_staff_workload_imbalance(self):
        """Test 7: Staff workload imbalance evaluates department acuity ratios and minimum floors."""
        workloads, recs = evaluate_staff_workload_imbalance(self.df_departments, self.summary)
        
        # General Medicine should be overloaded (> 4.8)
        gen_med = next((w for w in workloads if "General Medicine" in w.department), None)
        self.assertIsNotNone(gen_med)
        self.assertGreaterEqual(gen_med.workload_per_staff, THRESHOLDS["STAFF_WORKLOAD_RATIO"]["critical"])

        # Pediatrics and Cardiology should be optimal (< 3.5) with available float capacity above floor
        peds = next((w for w in workloads if "Pediatrics" in w.department), None)
        self.assertIsNotNone(peds)
        self.assertLessEqual(peds.workload_per_staff, THRESHOLDS["STAFF_WORKLOAD_RATIO"]["warning"])
        self.assertGreater(peds.available_float_staff, 0)
        self.assertGreaterEqual(peds.staff_on_duty, peds.min_staff_floor)

        # Viable recommendations should respect minimum floor
        for r in recs:
            self.assertTrue(r.is_floor_respected)
            self.assertGreaterEqual(r.donor_remaining_staff, r.donor_min_floor)

    def test_08_feasible_intervention(self):
        """Test 8: Feasible intervention passes safety checks and improves simulation metrics."""
        res = evaluate_multiple_candidate_interventions(
            candidate_ids=["cand_discharge_lounge", "cand_evs_turnover"],
            additional_arrivals=30,
            ct_scanner_status="Unavailable",
            discharge_delay=True,
            duration_hours=12,
            summary=self.summary,
            df_departments=self.df_departments
        )
        for cand, feas, sim_res in res.evaluated_responses:
            self.assertTrue(feas.is_feasible)
            self.assertIsNotNone(sim_res)
            # Response should reduce waiting patients or improve available beds compared to disrupted
            self.assertLessEqual(sim_res.final_waiting, res.disrupted.final_waiting)

    def test_09_infeasible_intervention(self):
        """Test 9: Infeasible intervention is rejected by safety engine with clear explanation."""
        cands = get_available_candidate_interventions("Unavailable", True)
        cand_infeas_staff = next(c for c in cands if c.id == "cand_infeasible_extreme_pull")
        cand_infeas_bed = next(c for c in cands if c.id == "cand_infeasible_overcapacity_beds")

        feas_staff = evaluate_intervention_feasibility(cand_infeas_staff, self.summary, self.df_departments, None)
        self.assertFalse(feas_staff.is_feasible)
        self.assertFalse(feas_staff.staff_floor_passed)
        self.assertGreater(len(feas_staff.rejection_reasons), 0)
        self.assertTrue(any("STAFFING FLOOR" in r for r in feas_staff.rejection_reasons))

        feas_bed = evaluate_intervention_feasibility(cand_infeas_bed, self.summary, self.df_departments, None)
        self.assertFalse(feas_bed.is_feasible)
        self.assertFalse(feas_bed.bed_limit_passed)
        self.assertTrue(any("PHYSICAL FACILITY LIMIT" in r for r in feas_bed.rejection_reasons))

    def test_10_missing_or_incomplete_model_data(self):
        """Test 10: Graceful handling of empty or missing summary data."""
        empty_summary = {}
        empty_df = pd.DataFrame([{"Department": "General", "Capacity": 10, "Occupied": 5, "Staff": 2}])
        
        # Engine should fall back safely without raising UnboundLocalError or KeyError
        findings = detect_hospital_bottlenecks(empty_summary, empty_df, self.df_resources, None)
        self.assertIsInstance(findings, list)

        hc = evaluate_hidden_capacity(empty_summary, None)
        self.assertIsNotNone(hc)
        self.assertTrue(hc.breakdown.is_invariant_valid)

        workloads, recs = evaluate_staff_workload_imbalance(empty_df, empty_summary)
        self.assertIsInstance(workloads, list)

    def test_11_consistency_of_baseline_and_disrupted_comparisons(self):
        """Test 11: Deterministic consistency across repeated simulation evaluations."""
        res1 = evaluate_multiple_candidate_interventions(
            candidate_ids=["cand_comprehensive_bundle"],
            additional_arrivals=30,
            ct_scanner_status="Unavailable",
            discharge_delay=True,
            duration_hours=12,
            summary=self.summary,
            df_departments=self.df_departments
        )
        res2 = evaluate_multiple_candidate_interventions(
            candidate_ids=["cand_comprehensive_bundle"],
            additional_arrivals=30,
            ct_scanner_status="Unavailable",
            discharge_delay=True,
            duration_hours=12,
            summary=self.summary,
            df_departments=self.df_departments
        )

        # Baseline numbers must be identical
        self.assertEqual(res1.baseline.final_occupied, res2.baseline.final_occupied)
        self.assertEqual(res1.baseline.final_waiting, res2.baseline.final_waiting)
        # Disrupted numbers must be identical
        self.assertEqual(res1.disrupted.final_occupied, res2.disrupted.final_occupied)
        self.assertEqual(res1.disrupted.final_waiting, res2.disrupted.final_waiting)

    def test_12_preservation_of_dashboard_and_what_if_simulator(self):
        """Test 12: Preservation of existing Dashboard and What-If Simulator features."""
        # Summary has all required Phase 1 and Phase 2 keys
        self.assertEqual(self.summary["total_beds"], 280)
        self.assertEqual(self.summary["occupied_beds"], 238)
        self.assertEqual(self.summary["available_beds"], 42)
        self.assertEqual(self.summary["occupancy_rate_pct"], 85.0)

        # What-If Simulator runner produces valid Scenario A, B, C results
        base_res, disr_res, resp_res, meta = run_hospital_what_if_simulation(
            additional_arrivals=30,
            ct_scanner_status="Unavailable",
            discharge_delay=True,
            duration_hours=12
        )
        self.assertIsInstance(base_res, ScenarioResult)
        self.assertIsInstance(disr_res, ScenarioResult)
        self.assertIsInstance(resp_res, ScenarioResult)
        self.assertIn("primary_disruption_bottleneck", meta)
        self.assertGreater(disr_res.final_waiting, 0)
        self.assertLess(resp_res.final_waiting, disr_res.final_waiting)


if __name__ == "__main__":
    unittest.main()
