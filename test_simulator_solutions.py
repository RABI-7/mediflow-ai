"""
=============================================================================
 MediFlow AI — Solution Recommendation & Comparison Engine Test Suite
=============================================================================
 Tests:
 1. Tailored solution generation for Bed Shortage.
 2. Tailored solution generation for CT Scanner Outage.
 3. Tailored solution generation for Delayed Discharges.
 4. Tailored solution generation for Staff Workload Imbalance.
 5. Multi-lever bundle for Combined Disruptions.
 6. Operational feasibility enforcement & rejection of infeasible actions.
 7. Simulation impact metrics (deltas and percentages, no hardcoding).
 8. Objective ranking and selection rationale.
=============================================================================
"""

import unittest
import pandas as pd
from simulator import run_hospital_what_if_simulation
from simulator_solutions import (
    generate_scenario_tailored_solutions,
    simulate_and_rank_solutions,
    calculate_impact_metrics,
    ProposedSolution,
    RankedSolutionEvaluation
)
from app import load_synthetic_hospital_data


class TestSimulatorSolutions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary, cls.df_arrivals, cls.df_departments, cls.df_resources = load_synthetic_hospital_data()

    def test_01_bed_shortage_solutions(self):
        """Test that bed shortage surge generates bed turnaround and flex bed solutions."""
        _, disrupted, _, _ = run_hospital_what_if_simulation(
            additional_arrivals=45,
            ct_scanner_status="Available",
            discharge_delay=True,
            duration_hours=12
        )
        solutions = generate_scenario_tailored_solutions(
            disrupted_res=disrupted,
            summary=self.summary,
            df_departments=self.df_departments,
            additional_arrivals=45,
            ct_scanner_status="Available",
            discharge_delay=True,
            duration_hours=12
        )
        sol_ids = [s.id for s in solutions]
        self.assertIn("sol_evs_turnover", sol_ids, "EVS Bed Turnover must be recommended under bed shortage.")
        self.assertIn("sol_surge_flex", sol_ids, "Surge Flex Beds must be recommended under bed shortage.")

        # Simulate EVS turnover solution
        ranked, best = simulate_and_rank_solutions(
            selected_solution_ids=["sol_evs_turnover"],
            available_solutions=solutions,
            duration_hours=12,
            additional_arrivals=45,
            ct_scanner_status="Available",
            discharge_delay=True,
            disrupted_res=disrupted
        )
        self.assertIsNotNone(best)
        self.assertTrue(best.is_feasible)
        self.assertLess(best.impact.delta_waiting, 0, "EVS turnover must reduce waiting queue vs disrupted.")

    def test_02_ct_scanner_outage_solutions(self):
        """Test that CT outage specifically generates mobile CT diversion protocol."""
        _, disrupted, _, _ = run_hospital_what_if_simulation(
            additional_arrivals=10,
            ct_scanner_status="Unavailable",
            discharge_delay=False,
            duration_hours=12
        )
        solutions = generate_scenario_tailored_solutions(
            disrupted_res=disrupted,
            summary=self.summary,
            df_departments=self.df_departments,
            additional_arrivals=10,
            ct_scanner_status="Unavailable",
            discharge_delay=False,
            duration_hours=12
        )
        sol_ids = [s.id for s in solutions]
        self.assertIn("sol_ct_diversion", sol_ids, "Mobile CT Diversion must be recommended when CT is down.")

        ranked, best = simulate_and_rank_solutions(
            selected_solution_ids=["sol_ct_diversion"],
            available_solutions=solutions,
            duration_hours=12,
            additional_arrivals=10,
            ct_scanner_status="Unavailable",
            discharge_delay=False,
            disrupted_res=disrupted
        )
        self.assertIsNotNone(best)
        self.assertLess(best.impact.delta_ct_wait, 0, "Mobile CT must decrease CT wait time.")
        self.assertLess(best.impact.delta_ct_queue, 0, "Mobile CT must decrease CT queue.")

    def test_03_delayed_discharges_solutions(self):
        """Test that delayed discharges specifically generates Discharge Lounge solution."""
        _, disrupted, _, _ = run_hospital_what_if_simulation(
            additional_arrivals=10,
            ct_scanner_status="Available",
            discharge_delay=True,
            duration_hours=12
        )
        solutions = generate_scenario_tailored_solutions(
            disrupted_res=disrupted,
            summary=self.summary,
            df_departments=self.df_departments,
            additional_arrivals=10,
            ct_scanner_status="Available",
            discharge_delay=True,
            duration_hours=12
        )
        sol_ids = [s.id for s in solutions]
        self.assertIn("sol_discharge_lounge", sol_ids, "Discharge Lounge must be recommended under discharge delays.")

        ranked, best = simulate_and_rank_solutions(
            selected_solution_ids=["sol_discharge_lounge"],
            available_solutions=solutions,
            duration_hours=12,
            additional_arrivals=10,
            ct_scanner_status="Available",
            discharge_delay=True,
            disrupted_res=disrupted
        )
        self.assertIsNotNone(best)
        self.assertGreater(best.impact.delta_discharges, 0, "Discharge Lounge must increase total discharges.")

    def test_04_staff_workload_rebalance_solution(self):
        """Test nurse rebalancing when occupancy is high, checking floor safety."""
        _, disrupted, _, _ = run_hospital_what_if_simulation(
            additional_arrivals=40,
            ct_scanner_status="Available",
            discharge_delay=False,
            duration_hours=12
        )
        solutions = generate_scenario_tailored_solutions(
            disrupted_res=disrupted,
            summary=self.summary,
            df_departments=self.df_departments,
            additional_arrivals=40,
            ct_scanner_status="Available",
            discharge_delay=False,
            duration_hours=12
        )
        sol_staff = next((s for s in solutions if s.id == "sol_staff_rebalance"), None)
        self.assertIsNotNone(sol_staff, "Staff rebalance should be generated when occupancy is elevated.")
        self.assertTrue(sol_staff.feasibility.is_feasible, "Safe reallocation (above floor) must be feasible.")

    def test_05_combined_disruptions_and_bundle(self):
        """Test compound disruptions (surge + CT down + discharge delay) and bundle effectiveness."""
        _, disrupted, _, _ = run_hospital_what_if_simulation(
            additional_arrivals=35,
            ct_scanner_status="Unavailable",
            discharge_delay=True,
            duration_hours=12
        )
        solutions = generate_scenario_tailored_solutions(
            disrupted_res=disrupted,
            summary=self.summary,
            df_departments=self.df_departments,
            additional_arrivals=35,
            ct_scanner_status="Unavailable",
            discharge_delay=True,
            duration_hours=12
        )
        sol_ids = [s.id for s in solutions]
        self.assertIn("sol_combined_bundle", sol_ids)

        ranked, best = simulate_and_rank_solutions(
            selected_solution_ids=["sol_evs_turnover", "sol_ct_diversion", "sol_combined_bundle"],
            available_solutions=solutions,
            duration_hours=12,
            additional_arrivals=35,
            ct_scanner_status="Unavailable",
            discharge_delay=True,
            disrupted_res=disrupted
        )
        self.assertIsNotNone(best)
        # Combined bundle should achieve high objective score by addressing beds, CT, and discharges
        bundle_eval = next((e for e in ranked if e.solution.id == "sol_combined_bundle"), None)
        self.assertIsNotNone(bundle_eval)
        self.assertGreater(bundle_eval.objective_score, 0)
        self.assertLess(bundle_eval.impact.delta_waiting, 0)
        self.assertLess(bundle_eval.impact.delta_ct_wait, 0)
        self.assertGreater(bundle_eval.impact.delta_discharges, 0)

    def test_06_infeasible_rejections(self):
        """Verify that stripping staff below floors or exceeding licensed beds is strictly rejected."""
        _, disrupted, _, _ = run_hospital_what_if_simulation(
            additional_arrivals=25,
            ct_scanner_status="Available",
            discharge_delay=False,
            duration_hours=12
        )
        solutions = generate_scenario_tailored_solutions(
            disrupted_res=disrupted,
            summary=self.summary,
            df_departments=self.df_departments,
            additional_arrivals=25,
            ct_scanner_status="Available",
            discharge_delay=False,
            duration_hours=12
        )
        ranked, _ = simulate_and_rank_solutions(
            selected_solution_ids=["sol_infeasible_staff_strip", "sol_infeasible_hallway_beds"],
            available_solutions=solutions,
            duration_hours=12,
            additional_arrivals=25,
            ct_scanner_status="Available",
            discharge_delay=False,
            disrupted_res=disrupted
        )
        for ev in ranked:
            self.assertFalse(ev.is_feasible, f"{ev.solution.title} must be rejected as infeasible.")
            self.assertIsNone(ev.sim_result, "Infeasible solutions must not produce simulation results.")
            self.assertIsNotNone(ev.rejection_explanation)

    def test_07_no_hardcoding_and_delta_math(self):
        """Verify delta calculations: delta = intervened - disrupted; pct = delta / disrupted * 100."""
        _, disrupted, _, _ = run_hospital_what_if_simulation(
            additional_arrivals=45,
            ct_scanner_status="Available",
            discharge_delay=True,
            duration_hours=12
        )
        solutions = generate_scenario_tailored_solutions(
            disrupted_res=disrupted,
            summary=self.summary,
            df_departments=self.df_departments,
            additional_arrivals=45,
            ct_scanner_status="Available",
            discharge_delay=True,
            duration_hours=12
        )
        ranked, best = simulate_and_rank_solutions(
            selected_solution_ids=["sol_evs_turnover"],
            available_solutions=solutions,
            duration_hours=12,
            additional_arrivals=45,
            ct_scanner_status="Available",
            discharge_delay=True,
            disrupted_res=disrupted
        )
        self.assertIsNotNone(best)
        imp = best.impact
        self.assertEqual(imp.delta_waiting, best.sim_result.final_waiting - disrupted.final_waiting)
        self.assertEqual(imp.delta_available, best.sim_result.final_available - disrupted.final_available)
        self.assertEqual(imp.delta_occupied, best.sim_result.final_occupied - disrupted.final_occupied)
        self.assertEqual(imp.delta_discharges, best.sim_result.total_discharges - disrupted.total_discharges)
        self.assertNotEqual(imp.pct_waiting, 0.0)

    def test_08_explainability_fields(self):
        """Verify that every proposed solution includes all required explainability fields."""
        _, disrupted, _, _ = run_hospital_what_if_simulation(
            additional_arrivals=30,
            ct_scanner_status="Unavailable",
            discharge_delay=True,
            duration_hours=12
        )
        solutions = generate_scenario_tailored_solutions(
            disrupted_res=disrupted,
            summary=self.summary,
            df_departments=self.df_departments,
            additional_arrivals=30,
            ct_scanner_status="Unavailable",
            discharge_delay=True,
            duration_hours=12
        )
        for s in solutions:
            self.assertTrue(len(s.target_bottleneck) > 0, "Must specify target bottleneck.")
            self.assertTrue(len(s.evidence_summary) > 0, "Must specify evidence summary.")
            self.assertTrue(len(s.operational_rationale) > 0, "Must specify operational rationale.")
            self.assertTrue(len(s.assumptions) > 0, "Must specify model assumptions.")
            self.assertIn(s.evidence_quality, ["Strong Model Evidence", "Partial Model Evidence", "Insufficient Evidence"])
            self.assertTrue(len(s.required_human_approval) > 0, "Must specify required human approval.")
            self.assertTrue(len(s.clinical_priority_safeguard) > 0, "Must specify clinical priority safeguard.")
            self.assertTrue(len(s.limitations_or_missing_data) > 0, "Must specify limitations or missing data.")


if __name__ == "__main__":
    unittest.main()
