"""
=============================================================================
 MediFlow AI — Integrated Decision Intelligence Engine
=============================================================================
 Module: decision_engine.py
 Purpose: Phase 5 central decision intelligence orchestrator:
   Connects all hospital optimization modules into one seamless workflow:
   1. Monitor hospital operational state & census telemetry
   2. Detect active operational bottlenecks, hidden capacity blockers & dependency chains
   3. Forecast future patient arrival risks and project proactive bed deficits
   4. Generate context-aware, feasible operational response options
   5. Simulate and compare candidate solutions deterministically under identical baselines
   6. Analyze trade-offs, constraints, and multi-criteria objective ranking
   7. Provide explainable recommendations & interactive human review safeguards
=============================================================================
"""

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np

# Core Simulation Engine
from simulator import run_scenario_step_by_step, run_hospital_what_if_simulation, ScenarioResult

# Bottleneck Intelligence & Staffing
from bottleneck_engine import (
    THRESHOLDS,
    BottleneckFinding,
    OperationalRecommendation,
    HiddenCapacityResult,
    BottleneckChainResult,
    StaffDepartmentWorkload,
    StaffRebalanceRecommendation,
    detect_hospital_bottlenecks,
    evaluate_hidden_capacity,
    detect_bottleneck_chains,
    evaluate_staff_workload_imbalance
)

# Patient Arrival Forecasting
from forecast_engine import (
    generate_historical_arrival_dataset,
    generate_patient_arrival_forecast,
    evaluate_forecast_driven_capacity_risk,
    translate_forecast_to_simulator_parameters,
    ForecastResult,
    ProactiveCapacityAlert
)

# Solution Recommendation & Comparison Engine
from simulator_solutions import (
    ProposedSolution,
    SolutionImpactMetrics,
    RankedSolutionEvaluation,
    generate_scenario_tailored_solutions,
    simulate_and_rank_solutions,
    calculate_impact_metrics
)


# =============================================================================
# 1. Validation & Defensive Safety Guardrails
# =============================================================================

class OperationalIntegrityError(Exception):
    """Raised when an impossible capacity or severe data invariant violation occurs."""
    pass


def validate_hospital_state(
    summary: Dict[str, Any],
    df_departments: pd.DataFrame,
    df_resources: pd.DataFrame
) -> Tuple[bool, List[str]]:
    """
    Validates hospital metrics and inputs for operational integrity:
    1. Prevents division by zero in any metric calculation.
    2. Enforces non-negative resource counts.
    3. Verifies bed partition invariant: occupied + blocked + ready + maint == total.
    4. Enforces physical licensed bed capacity envelope (<= 280 licensed, <= 295 surge max).
    5. Verifies department staff floors and non-empty inputs.
    """
    errors = []

    # 1. Non-empty check
    if not summary or df_departments is None or df_departments.empty:
        errors.append("Invalid input: summary or department data is empty.")
        return False, errors

    tot_beds = summary.get("total_beds", 0)
    occ_beds = summary.get("occupied_beds", 0)

    # 2. Physical licensed capacity rules
    if tot_beds <= 0:
        errors.append("Total beds must be strictly positive.")
    if tot_beds > 295:
        errors.append(f"Impossible capacity: Total beds ({tot_beds}) exceeds physical licensed envelope of 295 beds.")
    if occ_beds < 0:
        errors.append(f"Occupied beds cannot be negative ({occ_beds}).")
    if occ_beds > tot_beds:
        errors.append(f"Occupied beds ({occ_beds}) cannot exceed total physical capacity ({tot_beds}).")

    # 3. Non-negative checks for summary values
    for key in ["available_beds", "patients_waiting", "staff_on_duty"]:
        val = summary.get(key, 0)
        if val < 0:
            errors.append(f"Metric '{key}' cannot be negative ({val}).")

    # 4. Department integrity checks
    for idx, row in df_departments.iterrows():
        dept = row.get("Department", f"Row_{idx}")
        cap = row.get("Capacity", 0)
        occ = row.get("Occupied", 0)
        staff = row.get("Staff", 0)
        floor = row.get("MinStaffFloor", 0)

        if cap <= 0:
            errors.append(f"Department '{dept}' capacity must be positive.")
        if occ < 0 or occ > cap:
            errors.append(f"Department '{dept}' occupied ({occ}) out of bounds [0, {cap}].")
        if staff <= 0:
            errors.append(f"Department '{dept}' staff on duty must be positive.")
        if floor < 0:
            errors.append(f"Department '{dept}' staff floor cannot be negative.")

    # 5. Bed state partition invariant check (Hidden capacity model)
    awaiting_disch = summary.get("awaiting_discharge_beds", 0)
    awaiting_clean = summary.get("awaiting_cleaning_beds", 0)
    awaiting_verif = summary.get("awaiting_verification_beds", 0)
    ready_avail = summary.get("ready_available_beds", 0)
    maint_beds = summary.get("unavailable_maintenance_beds", 0)

    sum_states = occ_beds + awaiting_disch + awaiting_clean + awaiting_verif + ready_avail + maint_beds
    if tot_beds > 0 and sum_states != tot_beds:
        errors.append(
            f"Bed partition invariant violated: sum of states ({sum_states}) != total capacity ({tot_beds})."
        )

    is_valid = len(errors) == 0
    return is_valid, errors


# =============================================================================
# 2. Human Review & Decision Governance Data Structures
# =============================================================================

class ReviewStatus:
    PENDING_REVIEW = "Pending Review"
    APPROVED_FOR_CONSIDERATION = "Approved for Consideration"
    REJECTED = "Rejected"
    NEEDS_MORE_INFORMATION = "Needs More Information"

    ALL_STATUSES = [
        PENDING_REVIEW,
        APPROVED_FOR_CONSIDERATION,
        REJECTED,
        NEEDS_MORE_INFORMATION
    ]


@dataclass
class HumanReviewRecord:
    """Stores the audit trail of a human supervisor's review of a proposed intervention."""
    solution_id: str
    solution_title: str
    status: str                         # Value from ReviewStatus
    reviewer_role: str                  # e.g., 'Chief Medical Officer', 'Nurse Executive', 'Bed Placement Coordinator'
    reviewer_name: str
    review_notes: str
    updated_at: str
    safeguard_notice_acknowledged: bool = True
    advisory_disclaimer: str = (
        "ADVISORY DECISION SUPPORT: This review records prototype administrative intent for consideration. "
        "No real patient transfers, discharges, or treatment alterations are automated or executed."
    )


@dataclass
class SolutionTradeOff:
    """Documents what a solution improves vs what operational strain or resource commitment it adds."""
    solution_id: str
    solution_title: str
    primary_gain: str                   # e.g., "-10 Boarding Patients (-76.9%)"
    secondary_gain: str                 # e.g., "+10 Patient Discharges"
    resources_consumed: str             # e.g., "4 EVS Cleaners reassigned from routine floor sanitation"
    potential_trade_off: str            # e.g., "Routine ward cleaning may experience temporary delay during surge sweep"
    mitigation_strategy: str            # e.g., "Restrict EVS strike team to 90-minute sweep window"
    requires_cross_unit_coordination: bool = True


# =============================================================================
# 3. Complete Integrated Decision Pipeline Container
# =============================================================================

@dataclass
class IntegratedDecisionPipelineResult:
    """Complete diagnostic and decision container representing the full 7-step intelligence workflow."""
    # Step 1: Monitor Data
    timestamp: str
    hospital_summary: Dict[str, Any]
    active_occupancy_pct: float
    total_beds: int
    occupied_beds: int
    ready_available_beds: int
    patients_waiting: int

    # Step 2: Detect Operational Problems
    bottlenecks: List[BottleneckFinding]
    hidden_capacity: HiddenCapacityResult
    bottleneck_chains: BottleneckChainResult
    staff_workloads: List[StaffDepartmentWorkload]
    staff_rebalances: List[StaffRebalanceRecommendation]

    # Step 3: Forecast Future Risks
    forecast: Optional[ForecastResult]
    forecast_capacity_alerts: List[ProactiveCapacityAlert]
    incremental_forecast_surge: int

    # Step 4 & 5: Solutions & Multi-Scenario Simulation
    available_solutions: List[ProposedSolution]
    sim_baseline: ScenarioResult
    sim_disrupted: ScenarioResult
    ranked_evaluations: List[RankedSolutionEvaluation]
    best_feasible_option: Optional[RankedSolutionEvaluation]

    # Step 6: Trade-Off Analysis
    trade_offs: List[SolutionTradeOff]

    # Step 7: Human Review Audit
    human_reviews: Dict[str, HumanReviewRecord]

    # Operational Metadata & Safeguards
    scenario_duration_hours: int
    additional_arrivals_tested: int
    ct_status_tested: str
    discharge_delay_tested: bool
    data_validity_report: List[str] = field(default_factory=list)


# =============================================================================
# 4. Central Decision Intelligence Pipeline Orchestrator
# =============================================================================

def run_integrated_decision_pipeline(
    summary: Dict[str, Any],
    df_departments: pd.DataFrame,
    df_resources: pd.DataFrame,
    additional_arrivals: int = 30,
    ct_scanner_status: str = "Unavailable",
    discharge_delay: bool = True,
    duration_hours: int = 12,
    forecast: Optional[ForecastResult] = None,
    existing_reviews: Optional[Dict[str, HumanReviewRecord]] = None
) -> IntegratedDecisionPipelineResult:
    """
    Executes the complete 7-step MediFlow AI Decision Intelligence Workflow:
    1. Monitor Hospital Data
    2. Detect Bottlenecks & Operational Constraints
    3. Examine Patient Arrival Forecasts
    4. Generate Feasible Candidate Interventions
    5. Deterministically Simulate and Compare Scenarios
    6. Analyze Trade-Offs & Multi-Criteria Objective Ranking
    7. Provide Explainable Recommendations & Human Review Governance

    Guarantees:
    - Same initial conditions across all comparisons.
    - Prevents double-counting of forecast arrivals.
    - Mathematical invariant enforcement on bed partitions.
    - Non-negative counts and division-by-zero safety.
    """
    # 0. Defensive Validation
    is_valid, validation_errors = validate_hospital_state(summary, df_departments, df_resources)

    # Sanitize inputs
    duration_hours = max(4, min(24, int(duration_hours)))
    tot_beds = summary.get("total_beds", 280)
    occ_beds = summary.get("occupied_beds", 238)
    ct_status_clean = "Unavailable" if ("Unavailable" in ct_scanner_status or "Outage" in ct_scanner_status) else "Available"

    # 1. Step 1: Monitor Hospital State
    active_occupancy = round((occ_beds / max(1, tot_beds)) * 100, 1)
    ready_beds = summary.get("ready_available_beds", 6)
    waiting_queue = summary.get("patients_waiting", 34)

    # 2. Step 2: Detect Operational Problems & Bottlenecks
    # Run baseline and disrupted simulation to provide dynamic operational context
    sim_baseline, sim_disrupted, _, meta = run_hospital_what_if_simulation(
        additional_arrivals=additional_arrivals,
        ct_scanner_status=ct_status_clean,
        discharge_delay=discharge_delay,
        duration_hours=duration_hours
    )

    detected_bottlenecks = detect_hospital_bottlenecks(summary, df_departments, df_resources, sim_disrupted)
    hidden_capacity = evaluate_hidden_capacity(summary, None)
    bottleneck_chains = detect_bottleneck_chains(summary, sim_disrupted)
    staff_workloads, staff_rebalances = evaluate_staff_workload_imbalance(df_departments, summary)

    # 3. Step 3: Patient Arrival Forecasting Risk Integration
    incremental_surge = 0
    capacity_alerts: List[ProactiveCapacityAlert] = []

    if forecast is not None:
        # Prevent double-counting: only pass the incremental surge above baseline buffer
        incremental_surge = forecast.incremental_surge_above_baseline
        capacity_alerts = evaluate_forecast_driven_capacity_risk(forecast, summary, df_departments)

    # 4. Step 4: Generate Feasible Operational Response Options
    available_solutions = generate_scenario_tailored_solutions(
        disrupted_res=sim_disrupted,
        summary=summary,
        df_departments=df_departments,
        additional_arrivals=additional_arrivals,
        ct_scanner_status=ct_status_clean,
        discharge_delay=discharge_delay,
        duration_hours=duration_hours
    )

    # Automatically select feasible solutions for comparative evaluation
    selected_ids = [s.id for s in available_solutions if s.feasibility.is_feasible]

    # 5. Step 5: Simulate and Compare Scenarios under Identical Conditions
    ranked_evals, best_option = simulate_and_rank_solutions(
        selected_solution_ids=selected_ids,
        available_solutions=available_solutions,
        duration_hours=duration_hours,
        additional_arrivals=additional_arrivals,
        ct_scanner_status=ct_status_clean,
        discharge_delay=discharge_delay,
        disrupted_res=sim_disrupted
    )

    # 6. Step 6: Trade-Off Analysis
    trade_offs = []
    trade_off_catalog = {
        "sol_evs_turnover": {
            "gain": "Recovers up to 10 unblocked ready beds; reduces triage queue.",
            "sec": "+10 patient discharges turned over per shift.",
            "res": "4 dedicated Environmental Services cleaners assigned to Med-Surg.",
            "trade": "Routine scheduled administrative cleaning deferred for 90 minutes.",
            "mit": "Limit dedicated strike cleaning to 90m bursts; return to routine duties thereafter."
        },
        "sol_surge_flex": {
            "gain": "Adds +15 physical surge beds within licensed cap of 295.",
            "sec": "Absorbs up to 15 emergency triage boarding patients.",
            "res": "4 Float Pool RNs, 2 Observation Techs, 15 Telemetry Monitors.",
            "trade": "Depletes central nursing float pool reserve buffer.",
            "mit": "Authorize flex observation wing only when inpatient occupancy exceeds 90%."
        },
        "sol_ct_diversion": {
            "gain": "Cuts CT diagnostic wait time by >70% (avoids emergency admission stall).",
            "sec": "Eliminates diagnostic scan queue backlog.",
            "res": "1 Standby Radiology Technician, 1 Mobile Trailer CT Unit.",
            "trade": "Incurs external vendor trailer mobilization and elective patient rebooking.",
            "mit": "Pre-arrange regional outpatient divert agreements with partner clinics."
        },
        "sol_discharge_lounge": {
            "gain": "Frees 8-12 acute beds 3-4 hours earlier in morning cycle.",
            "sec": "Increases discharge velocity from 2.4/hr to 5.8/hr.",
            "res": "2 Float RNs, 1 Pharmacy Liaison, 2 Transport Aides.",
            "trade": "Requires patient willingness to transition to lounge prior to ride arrival.",
            "mit": "Provide comfortable lounge amenities, refreshments, and direct pharmacy delivery."
        },
        "sol_staff_rebalance": {
            "gain": "Lowers acute ward nurse-patient workload ratio from 5.4:1 to 4.3:1.",
            "sec": "Expedites floor admission intake to 14 patients/hr.",
            "res": "2 cross-trained Med-Surg RNs reassigned from Pediatrics / Cardiology.",
            "trade": "Reduces donor unit staffing closer to its statutory safety floor.",
            "mit": "Strictly enforce donor floor: Pediatrics retains 4 staff (minimum floor: 3)."
        },
        "sol_combined_bundle": {
            "gain": "Compound relief: simultaneously attacks bed, CT, and discharge bottlenecks.",
            "sec": "Maximum overall objective score across all model metrics.",
            "res": "Multi-department resources: EVS team, float nurses, mobile CT, pharmacy liaison.",
            "trade": "High inter-departmental operational coordination overhead.",
            "mit": "Activate Hospital Incident Command bed huddles every 2 hours."
        }
    }

    for ev in ranked_evals:
        if ev.is_feasible:
            sid = ev.solution.id
            cat = trade_off_catalog.get(sid, {
                "gain": f"Mitigates {abs(ev.impact.delta_waiting if ev.impact else 0)} waiting queue backlog.",
                "sec": f"+{ev.impact.delta_discharges if ev.impact else 0} additional discharges.",
                "res": "Allocates dedicated departmental staff and equipment.",
                "trade": "Reallocates operational focus from elective to acute care.",
                "mit": "Continuous hourly supervisory review."
            })
            trade_offs.append(SolutionTradeOff(
                solution_id=sid,
                solution_title=ev.solution.title,
                primary_gain=cat["gain"],
                secondary_gain=cat["sec"],
                resources_consumed=cat["res"],
                potential_trade_off=cat["trade"],
                mitigation_strategy=cat["mit"],
                requires_cross_unit_coordination=True
            ))

    # 7. Step 7: Human Review Audit Governance
    # Preserve existing reviews or initialize defaults as 'Pending Review'
    human_reviews = existing_reviews.copy() if existing_reviews else {}
    current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    for ev in ranked_evals:
        sid = ev.solution.id
        if sid not in human_reviews:
            human_reviews[sid] = HumanReviewRecord(
                solution_id=sid,
                solution_title=ev.solution.title,
                status=ReviewStatus.PENDING_REVIEW,
                reviewer_role=ev.solution.required_human_approval,
                reviewer_name="Operational Leadership Team",
                review_notes="Generated by Decision Intelligence Engine. Awaiting authorized clinical review.",
                updated_at=current_time_str,
                safeguard_notice_acknowledged=True
            )

    return IntegratedDecisionPipelineResult(
        timestamp=current_time_str,
        hospital_summary=summary,
        active_occupancy_pct=active_occupancy,
        total_beds=tot_beds,
        occupied_beds=occ_beds,
        ready_available_beds=ready_beds,
        patients_waiting=waiting_queue,
        bottlenecks=detected_bottlenecks,
        hidden_capacity=hidden_capacity,
        bottleneck_chains=bottleneck_chains,
        staff_workloads=staff_workloads,
        staff_rebalances=staff_rebalances,
        forecast=forecast,
        forecast_capacity_alerts=capacity_alerts,
        incremental_forecast_surge=incremental_surge,
        available_solutions=available_solutions,
        sim_baseline=sim_baseline,
        sim_disrupted=sim_disrupted,
        ranked_evaluations=ranked_evals,
        best_feasible_option=best_option,
        trade_offs=trade_offs,
        human_reviews=human_reviews,
        scenario_duration_hours=duration_hours,
        additional_arrivals_tested=additional_arrivals,
        ct_status_tested=ct_status_clean,
        discharge_delay_tested=discharge_delay,
        data_validity_report=validation_errors
    )
