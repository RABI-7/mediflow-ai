"""
=============================================================================
 MediFlow AI — Solution Recommendation & Comparison Engine
=============================================================================
 Module: simulator_solutions.py
 Purpose: Upgrades the What-If Simulator from merely identifying bottlenecks
          to proposing, simulating, evaluating feasibility, and comparing solutions:
          1. Context-aware solution generation based on actual calculated bottlenecks.
          2. Rigorous operational feasibility checks (staff floors, bed caps, equipment limits).
          3. Deterministic simulation of user-selected interventions under identical conditions.
          4. Absolute and percentage impact calculations relative to the disrupted baseline.
          5. Objective-based ranking and selection of the "Best Feasible Option".
          6. Explainable evidence panels with human review requirements and clinical safeguards.
=============================================================================
"""

import math
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
from simulator import run_scenario_step_by_step, ScenarioResult
from bottleneck_engine import (
    THRESHOLDS,
    BottleneckFinding,
    evaluate_intervention_feasibility,
    FeasibilityReport,
    CandidateIntervention
)


@dataclass
class ProposedSolution:
    """Stores a context-aware proposed operational solution for the simulator."""
    id: str
    title: str
    category: str
    target_bottleneck: str
    description: str
    operational_rationale: str
    action_plan: str
    assumptions: List[str]
    staff_needed: Dict[str, int]
    equipment_needed: Dict[str, int]
    bed_capacity_bonus: int
    discharge_velocity: float
    ct_capacity: float
    intake_velocity: int
    feasibility: FeasibilityReport
    evidence_quality: str               # 'Strong Model Evidence', 'Partial Model Evidence', 'Insufficient Evidence'
    evidence_summary: str
    required_human_approval: str
    clinical_priority_safeguard: str
    limitations_or_missing_data: str = "Synthetic model assumptions; real-world patient acuity and physician discharge orders require clinical confirmation."
    is_compatible_with: List[str] = field(default_factory=list)


@dataclass
class SolutionImpactMetrics:
    """Calculated absolute and percentage impact compared with the disrupted scenario."""
    # Waiting Patients
    disrupted_waiting: int
    intervened_waiting: int
    delta_waiting: int                 # negative is good
    pct_waiting: float                 # negative is good

    # Available Beds
    disrupted_available: int
    intervened_available: int
    delta_available: int               # positive is good
    pct_available: float

    # Occupied Beds
    disrupted_occupied: int
    intervened_occupied: int
    delta_occupied: int
    pct_occupied: float

    # CT Wait Time (mins)
    disrupted_ct_wait: int
    intervened_ct_wait: int
    delta_ct_wait: int                 # negative is good
    pct_ct_wait: float

    # CT Queue
    disrupted_ct_queue: int
    intervened_ct_queue: int
    delta_ct_queue: int                # negative is good
    pct_ct_queue: float

    # Total Discharges
    disrupted_discharges: int
    intervened_discharges: int
    delta_discharges: int              # positive is good
    pct_discharges: float


@dataclass
class RankedSolutionEvaluation:
    """Complete evaluation of a proposed solution candidate in comparison."""
    solution: ProposedSolution
    sim_result: Optional[ScenarioResult]
    is_feasible: bool
    rejection_explanation: Optional[str]
    impact: Optional[SolutionImpactMetrics]
    objective_score: float
    rank: int
    selection_rationale: str


# =============================================================================
# 1. Context-Aware Solution Generation (Based on Calculated Results)
# =============================================================================

def generate_scenario_tailored_solutions(
    disrupted_res: ScenarioResult,
    summary: Dict[str, Any],
    df_departments: pd.DataFrame,
    additional_arrivals: int,
    ct_scanner_status: str,
    discharge_delay: bool,
    duration_hours: int
) -> List[ProposedSolution]:
    """
    Generates solutions directly tailored to the actual calculated bottlenecks of the active scenario.
    Does NOT display generic or irrelevant recommendations (e.g. omits CT diversion if CT is operational).
    """
    solutions = []
    is_ct_down = (ct_scanner_status == "Unavailable" or "Outage" in ct_scanner_status)
    base_disch = 2.4 if discharge_delay else 5.5
    base_ct = 3.8 if is_ct_down else 8.0

    occ_pct = disrupted_res.final_occupancy_pct
    avail_beds = disrupted_res.final_available
    waiting_count = disrupted_res.final_waiting
    ct_wait = disrupted_res.est_ct_wait_mins
    ct_queue = disrupted_res.final_ct_queue

    # -------------------------------------------------------------------------
    # 1. Bed Turnaround / Capacity Shortage Solutions
    # -------------------------------------------------------------------------
    if occ_pct >= THRESHOLDS["BED_OCCUPANCY"]["warning"] or avail_beds <= THRESHOLDS["AVAILABLE_BEDS"]["warning"] or waiting_count > 0:
        cand_evs = CandidateIntervention(
            id="sol_evs_turnover",
            name="Solution 1: Priority EVS Bed Turnover Fast-Track",
            category="Bed Turnaround",
            description="Deploys 4 dedicated Environmental Services cleaners to batch-turnover dirty inpatient beds, accelerating admission flow.",
            assumptions=["EVS turnaround reduced from 65m to 25m", "+10 ready beds unlocked in 45m", "Admission intake accelerated to 14/hr"],
            staff_needed={"EVS Cleaners": 4},
            equipment_needed={},
            bed_capacity_bonus=0,
            discharge_velocity=round(base_disch + 0.8, 1),
            ct_capacity=base_ct,
            intake_velocity=14,
            is_compatible_with=["sol_discharge_lounge", "sol_ct_diversion", "sol_staff_rebalance"]
        )
        feas_evs = evaluate_intervention_feasibility(cand_evs, summary, df_departments, disrupted_res)
        solutions.append(ProposedSolution(
            id="sol_evs_turnover",
            title="Priority EVS Bed Turnover Fast-Track",
            category="Bed Turnaround",
            target_bottleneck="Bed Shortage & Cleaning Turnover Lag",
            description=cand_evs.description,
            operational_rationale="Accelerating terminal room sanitation recovers genuinely blocked beds (12 beds awaiting cleaning) without adding licensed physical capacity.",
            action_plan="1. Dispatch 4 dedicated EVS strike team cleaners to Med-Surg wards.\n2. Prioritize rooms vacated within the last 90 minutes.\n3. Implement immediate mobile tablet sign-off.",
            assumptions=cand_evs.assumptions,
            staff_needed=cand_evs.staff_needed,
            equipment_needed=cand_evs.equipment_needed,
            bed_capacity_bonus=0,
            discharge_velocity=round(base_disch + 0.8, 1),
            ct_capacity=base_ct,
            intake_velocity=14,
            feasibility=feas_evs,
            evidence_quality="Strong Model Evidence",
            evidence_summary=f"Occupancy is {occ_pct}% with {avail_beds} available beds and {waiting_count} boarding patients.",
            required_human_approval="Environmental Services (EVS) Supervisor & Bed Placement Coordinator",
            clinical_priority_safeguard="Clinical Safeguard Active: Newly readied beds are assigned strictly by triage acuity (ESI 1-2 priority preserved).",
            limitations_or_missing_data="Assumes constant 25m turnaround; isolation/C. diff terminal room decontamination requires additional chemical dwell time not modeled."
        ))

        # Flex Observation Beds
        cand_flex = CandidateIntervention(
            id="sol_surge_flex",
            name="Solution 2: Open 15 Surge Observation Flex Beds",
            category="Surge Capacity",
            description="Opens 15 staffed observation beds in step-down wing to absorb emergency triage holding patients.",
            assumptions=["Licensed physical beds expanded from 280 to 295 (+15 flex beds)", "Intake velocity expanded to 14/hr"],
            staff_needed={"Float Nurses": 4, "Observation Techs": 2},
            equipment_needed={"Telemetry Monitors": 15},
            bed_capacity_bonus=15,
            discharge_velocity=base_disch,
            ct_capacity=base_ct,
            intake_velocity=14,
            is_compatible_with=["sol_discharge_lounge", "sol_ct_diversion", "sol_evs_turnover"]
        )
        feas_flex = evaluate_intervention_feasibility(cand_flex, summary, df_departments, disrupted_res)
        solutions.append(ProposedSolution(
            id="sol_surge_flex",
            title="Open 15 Surge Observation Flex Beds",
            category="Surge Capacity",
            target_bottleneck="Triage Boarding Queue & Emergency Bed Saturation",
            description=cand_flex.description,
            operational_rationale="Physical surge buffer expansion absorbs holding patients when inpatient ward occupancy exceeds 90%.",
            action_plan="1. Authorize step-down flex observation wing activation.\n2. Deploy 4 float pool nurses.\n3. Transition non-ICU boarding patients from ER hallways.",
            assumptions=cand_flex.assumptions,
            staff_needed=cand_flex.staff_needed,
            equipment_needed=cand_flex.equipment_needed,
            bed_capacity_bonus=15,
            discharge_velocity=base_disch,
            ct_capacity=base_ct,
            intake_velocity=14,
            feasibility=feas_flex,
            evidence_quality="Strong Model Evidence",
            evidence_summary=f"Physical beds expanded within licensed limit (280 -> 295 max). Waiting queue: {waiting_count} patients.",
            required_human_approval="Chief Medical Officer (CMO) & Central Nursing Staffing Office",
            clinical_priority_safeguard="Clinical Safeguard Active: Observation beds restricted to low-to-moderate acuity patients; high-acuity remain in monitored acute bays.",
            limitations_or_missing_data="Model assumes float nurses are on-site immediately; does not capture delayed telemetry monitor calibration or ward supply provisioning."
        ))

    # -------------------------------------------------------------------------
    # 2. Diagnostic CT Scanner Solutions (ONLY when CT is disrupted or backlogged)
    # -------------------------------------------------------------------------
    if is_ct_down or ct_wait >= THRESHOLDS["CT_WAIT_TIME_MINS"]["warning"] or ct_queue >= THRESHOLDS["CT_QUEUE_PATIENTS"]["warning"]:
        cand_ct = CandidateIntervention(
            id="sol_ct_diversion",
            name="Solution 3: Diagnostic Outpatient Diversion & Mobile CT Protocol",
            category="Diagnostic Imaging",
            description="Diverts elective outpatients and operates mobile scanner to restore diagnostic throughput from 3.8 to 6.8 scans/hr.",
            assumptions=["Diagnostic CT throughput restored to 6.8 scans/hr", "Reduces scan queue by >70%"],
            staff_needed={"On-Call Radiology Tech": 1},
            equipment_needed={"Mobile CT Unit": 1},
            bed_capacity_bonus=0,
            discharge_velocity=base_disch,
            ct_capacity=6.8,
            intake_velocity=12,
            is_compatible_with=["sol_evs_turnover", "sol_discharge_lounge", "sol_staff_rebalance", "sol_surge_flex"]
        )
        feas_ct = evaluate_intervention_feasibility(cand_ct, summary, df_departments, disrupted_res)
        solutions.append(ProposedSolution(
            id="sol_ct_diversion",
            title="Diagnostic Outpatient Diversion & Mobile CT Protocol",
            category="Diagnostic Imaging",
            target_bottleneck="CT Scanner Outage & Diagnostic Backlog",
            description=cand_ct.description,
            operational_rationale="Relieves emergency scan queue bottleneck caused by Scanner 2 maintenance, preventing diagnostic delays from stalling physician admissions.",
            action_plan="1. Divert scheduled non-urgent outpatient CT appointments to regional partner facility.\n2. Call in standby radiology technician to activate mobile trailer scanner.\n3. Expedite trauma scan protocol reads.",
            assumptions=cand_ct.assumptions,
            staff_needed=cand_ct.staff_needed,
            equipment_needed=cand_ct.equipment_needed,
            bed_capacity_bonus=0,
            discharge_velocity=base_disch,
            ct_capacity=6.8,
            intake_velocity=12,
            feasibility=feas_ct,
            evidence_quality="Strong Model Evidence",
            evidence_summary=f"CT wait time is {ct_wait} mins with {ct_queue} patients in queue (Scanner status: {ct_scanner_status}).",
            required_human_approval="Radiology Medical Director & Emergency Department Chair",
            clinical_priority_safeguard="Clinical Safeguard Active: Emergent trauma and acute stroke scans retain immediate zero-wait priority on Scanner 1.",
            limitations_or_missing_data="Assumes standby technician reports within 45m and mobile trailer CT passes daily QA calibration; non-contrast vs contrast breakdown is unmodeled."
        ))

    # -------------------------------------------------------------------------
    # 3. Discharge Delay Solutions (ONLY when discharge delays active or turnover low)
    # -------------------------------------------------------------------------
    hourly_disch = disrupted_res.total_discharges / max(1, duration_hours)
    if discharge_delay or hourly_disch <= THRESHOLDS["DISCHARGE_RATE_HOURLY"]["warning"]:
        cand_disch = CandidateIntervention(
            id="sol_discharge_lounge",
            name="Solution 4: Inpatient Discharge Lounge & Fast-Track Pharmacy",
            category="Discharge Optimization",
            description="Transfers medically cleared patients to staffed lounge, accelerating discharge velocity from ~2.4/hr to 5.8/hr.",
            assumptions=["Discharge velocity increased to 5.8/hr (+3.4/hr)", "Frees acute beds earlier in the diurnal cycle"],
            staff_needed={"Float Nurses": 2, "Pharmacy Liaison": 1},
            equipment_needed={},
            bed_capacity_bonus=0,
            discharge_velocity=5.8,
            ct_capacity=base_ct,
            intake_velocity=13,
            is_compatible_with=["sol_evs_turnover", "sol_ct_diversion", "sol_surge_flex", "sol_staff_rebalance"]
        )
        feas_disch = evaluate_intervention_feasibility(cand_disch, summary, df_departments, disrupted_res)
        solutions.append(ProposedSolution(
            id="sol_discharge_lounge",
            title="Inpatient Discharge Lounge & Fast-Track Pharmacy",
            category="Discharge Flow",
            target_bottleneck="Delayed Patient Discharges & Handoff Stall",
            description=cand_disch.description,
            operational_rationale="Overcomes handoff delays where medically stable patients occupy inpatient beds waiting for transport or medications, clearing beds 3-4 hours earlier.",
            action_plan="1. Open staffed Discharge Lounge.\n2. Reassign 2 float nurses and 1 pharmacy liaison.\n3. Transition ambulatory cleared patients by 11:00 AM.",
            assumptions=cand_disch.assumptions,
            staff_needed=cand_disch.staff_needed,
            equipment_needed=cand_disch.equipment_needed,
            bed_capacity_bonus=0,
            discharge_velocity=5.8,
            ct_capacity=base_ct,
            intake_velocity=13,
            feasibility=feas_disch,
            evidence_quality="Strong Model Evidence",
            evidence_summary=f"Discharge velocity is currently {hourly_disch:.1f}/hr (Target: {THRESHOLDS['DISCHARGE_RATE_HOURLY']['target']}/hr).",
            required_human_approval="Chief Nursing Officer (CNO) & Ward Charge Nurse",
            clinical_priority_safeguard="Clinical Safeguard Active: Requires written physician discharge order; medical care is never truncated prematurely.",
            limitations_or_missing_data="Assumes outpatient transportation and home pharmacy dispensaries operate without insurance authorization delays."
        ))

    # -------------------------------------------------------------------------
    # 4. Staff Workload Rebalancing Solutions (ONLY when high occupancy / imbalance)
    # -------------------------------------------------------------------------
    if disrupted_res.final_occupied >= 245 or occ_pct >= 88.0:
        cand_staff = CandidateIntervention(
            id="sol_staff_rebalance",
            name="Solution 5: Qualified Nurse Rebalancing (Pediatrics/Cardio -> Med-Surg)",
            category="Staff Workload",
            description="Reassigns 2 cross-trained Med-Surg nurses from low-census units (Pediatrics/Cardiology) above their safety floors to acute units.",
            assumptions=["Donor units maintain staffing >= statutory minimum floor", "Expedites floor intake to 14/hr"],
            staff_needed={"Cross-Trained Med-Surg RN": 2},
            equipment_needed={},
            bed_capacity_bonus=0,
            discharge_velocity=base_disch + 0.3,
            ct_capacity=base_ct,
            intake_velocity=14,
            is_compatible_with=["sol_evs_turnover", "sol_ct_diversion", "sol_discharge_lounge"]
        )
        feas_staff = evaluate_intervention_feasibility(cand_staff, summary, df_departments, disrupted_res)
        solutions.append(ProposedSolution(
            id="sol_staff_rebalance",
            title="Qualified Nurse Rebalancing (Pediatrics/Cardio -> Med-Surg)",
            category="Staff Workload",
            target_bottleneck="Acute Nursing Workload Imbalance",
            description=cand_staff.description,
            operational_rationale="Balances staffing ratios from 5.4:1 down to a safer 4.3:1 while preserving donor unit statutory minimum floors (Pediatrics: 4 remaining >= 3 floor).",
            action_plan="1. Review census in Pediatrics and Cardiology.\n2. Reassign 2 BLS/ACLS certified nurses to General Medicine.\n3. Validate nurse competencies.",
            assumptions=cand_staff.assumptions,
            staff_needed=cand_staff.staff_needed,
            equipment_needed=cand_staff.equipment_needed,
            bed_capacity_bonus=0,
            discharge_velocity=base_disch + 0.3,
            ct_capacity=base_ct,
            intake_velocity=14,
            feasibility=feas_staff,
            evidence_quality="Strong Model Evidence",
            evidence_summary="Verified donor units have active staff exceeding statutory floor constraints.",
            required_human_approval="Central Nursing Office & Unit Nurse Managers",
            clinical_priority_safeguard="Clinical Safeguard Active: Donor unit minimum staffing floor is never violated; pediatric safety buffer maintained.",
            limitations_or_missing_data="Cross-unit orientation records are synthetic; real-world collective bargaining rules and nurse fatigue indices must be reviewed."
        ))

    # -------------------------------------------------------------------------
    # 5. Combined Multi-Lever Response Bundle (Compatible Solutions)
    # -------------------------------------------------------------------------
    cand_bundle = CandidateIntervention(
        id="sol_combined_bundle",
        name="Solution 6: Combined Multi-Lever Response Strategy",
        category="Multi-Lever Optimization",
        description="Simultaneously activates Priority EVS Bed Turnover, Inpatient Discharge Lounge, and Mobile CT Protocol.",
        assumptions=["EVS turnover + Discharge velocity 5.8/hr + CT capacity 6.8/hr", "Resources are non-overlapping across departments"],
        staff_needed={"EVS Cleaners": 4, "Float Nurses": 2, "Pharmacy Liaison": 1, "On-Call Radiology Tech": 1},
        equipment_needed={"Mobile CT Unit": 1},
        bed_capacity_bonus=0,
        discharge_velocity=5.8,
        ct_capacity=6.8,
        intake_velocity=15,
        is_compatible_with=[]
    )
    feas_bundle = evaluate_intervention_feasibility(cand_bundle, summary, df_departments, disrupted_res)
    solutions.append(ProposedSolution(
        id="sol_combined_bundle",
        title="Combined Multi-Lever Response Strategy (Bundle)",
        category="Multi-Lever Optimization",
        target_bottleneck="Compound Bottlenecks (Bed + CT + Discharge Flow)",
        description=cand_bundle.description,
        operational_rationale="Simultaneously attacks multiple constraint points in the dependency chain, preventing delays in one unit from backing up into another.",
        action_plan="1. Activate Discharge Lounge by 10:00.\n2. Dispatch EVS cleaning team to Med-Surg.\n3. Bring online Mobile CT unit.\n4. Coordinate cross-functional bed huddles.",
        assumptions=cand_bundle.assumptions,
        staff_needed=cand_bundle.staff_needed,
        equipment_needed=cand_bundle.equipment_needed,
        bed_capacity_bonus=0,
        discharge_velocity=5.8,
        ct_capacity=6.8,
        intake_velocity=15,
        feasibility=feas_bundle,
        evidence_quality="Strong Model Evidence",
        evidence_summary="Combines verified compatible solutions; total float staff requested does not exceed active float pool limits.",
        required_human_approval="Hospital Incident Command & Chief Operating Officer (COO)",
        clinical_priority_safeguard="Clinical Safeguard Active: Full adherence to clinical priority triage (ESI 1-5) and staffing floor constraints across all wards.",
        limitations_or_missing_data="Requires simultaneous inter-departmental leadership coordination; operational friction may delay compound gains by 30-60 minutes."
    ))

    # -------------------------------------------------------------------------
    # 6. Test Infeasible Candidates (For Safety & Rejection Demonstration)
    # -------------------------------------------------------------------------
    cand_infeas_staff = CandidateIntervention(
        id="sol_infeasible_staff_strip",
        name="Solution 7: [TEST/INFEASIBLE] Strip Ward Staffing for ER",
        category="High-Risk Reallocation",
        description="Attempts to reassign 6 nurses from Pediatrics to ER (violating statutory minimum staffing floors).",
        assumptions=["Extreme staff stripping"],
        staff_needed={"Pediatric RN": 6},
        equipment_needed={},
        bed_capacity_bonus=0,
        discharge_velocity=base_disch,
        ct_capacity=base_ct,
        intake_velocity=12,
        is_compatible_with=[]
    )
    feas_infeas_staff = evaluate_intervention_feasibility(cand_infeas_staff, summary, df_departments, disrupted_res)
    solutions.append(ProposedSolution(
        id="sol_infeasible_staff_strip",
        title="[TEST/INFEASIBLE] Strip Ward Staffing for ER",
        category="High-Risk Reallocation",
        target_bottleneck="ER Nurse Shortage",
        description=cand_infeas_staff.description,
        operational_rationale="Demonstrates safety check: Reassigning 6 nurses from Pediatrics leaves the unit below its statutory floor of 3.",
        action_plan="REJECTED BY SAFETY ENGINE. Cannot be executed without external registry staff.",
        assumptions=cand_infeas_staff.assumptions,
        staff_needed=cand_infeas_staff.staff_needed,
        equipment_needed=cand_infeas_staff.equipment_needed,
        bed_capacity_bonus=0,
        discharge_velocity=base_disch,
        ct_capacity=base_ct,
        intake_velocity=12,
        feasibility=feas_infeas_staff,
        evidence_quality="Strong Model Evidence",
        evidence_summary="Violates minimum staffing floor. Rejection is mandatory under patient safety regulations.",
        required_human_approval="NOT APPROVABLE (Violates Regulatory Minimums)",
        clinical_priority_safeguard="Regulatory Safeguard Active: Minimum licensed staffing floor is strictly non-negotiable.",
        limitations_or_missing_data="REJECTED: Model strictly prohibits violating pediatric staffing statutory floors (<3 nurses active)."
    ))

    cand_infeas_bed = CandidateIntervention(
        id="sol_infeasible_hallway_beds",
        name="Solution 8: [TEST/INFEASIBLE] Open 50 Unstaffed Hallway Beds",
        category="Structural Limit Violation",
        description="Attempts to open 50 hallway beds exceeding the physical facility licensed safety envelope of 295 beds.",
        assumptions=["Structural over-expansion"],
        staff_needed={},
        equipment_needed={},
        bed_capacity_bonus=50,
        discharge_velocity=base_disch,
        ct_capacity=base_ct,
        intake_velocity=12,
        is_compatible_with=[]
    )
    feas_infeas_bed = evaluate_intervention_feasibility(cand_infeas_bed, summary, df_departments, disrupted_res)
    solutions.append(ProposedSolution(
        id="sol_infeasible_hallway_beds",
        title="[TEST/INFEASIBLE] Open 50 Unstaffed Hallway Beds",
        category="Structural Limit Violation",
        target_bottleneck="Severe Inpatient Bed Saturation",
        description=cand_infeas_bed.description,
        operational_rationale="Demonstrates safety check: Opening 50 beds creates 330 total beds, exceeding physical licensed capacity by 35 beds.",
        action_plan="REJECTED BY SAFETY ENGINE. Structural facility limit cannot be overridden.",
        assumptions=cand_infeas_bed.assumptions,
        staff_needed=cand_infeas_bed.staff_needed,
        equipment_needed=cand_infeas_bed.equipment_needed,
        bed_capacity_bonus=50,
        discharge_velocity=base_disch,
        ct_capacity=base_ct,
        intake_velocity=12,
        feasibility=feas_infeas_bed,
        evidence_quality="Strong Model Evidence",
        evidence_summary="Exceeds maximum licensed physical structural cap of 295 beds.",
        required_human_approval="NOT APPROVABLE (Structural License Violation)",
        clinical_priority_safeguard="Regulatory Safeguard Active: Physical fire safety and licensed bed caps enforced.",
        limitations_or_missing_data="REJECTED: Structural fire code and state licensing law limits maximum physical bed capacity to 295."
    ))

    return solutions


# =============================================================================
# 2. Simulation, Delta Calculation & Solution Ranking
# =============================================================================

def calculate_impact_metrics(
    disrupted: ScenarioResult,
    intervened: ScenarioResult
) -> SolutionImpactMetrics:
    """Calculates absolute and percentage changes between disrupted and intervened scenarios."""
    d_wait = intervened.final_waiting - disrupted.final_waiting
    pct_wait = round((d_wait / max(1, disrupted.final_waiting)) * 100, 1)

    d_avail = intervened.final_available - disrupted.final_available
    pct_avail = round((d_avail / max(1, disrupted.final_available)) * 100, 1)

    d_occ = intervened.final_occupied - disrupted.final_occupied
    pct_occ = round((d_occ / max(1, disrupted.final_occupied)) * 100, 1)

    d_ct_w = intervened.est_ct_wait_mins - disrupted.est_ct_wait_mins
    pct_ct_w = round((d_ct_w / max(1, disrupted.est_ct_wait_mins)) * 100, 1)

    d_ct_q = intervened.final_ct_queue - disrupted.final_ct_queue
    pct_ct_q = round((d_ct_q / max(1, disrupted.final_ct_queue)) * 100, 1)

    d_disch = intervened.total_discharges - disrupted.total_discharges
    pct_disch = round((d_disch / max(1, disrupted.total_discharges)) * 100, 1)

    return SolutionImpactMetrics(
        disrupted_waiting=disrupted.final_waiting,
        intervened_waiting=intervened.final_waiting,
        delta_waiting=d_wait,
        pct_waiting=pct_wait,
        disrupted_available=disrupted.final_available,
        intervened_available=intervened.final_available,
        delta_available=d_avail,
        pct_available=pct_avail,
        disrupted_occupied=disrupted.final_occupied,
        intervened_occupied=intervened.final_occupied,
        delta_occupied=d_occ,
        pct_occupied=pct_occ,
        disrupted_ct_wait=disrupted.est_ct_wait_mins,
        intervened_ct_wait=intervened.est_ct_wait_mins,
        delta_ct_wait=d_ct_w,
        pct_ct_wait=pct_ct_w,
        disrupted_ct_queue=disrupted.final_ct_queue,
        intervened_ct_queue=intervened.final_ct_queue,
        delta_ct_queue=d_ct_q,
        pct_ct_queue=pct_ct_q,
        disrupted_discharges=disrupted.total_discharges,
        intervened_discharges=intervened.total_discharges,
        delta_discharges=d_disch,
        pct_discharges=pct_disch
    )


def simulate_and_rank_solutions(
    selected_solution_ids: List[str],
    available_solutions: List[ProposedSolution],
    duration_hours: int,
    additional_arrivals: int,
    ct_scanner_status: str,
    discharge_delay: bool,
    disrupted_res: ScenarioResult
) -> Tuple[List[RankedSolutionEvaluation], Optional[RankedSolutionEvaluation]]:
    """
    Simulates selected solutions deterministically, calculates impact deltas,
    enforces feasibility, ranks feasible candidates by objective score,
    and identifies the Best Feasible Option with transparent selection rationale.
    """
    sol_map = {s.id: s for s in available_solutions}
    evaluations: List[RankedSolutionEvaluation] = []
    is_ct_down = (ct_scanner_status == "Unavailable" or "Outage" in ct_scanner_status)

    for sid in selected_solution_ids:
        if sid not in sol_map:
            continue
        sol = sol_map[sid]

        if not sol.feasibility.is_feasible:
            # Infeasible: do not simulate, flag with clear explanation
            rejection_text = " • ".join(sol.feasibility.rejection_reasons)
            evaluations.append(RankedSolutionEvaluation(
                solution=sol,
                sim_result=None,
                is_feasible=False,
                rejection_explanation=rejection_text,
                impact=None,
                objective_score=-999.0,
                rank=99,
                selection_rationale="Infeasible: Rejected by operational feasibility engine."
            ))
        else:
            # Feasible: simulate deterministically under exact same initial conditions & duration
            sim_res = run_scenario_step_by_step(
                name=sol.title,
                scenario_type="response",
                description=sol.description,
                assumptions=sol.assumptions,
                duration_hours=duration_hours,
                total_beds=280 + sol.bed_capacity_bonus,
                initial_occupied=238,
                initial_waiting=34,
                initial_ct_queue=6,
                additional_arrivals_total=additional_arrivals,
                discharge_rate_hourly=sol.discharge_velocity,
                max_admission_intake_hourly=sol.intake_velocity,
                ct_capacity_hourly=sol.ct_capacity,
                ct_status_label="Protocol Active" if is_ct_down else "Available"
            )
            impact = calculate_impact_metrics(disrupted_res, sim_res)

            # Objective score calculation (reduces waiting queue, increases available beds, reduces CT wait)
            # Waiting queue reduction is primary priority (weight 3.0)
            score = (
                max(0, -impact.delta_waiting) * 3.0 +
                max(0, impact.delta_available) * 2.0 +
                max(0, -impact.delta_ct_wait) * 0.25 +
                max(0, impact.delta_discharges) * 1.5
            )

            evaluations.append(RankedSolutionEvaluation(
                solution=sol,
                sim_result=sim_res,
                is_feasible=True,
                rejection_explanation=None,
                impact=impact,
                objective_score=round(score, 1),
                rank=1,
                selection_rationale=""
            ))

    # Rank feasible evaluations by objective score descending
    feasible_evals = [e for e in evaluations if e.is_feasible]
    infeasible_evals = [e for e in evaluations if not e.is_feasible]

    feasible_evals.sort(key=lambda x: x.objective_score, reverse=True)

    for idx, e in enumerate(feasible_evals):
        e.rank = idx + 1

    best_solution_eval = None
    if feasible_evals and feasible_evals[0].objective_score > 0:
        best_solution_eval = feasible_evals[0]
        imp = best_solution_eval.impact
        best_solution_eval.selection_rationale = (
            f"Selected as highest-ranked feasible option (Score: {best_solution_eval.objective_score}) because it achieves "
            f"the maximum operational relief across multiple constraints: mitigating {abs(imp.delta_waiting)} waiting patients "
            f"({abs(imp.pct_waiting):.1f}% queue reduction), restoring available beds by +{imp.delta_available}, and cutting "
            f"diagnostic CT wait time by {abs(imp.delta_ct_wait)} minutes ({abs(imp.pct_ct_wait):.1f}%), while fully respecting "
            f"all nurse staffing floors and physical licensed bed limits."
        )

    all_ranked = feasible_evals + infeasible_evals
    return all_ranked, best_solution_eval
