"""
=============================================================================
 MediFlow AI — Hospital Resource Simulation Engine
=============================================================================
 Module: simulator.py
 Purpose: Modular simulation engine for Hospital What-If scenarios.

 Models:
 - Physical bed capacity, occupied beds, and available beds (with hard capacity caps).
 - Hourly inflow of patient arrivals and inpatient bed requests.
 - Inpatient discharge rates and the impact of discharge delays.
 - Diagnostic capacity and CT scanner availability / backlog.
 - Three side-by-side comparative scenarios:
     A. Baseline (Normal operations)
     B. Disrupted (User-configured disruption)
     C. Proposed Response (Illustrative operational mitigation)

 All data generated and processed is 100% synthetic for demonstration.
=============================================================================
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import pandas as pd


@dataclass
class ScenarioResult:
    """Stores the complete simulation outcome for a single scenario."""
    name: str
    scenario_type: str  # 'baseline', 'disrupted', 'response'
    description: str
    assumptions: List[str]
    total_beds: int
    initial_occupied: int
    initial_waiting: int
    final_occupied: int
    final_available: int
    final_waiting: int
    peak_waiting: int
    projected_bed_demand: int
    total_admissions_fulfilled: int
    total_discharges: int
    final_occupancy_pct: float
    ct_scanner_status: str
    final_ct_queue: int
    est_ct_wait_mins: int
    ct_scans_completed: int
    hourly_df: pd.DataFrame
    primary_bottleneck: str


def run_scenario_step_by_step(
    name: str,
    scenario_type: str,
    description: str,
    assumptions: List[str],
    duration_hours: int,
    total_beds: int,
    initial_occupied: int,
    initial_waiting: int,
    initial_ct_queue: int,
    additional_arrivals_total: int,
    discharge_rate_hourly: float,
    max_admission_intake_hourly: int,
    ct_capacity_hourly: float,
    ct_status_label: str
) -> ScenarioResult:
    """
    Simulates hospital resource flow hour-by-hour over the specified duration.

    Guarantees:
    - Occupied beds never exceed physical total_beds.
    - Available beds is always >= 0.
    - Excess demand rolls directly into the waiting queue.
    - No negative numbers are generated.
    """
    # 24-hour baseline arrival curve (realistic diurnal pattern)
    hourly_base_arrivals = [8, 6, 5, 4, 4, 5, 7, 10, 13, 15, 14, 13, 14, 15, 16, 14, 12, 10, 9, 8, 7, 6, 5, 4]
    # Baseline CT scan demand pattern (scans requested per hour)
    hourly_base_ct_demands = [4, 3, 3, 2, 2, 3, 4, 6, 7, 8, 7, 6, 7, 8, 8, 7, 6, 5, 5, 4, 4, 3, 3, 3]

    occupied_beds = initial_occupied
    waiting_queue = initial_waiting
    ct_queue = initial_ct_queue

    total_bed_demand = 0
    total_admissions_fulfilled = 0
    total_discharges = 0
    total_ct_completed = 0
    peak_waiting = initial_waiting

    hourly_rows = []

    # Distribute additional arrivals smoothly across the duration
    extra_arrival_rate = additional_arrivals_total / max(1, duration_hours)

    for h in range(1, duration_hours + 1):
        # 1. Incoming Patient Influx
        base_arr = hourly_base_arrivals[(h - 1) % len(hourly_base_arrivals)]
        total_hour_arrivals = round(base_arr + extra_arrival_rate)

        # Proportion needing inpatient bed admission (~35%)
        new_bed_requests = round(total_hour_arrivals * 0.35)
        total_bed_demand += new_bed_requests

        # 2. Patient Discharges
        # Only currently occupied beds can be discharged
        hour_discharges = min(round(discharge_rate_hourly), occupied_beds)
        occupied_beds = max(0, occupied_beds - hour_discharges)
        total_discharges += hour_discharges

        # Physical available beds right now
        available_beds = max(0, total_beds - occupied_beds)

        # 3. Patient Admissions
        # Patients seeking admission include existing waiting queue + new bed requests
        waiting_queue += new_bed_requests

        # Admissions are constrained by:
        # a) Physical available beds
        # b) Hospital intake processing capacity (staff turnover rate)
        admitted = min(waiting_queue, available_beds, max_admission_intake_hourly)
        occupied_beds += admitted
        waiting_queue = max(0, waiting_queue - admitted)
        available_beds = max(0, total_beds - occupied_beds)
        total_admissions_fulfilled += admitted

        if waiting_queue > peak_waiting:
            peak_waiting = waiting_queue

        # 4. Diagnostic CT Scanner Workflow
        base_ct = hourly_base_ct_demands[(h - 1) % len(hourly_base_ct_demands)]
        extra_ct = round(extra_arrival_rate * 0.28)  # ~28% of surge arrivals require CT imaging
        hour_ct_demand = base_ct + extra_ct

        # Scans completed this hour
        possible_scans = min(ct_queue + hour_ct_demand, round(ct_capacity_hourly))
        ct_queue = max(0, ct_queue + hour_ct_demand - round(ct_capacity_hourly))
        total_ct_completed += possible_scans

        est_ct_wait_minutes = round((ct_queue / max(1.0, ct_capacity_hourly)) * 60)

        # Record hourly snapshot
        hourly_rows.append({
            "Hour": f"Hour {h}",
            "HourNum": h,
            "Occupied Beds": occupied_beds,
            "Available Beds": available_beds,
            "Patients Waiting": waiting_queue,
            "Bed Demand (New)": new_bed_requests,
            "Discharges": hour_discharges,
            "CT Queue": ct_queue,
            "CT Wait Time (min)": est_ct_wait_minutes,
            "CT Scans Completed": possible_scans
        })

    hourly_df = pd.DataFrame(hourly_rows)
    final_occupancy_pct = round((occupied_beds / max(1, total_beds)) * 100, 1)
    final_ct_wait = round((ct_queue / max(1.0, ct_capacity_hourly)) * 60)

    # Automated Primary Bottleneck Classification
    if final_occupancy_pct >= 96.0 or waiting_queue >= 20:
        bottleneck = "Inpatient Bed Shortage (Critical Bed Saturation)"
    elif final_ct_wait >= 100 or ct_queue >= 15:
        bottleneck = "Diagnostic Bottleneck (CT Scanner Backlog)"
    elif discharge_rate_hourly <= 3.2:
        bottleneck = "Discharge Processing Delay (Bed Turnover Congestion)"
    else:
        bottleneck = "Normal Operational Buffer (Capacity Stable)"

    return ScenarioResult(
        name=name,
        scenario_type=scenario_type,
        description=description,
        assumptions=assumptions,
        total_beds=total_beds,
        initial_occupied=initial_occupied,
        initial_waiting=initial_waiting,
        final_occupied=occupied_beds,
        final_available=available_beds,
        final_waiting=waiting_queue,
        peak_waiting=peak_waiting,
        projected_bed_demand=total_bed_demand,
        total_admissions_fulfilled=total_admissions_fulfilled,
        total_discharges=total_discharges,
        final_occupancy_pct=final_occupancy_pct,
        ct_scanner_status=ct_status_label,
        final_ct_queue=ct_queue,
        est_ct_wait_mins=final_ct_wait,
        ct_scans_completed=total_ct_completed,
        hourly_df=hourly_df,
        primary_bottleneck=bottleneck
    )


def run_hospital_what_if_simulation(
    additional_arrivals: int = 30,
    ct_scanner_status: str = "Unavailable",
    discharge_delay: bool = True,
    duration_hours: int = 12
) -> Tuple[ScenarioResult, ScenarioResult, ScenarioResult, Dict[str, Any]]:
    """
    Executes the three side-by-side simulation scenarios:
    1. Scenario A: Baseline (Normal Hospital Operations)
    2. Scenario B: Disrupted (User Configured Disruption)
    3. Scenario C: Proposed Response (Illustrative Hospital Response Strategy)

    Returns:
    - baseline_scenario
    - disrupted_scenario
    - response_scenario
    - meta dictionary with summary delta metrics
    """
    # Validation & Safe Clamping
    additional_arrivals = max(0, min(60, int(additional_arrivals)))
    if duration_hours not in [4, 8, 12, 24]:
        duration_hours = 12

    # Baseline hospital parameters (consistent with Phase 1 dashboard)
    base_total_beds = 280
    base_initial_occupied = 238
    base_initial_waiting = 34
    base_initial_ct_queue = 6

    # -------------------------------------------------------------------------
    # Scenario A: Baseline (Normal Operations)
    # -------------------------------------------------------------------------
    baseline_assumptions = [
        "Normal baseline patient arrival volume (0 unexpected surge arrivals).",
        "Both CT diagnostic scanners operational (8 scans/hour throughput).",
        "Standard discharge rate maintained (~5.5 discharges/hour).",
        "Staffing admission processing capacity at normal level (6 admissions/hour).",
        "Physical bed capacity strictly maintained at 280 beds."
    ]
    baseline = run_scenario_step_by_step(
        name="Scenario A: Baseline (Normal Operations)",
        scenario_type="baseline",
        description="Standard operating conditions without unexpected disruptions.",
        assumptions=baseline_assumptions,
        duration_hours=duration_hours,
        total_beds=base_total_beds,
        initial_occupied=base_initial_occupied,
        initial_waiting=base_initial_waiting,
        initial_ct_queue=base_initial_ct_queue,
        additional_arrivals_total=0,
        discharge_rate_hourly=5.5,
        max_admission_intake_hourly=6,
        ct_capacity_hourly=8.0,
        ct_status_label="Available (Both Units Operational)"
    )

    # -------------------------------------------------------------------------
    # Scenario B: Disrupted (User Configured Disruption)
    # -------------------------------------------------------------------------
    is_ct_down = (ct_scanner_status == "Unavailable" or "Outage" in ct_scanner_status)
    disrupted_ct_cap = 3.8 if is_ct_down else 8.0
    disrupted_ct_label = "Unavailable (1 Scanner Down - 50% Outage)" if is_ct_down else "Available (Both Operational)"
    disrupted_discharge_rate = 2.6 if discharge_delay else 5.5
    disrupted_intake_rate = 5 if is_ct_down else 6  # Diagnostic delay slightly slows admission workup

    disrupted_assumptions = [
        f"Unexpected Surge Arrivals: +{additional_arrivals} additional patients arriving across {duration_hours} hours.",
        f"CT Scanner Status: {disrupted_ct_label} ({disrupted_ct_cap:.1f} scans/hour capacity).",
        f"Patient Discharge Delay: {'ON (Discharges dropped to ~2.6/hr due to clearance & transport delays)' if discharge_delay else 'OFF (Standard ~5.5 discharges/hr)'}.",
        "Physical bed capacity strictly capped at 280 beds (zero overflow beds available)."
    ]
    disrupted = run_scenario_step_by_step(
        name="Scenario B: Disrupted",
        scenario_type="disrupted",
        description="Compounding strain caused by the selected disruptions.",
        assumptions=disrupted_assumptions,
        duration_hours=duration_hours,
        total_beds=base_total_beds,
        initial_occupied=base_initial_occupied,
        initial_waiting=base_initial_waiting,
        initial_ct_queue=base_initial_ct_queue,
        additional_arrivals_total=additional_arrivals,
        discharge_rate_hourly=disrupted_discharge_rate,
        max_admission_intake_hourly=disrupted_intake_rate,
        ct_capacity_hourly=disrupted_ct_cap,
        ct_status_label=disrupted_ct_label
    )

    # -------------------------------------------------------------------------
    # Scenario C: Proposed Response (Illustrative Response Strategy)
    # -------------------------------------------------------------------------
    # Proactive operational responses modeled:
    # 1. Flex Surge Beds: If surge arrivals >= 25, deploy +12 temporary flex observation beds.
    # 2. Discharge Coordination: If discharge delays exist, activate dedicated discharge lounge (+3.4 discharges/hr).
    # 3. Diagnostic Contingency: If CT scanner is down, divert non-emergency elective scans & utilize emergency mobile CT protocol.
    flex_beds = 12 if additional_arrivals >= 25 else 0
    response_total_beds = base_total_beds + flex_beds

    response_discharge_rate = 6.2 if discharge_delay else 5.8
    response_ct_cap = 7.0 if is_ct_down else 8.0
    response_ct_label = "Emergency Protocol Active (7.0 scans/hr)" if is_ct_down else "Available"
    response_intake_rate = 7  # Expedited triage nursing intake team

    response_assumptions = [
        f"Surge Bed Activation: {'+12 Temporary observation flex beds opened (Total: 292 beds)' if flex_beds > 0 else 'Standard 280 beds maintained'}.",
        f"Discharge Acceleration: {'Dedicated Discharge Lounge activated (+3.6 discharges/hr expedited)' if discharge_delay else 'Standard discharge coordination'}.",
        f"Diagnostic Priority Protocol: {'Divert elective scans offsite & deploy fast emergency CT protocol' if is_ct_down else 'Standard diagnostic throughput'}.",
        "Illustrative model: results represent prospective exploratory mitigation, not guaranteed clinical outcomes."
    ]
    response = run_scenario_step_by_step(
        name="Scenario C: Proposed Response",
        scenario_type="response",
        description="Illustrative operational strategy to restore hospital flow and relieve bottlenecks.",
        assumptions=response_assumptions,
        duration_hours=duration_hours,
        total_beds=response_total_beds,
        initial_occupied=base_initial_occupied,
        initial_waiting=base_initial_waiting,
        initial_ct_queue=base_initial_ct_queue,
        additional_arrivals_total=additional_arrivals,
        discharge_rate_hourly=response_discharge_rate,
        max_admission_intake_hourly=response_intake_rate,
        ct_capacity_hourly=response_ct_cap,
        ct_status_label=response_ct_label
    )

    # Meta Comparison Metrics
    meta = {
        "duration_hours": duration_hours,
        "additional_arrivals": additional_arrivals,
        "ct_scanner_status": ct_scanner_status,
        "discharge_delay": discharge_delay,
        "waiting_difference": disrupted.final_waiting - baseline.final_waiting,
        "waiting_mitigated": max(0, disrupted.final_waiting - response.final_waiting),
        "primary_disruption_bottleneck": disrupted.primary_bottleneck,
        "flex_beds_activated": flex_beds
    }

    return baseline, disrupted, response, meta
