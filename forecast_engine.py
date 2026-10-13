"""
=============================================================================
 MediFlow AI — Intelligent Patient Arrival Forecasting Engine
=============================================================================
 Module: forecast_engine.py
 Purpose: Phase 4 forecasting engine providing:
   1. Reproducible time-stamped synthetic arrival data generator (multi-week history)
   2. Modular statistical forecasting model (Time-of-Day & Day-of-Week Seasonal Profile)
   3. Multi-horizon forecasting (4h, 8h, 12h, 24h) with non-negative count estimates
   4. Uncertainty / prediction intervals (80% and 95%) derived from historical variance
   5. Chronological holdout evaluation (MAE, RMSE) without temporal data leakage
   6. Demand-surge threshold alerts and translation to What-If Simulator inputs
   7. Proactive decision-support link with Phase 3 capacity recovery & staffing

 All calculations are deterministic, reproducible, and documented.
 Synthetic demonstration data only.
=============================================================================
"""

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from simulator import run_scenario_step_by_step, ScenarioResult


# =============================================================================
# 1. Documented Forecast Thresholds & Operational Constants
# =============================================================================
FORECAST_THRESHOLDS = {
    "HOURLY_SURGE_PEAK": 14,      # >=14 arrivals in a single hour indicates acute surge peak
    "HOURLY_SURGE_CRITICAL": 18,  # >=18 arrivals indicates extreme volume stress
    "CUMULATIVE_SURGE_RATIO": 1.25, # Forecast volume > 125% of baseline capacity signals deficit
    "MIN_HISTORICAL_DAYS": 14,    # Minimum 14 days of history recommended for stable seasonality
}


# =============================================================================
# 2. Data Structures for Forecasting
# =============================================================================

@dataclass
class ForecastIntervalPoint:
    """Represents a forecasted arrival estimate for a single future hour."""
    horizon_hour: int             # 1, 2, ..., H
    timestamp: str                # e.g., '2026-10-13 14:00'
    hour_of_day: int              # 0 to 23
    day_name: str                 # 'Monday', 'Tuesday', ...
    predicted_arrivals: int       # Rounded non-negative integer count
    raw_estimate: float           # Continuous point estimate
    lower_ci_80: int              # 80% prediction interval lower bound
    upper_ci_80: int              # 80% prediction interval upper bound
    lower_ci_95: int              # 95% prediction interval lower bound
    upper_ci_95: int              # 95% prediction interval upper bound
    category_breakdown: Dict[str, int] = field(default_factory=dict)


@dataclass
class ForecastResult:
    """Complete multi-hour forecast output."""
    horizon_hours: int            # 4, 8, 12, or 24
    start_time: str
    end_time: str
    target_category: str          # 'All Facility', 'Emergency (Red)', 'Urgent (Yellow)', 'Routine (Green)'
    interval_forecasts: List[ForecastIntervalPoint]
    total_predicted_arrivals: int
    mean_arrivals_per_hour: float
    peak_predicted_hour: str
    peak_predicted_count: int
    is_surge_warning_active: bool
    surge_warning_message: str
    baseline_expected_arrivals: int
    incremental_surge_above_baseline: int
    model_name: str
    method_description: str
    assumptions: List[str]
    limitations: List[str]


@dataclass
class ForecastEvaluationMetrics:
    """Holdout evaluation metrics computed on chronological test split."""
    model_name: str
    baseline_model_name: str
    test_period_start: str
    test_period_end: str
    test_observations_count: int
    mae: float                    # Mean Absolute Error
    rmse: float                   # Root Mean Squared Error
    baseline_mae: float           # Naive baseline MAE
    baseline_rmse: float          # Naive baseline RMSE
    error_reduction_pct: float    # Relative accuracy improvement over naive baseline
    interpretation: str           # Plain-language explanation for hospital managers


@dataclass
class ProactiveCapacityAlert:
    """Early-warning proactive capacity indicator triggered by arrival forecast."""
    interval_hour: int
    timestamp: str
    predicted_arrivals: int
    projected_bed_deficit: int
    projected_waiting_queue: int
    severity: str                 # 'CRITICAL', 'WARNING', 'OPTIMAL'
    operational_action: str
    lead_time_recommended: str


# =============================================================================
# 3. Reproducible Time-Stamped Synthetic Arrival Generator
# =============================================================================

def generate_historical_arrival_dataset(
    start_date: str = "2026-08-18",
    num_days: int = 56,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generates reproducible, time-stamped hourly patient arrival records across multiple weeks.
    
    Models realistic clinical patterns:
    - Diurnal cycle: Trough at 02:00-05:00, morning rise, peak in afternoon 14:00-18:00
    - Day-of-week seasonality: Monday/Tuesday post-weekend surge (+15-20%), weekend shift
    - Multi-category breakdown: Emergency (Red), Urgent (Yellow), Routine (Green)
    - Reproducible stochastic variation using fixed random seed
    - Guaranteed non-negative count values
    """
    rng = np.random.default_rng(seed)
    
    # 24-hour diurnal profile multipliers (relative to hourly average)
    # Peak at 14:00 - 16:00, lowest at 03:00 - 05:00
    diurnal_profile = [
        0.42, 0.32, 0.25, 0.22, 0.25, 0.38, 0.62, 0.95,
        1.25, 1.48, 1.55, 1.45, 1.38, 1.45, 1.62, 1.68,
        1.55, 1.35, 1.15, 0.95, 0.82, 0.68, 0.55, 0.45
    ]
    
    # Day-of-week multipliers: Monday=0, Tuesday=1, ..., Sunday=6
    # Mondays and Tuesdays experience post-weekend spikes
    dow_multipliers = [1.18, 1.12, 1.02, 0.98, 0.96, 0.88, 0.86]

    base_emergency_rate = 4.2
    base_urgent_rate = 3.6
    base_routine_rate = 2.8

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    total_hours = num_days * 24

    records = []
    for h_idx in range(total_hours):
        current_dt = start_dt + timedelta(hours=h_idx)
        hour = current_dt.hour
        dow = current_dt.weekday()
        day_name = current_dt.strftime("%A")

        d_factor = diurnal_profile[hour]
        w_factor = dow_multipliers[dow]

        # Calculate lambda for Poisson generation with slight overdispersion
        lam_emer = max(0.5, base_emergency_rate * d_factor * w_factor)
        lam_urg = max(0.4, base_urgent_rate * d_factor * w_factor)
        # Routine clinic arrivals taper sharply at night (20:00 - 07:00)
        routine_taper = 0.05 if (hour < 7 or hour > 19) else 1.0
        lam_rout = max(0.0, base_routine_rate * d_factor * w_factor * routine_taper)

        # Draw Poisson arrivals with stochastic variation
        c_emer = int(rng.poisson(lam_emer))
        c_urg = int(rng.poisson(lam_urg))
        c_rout = int(rng.poisson(lam_rout))

        total_arr = c_emer + c_urg + c_rout

        records.append({
            "timestamp": current_dt.strftime("%Y-%m-%d %H:00"),
            "datetime": current_dt,
            "hour": hour,
            "day_of_week": dow,
            "day_name": day_name,
            "is_weekend": (dow >= 5),
            "emergency_red": c_emer,
            "urgent_yellow": c_urg,
            "routine_green": c_rout,
            "total_arrivals": total_arr
        })

    df = pd.DataFrame(records)
    return df


# =============================================================================
# 4. Seasonal Profile Forecasting Model
# =============================================================================

def train_seasonal_profile_model(
    df_history: pd.DataFrame,
    target_column: str = "total_arrivals"
) -> Dict[Tuple[int, int], Tuple[float, float]]:
    """
    Learns empirical historical mean and standard deviation conditioned on:
    (day_of_week, hour_of_day).
    
    Returns:
      Dictionary mapping (dow, hour) -> (mean_arrivals, std_arrivals)
    """
    if df_history.empty or target_column not in df_history.columns:
        # Fallback uniform default profile if data is empty
        return {(d, h): (10.0, 3.0) for d in range(7) for h in range(24)}

    profile = {}
    grouped = df_history.groupby(["day_of_week", "hour"])[target_column]
    means = grouped.mean()
    stds = grouped.std().fillna(1.5)  # Default fallback variance

    for (d, h), m in means.items():
        s = stds.get((d, h), 1.5)
        profile[(d, h)] = (float(m), float(max(0.8, s)))

    return profile


def generate_patient_arrival_forecast(
    df_history: pd.DataFrame,
    horizon_hours: int = 12,
    target_category: str = "All Facility",
    recent_trend_factor: float = 1.0
) -> ForecastResult:
    """
    Generates a deterministic, reproducible arrival forecast for horizons: 4, 8, 12, or 24 hours.
    
    Guarantees:
    - Non-negative arrival estimates
    - Integer-rounded interval arrival counts with underlying continuous rates
    - 80% and 95% empirical prediction intervals
    - Correct cumulative aggregation across selected horizon
    - Documented model assumptions and demand-surge alerts
    """
    # Map selection to historical column
    col_map = {
        "All Facility": "total_arrivals",
        "Emergency (Red)": "emergency_red",
        "Urgent (Yellow)": "urgent_yellow",
        "Routine (Green)": "routine_green"
    }
    target_col = col_map.get(target_category, "total_arrivals")

    # Validate inputs and enforce minimum data checks
    if df_history.empty or len(df_history) < 24:
        # Fallback illustrative estimate with clear labeling
        return _generate_fallback_illustrative_forecast(horizon_hours, target_category)

    # Learn empirical seasonal profile
    profile = train_seasonal_profile_model(df_history, target_col)

    # Also learn category-specific profiles for sub-breakdown
    prof_emer = train_seasonal_profile_model(df_history, "emergency_red")
    prof_urg = train_seasonal_profile_model(df_history, "urgent_yellow")
    prof_rout = train_seasonal_profile_model(df_history, "routine_green")

    # Determine simulation forecast start point (continues immediately from last history timestamp)
    last_dt = df_history["datetime"].max()
    forecast_points = []
    cum_total = 0
    cum_baseline = 0
    peak_count = -1
    peak_str = ""

    # Nominal baseline benchmark rate per hour
    nominal_hourly_base = 9.8 if target_category == "All Facility" else 4.0

    for step in range(1, horizon_hours + 1):
        f_dt = last_dt + timedelta(hours=step)
        f_hour = f_dt.hour
        f_dow = f_dt.weekday()
        f_day_name = f_dt.strftime("%A")

        mean_val, std_val = profile.get((f_dow, f_hour), (10.0, 2.5))
        
        # Apply gentle recent trend adjustment (bounded between 0.8 and 1.25)
        clamped_trend = max(0.8, min(1.25, recent_trend_factor))
        adjusted_val = max(0.0, mean_val * clamped_trend)
        point_pred = int(round(adjusted_val))

        # Prediction intervals (80% uses z=1.28, 95% uses z=1.96)
        lower_80 = max(0, int(math.floor(adjusted_val - 1.28 * std_val)))
        upper_80 = max(point_pred, int(math.ceil(adjusted_val + 1.28 * std_val)))

        lower_95 = max(0, int(math.floor(adjusted_val - 1.96 * std_val)))
        upper_95 = max(point_pred, int(math.ceil(adjusted_val + 1.96 * std_val)))

        # Sub-category distributions
        e_val = int(round(prof_emer.get((f_dow, f_hour), (4.0, 1.0))[0] * clamped_trend))
        u_val = int(round(prof_urg.get((f_dow, f_hour), (3.5, 1.0))[0] * clamped_trend))
        r_val = max(0, point_pred - e_val - u_val) if target_category == "All Facility" else 0

        point = ForecastIntervalPoint(
            horizon_hour=step,
            timestamp=f_dt.strftime("%Y-%m-%d %H:00"),
            hour_of_day=f_hour,
            day_name=f_day_name,
            predicted_arrivals=point_pred,
            raw_estimate=adjusted_val,
            lower_ci_80=lower_80,
            upper_ci_80=upper_80,
            lower_ci_95=lower_95,
            upper_ci_95=upper_95,
            category_breakdown={"Emergency": e_val, "Urgent": u_val, "Routine": r_val}
        )
        forecast_points.append(point)

        cum_total += point_pred
        cum_baseline += int(round(nominal_hourly_base))

        if point_pred > peak_count:
            peak_count = point_pred
            peak_str = f"{f_dt.strftime('%H:00')} ({f_day_name})"

    mean_per_hour = round(cum_total / max(1, horizon_hours), 1)
    inc_surge = max(0, cum_total - cum_baseline)

    # Surge warning detection
    is_surge = (peak_count >= FORECAST_THRESHOLDS["HOURLY_SURGE_PEAK"] or 
                (cum_total > cum_baseline * FORECAST_THRESHOLDS["CUMULATIVE_SURGE_RATIO"]))
    
    if is_surge:
        surge_msg = (
            f"🚨 DEMAND SURGE DETECTED: Forecast projects peak inflow of {peak_count} patients at {peak_str}. "
            f"Cumulative {horizon_hours}-hour arrival volume ({cum_total} patients) exceeds standard operational baseline by +{inc_surge} patients."
        )
    else:
        surge_msg = (
            f"🟢 STABLE VOLUME: Forecast projects nominal demand of {cum_total} arrivals ({mean_per_hour}/hr) "
            f"across the {horizon_hours}-hour window without acute surge threshold breaches."
        )

    start_str = forecast_points[0].timestamp if forecast_points else "N/A"
    end_str = forecast_points[-1].timestamp if forecast_points else "N/A"

    return ForecastResult(
        horizon_hours=horizon_hours,
        start_time=start_str,
        end_time=end_str,
        target_category=target_category,
        interval_forecasts=forecast_points,
        total_predicted_arrivals=cum_total,
        mean_arrivals_per_hour=mean_per_hour,
        peak_predicted_hour=peak_str,
        peak_predicted_count=peak_count,
        is_surge_warning_active=is_surge,
        surge_warning_message=surge_msg,
        baseline_expected_arrivals=cum_baseline,
        incremental_surge_above_baseline=inc_surge,
        model_name="Historical Seasonal Profile (Day-of-Week & Diurnal Decomposition)",
        method_description=(
            "Calculates empirical conditional expectations and variance conditioned on day-of-week and time-of-day. "
            "Generates non-negative point forecasts and 80%/95% prediction intervals based on historical empirical variance."
        ),
        assumptions=[
            "Diurnal time-of-day arrival rhythms remain stable over the forecast window.",
            "Day-of-week seasonal factors reflect recurrent hospital scheduling patterns.",
            "Variance is bounded using empirical historical standard deviation."
        ],
        limitations=[
            "Synthetic demonstration model; does not incorporate external meteorological data, holidays, or city-wide mass incidents.",
            "Forecast reflects volume demand only; does not predict individual patient diagnostic acuity or clinical diagnoses."
        ]
    )


def _generate_fallback_illustrative_forecast(horizon_hours: int, target_category: str) -> ForecastResult:
    """Fallback generator with explicit disclaimer when history is incomplete or missing."""
    now = datetime(2026, 10, 13, 8, 0)
    points = []
    cum = 0
    for h in range(1, horizon_hours + 1):
        f_dt = now + timedelta(hours=h)
        val = 10
        points.append(ForecastIntervalPoint(
            horizon_hour=h,
            timestamp=f_dt.strftime("%Y-%m-%d %H:00"),
            hour_of_day=f_dt.hour,
            day_name=f_dt.strftime("%A"),
            predicted_arrivals=val,
            raw_estimate=10.0,
            lower_ci_80=7,
            upper_ci_80=13,
            lower_ci_95=5,
            upper_ci_95=15,
            category_breakdown={"Emergency": 4, "Urgent": 4, "Routine": 2}
        ))
        cum += val

    return ForecastResult(
        horizon_hours=horizon_hours,
        start_time=points[0].timestamp,
        end_time=points[-1].timestamp,
        target_category=target_category,
        interval_forecasts=points,
        total_predicted_arrivals=cum,
        mean_arrivals_per_hour=10.0,
        peak_predicted_hour="14:00 (Illustrative)",
        peak_predicted_count=10,
        is_surge_warning_active=False,
        surge_warning_message="⚠️ INSUFFICIENT HISTORICAL DATA: Displaying illustrative baseline projection.",
        baseline_expected_arrivals=cum,
        incremental_surge_above_baseline=0,
        model_name="Fallback Illustrative Uniform Rate",
        method_description="Fixed default estimate due to missing historical arrival records.",
        assumptions=["Uniform hourly arrival distribution"],
        limitations=["Purely illustrative placeholder; not trained on empirical hospital data."]
    )


# =============================================================================
# 5. Chronological Holdout Model Evaluation (No Temporal Data Leakage)
# =============================================================================

def evaluate_forecast_model_performance(
    df_history: pd.DataFrame,
    test_days: int = 7
) -> ForecastEvaluationMetrics:
    """
    Evaluates forecasting performance using a strict chronological train/test split.
    
    Guarantees:
    - No temporal data leakage: Test observations are strictly strictly AFTER training observations.
    - Evaluates count metrics suitable for arrival data: MAE and RMSE.
    - Compares against a 24-hour seasonal naive baseline (same hour yesterday).
    - Explains error metrics in plain language.
    """
    if df_history.empty or len(df_history) < (test_days * 24 + 48):
        return ForecastEvaluationMetrics(
            model_name="Seasonal Profile Model",
            baseline_model_name="Seasonal Naive (Lag-24)",
            test_period_start="N/A",
            test_period_end="N/A",
            test_observations_count=0,
            mae=0.0,
            rmse=0.0,
            baseline_mae=0.0,
            baseline_rmse=0.0,
            error_reduction_pct=0.0,
            interpretation="Insufficient historical duration for chronological train/test holdout split."
        )

    # Strict chronological split: last test_days * 24 hours are holdout
    split_idx = len(df_history) - (test_days * 24)
    train_df = df_history.iloc[:split_idx].copy()
    test_df = df_history.iloc[split_idx:].copy()

    # Train profile on historical training partition ONLY
    profile = train_seasonal_profile_model(train_df, "total_arrivals")

    actuals = []
    preds = []
    naive_preds = []

    for idx in range(len(test_df)):
        row = test_df.iloc[idx]
        actual = row["total_arrivals"]
        dow = row["day_of_week"]
        hr = row["hour"]

        pred = profile.get((dow, hr), (10.0, 2.0))[0]

        # Naive baseline: arrival count exactly 24 hours prior
        prev_idx = split_idx + idx - 24
        if prev_idx >= 0 and prev_idx < len(df_history):
            naive_val = df_history.iloc[prev_idx]["total_arrivals"]
        else:
            naive_val = train_df["total_arrivals"].mean()

        actuals.append(actual)
        preds.append(pred)
        naive_preds.append(naive_val)

    actuals_arr = np.array(actuals)
    preds_arr = np.array(preds)
    naive_arr = np.array(naive_preds)

    # MAE and RMSE
    mae = float(np.mean(np.abs(actuals_arr - preds_arr)))
    rmse = float(np.sqrt(np.mean((actuals_arr - preds_arr) ** 2)))

    base_mae = float(np.mean(np.abs(actuals_arr - naive_arr)))
    base_rmse = float(np.sqrt(np.mean((actuals_arr - naive_arr) ** 2)))

    err_reduction = round(max(0.0, ((base_mae - mae) / max(0.01, base_mae)) * 100), 1)

    t_start = test_df["timestamp"].min()
    t_end = test_df["timestamp"].max()

    interp = (
        f"Chronological holdout validation across {len(test_df)} unseen test hours ({test_days} full days):\n"
        f"• Seasonal Model MAE: {mae:.2f} patients/hour (on average, forecast deviates by ~{mae:.1f} patients).\n"
        f"• Naive Baseline MAE: {base_mae:.2f} patients/hour.\n"
        f"• Accuracy Gain: {err_reduction}% error reduction over simple 24-hour naive persistence.\n"
        "Strict temporal split prevents future data leakage."
    )

    return ForecastEvaluationMetrics(
        model_name="Historical Seasonal Profile (Day-of-Week & Diurnal Decomposition)",
        baseline_model_name="Seasonal Naive Persistence (24-Hour Lag)",
        test_period_start=t_start,
        test_period_end=t_end,
        test_observations_count=len(test_df),
        mae=round(mae, 2),
        rmse=round(rmse, 2),
        baseline_mae=round(base_mae, 2),
        baseline_rmse=round(base_rmse, 2),
        error_reduction_pct=err_reduction,
        interpretation=interp
    )


# =============================================================================
# 6. Proactive Decision-Support Link (Connecting to Simulator & Phase 3)
# =============================================================================

def evaluate_forecast_driven_capacity_risk(
    forecast: ForecastResult,
    summary: Dict[str, Any],
    df_departments: pd.DataFrame
) -> List[ProactiveCapacityAlert]:
    """
    Projects cumulative resource strain over the forecast horizon by comparing
    projected arrival demand against ready available beds and discharge velocity.
    Generates proactive early-warning recommendations for clinical leadership review.
    """
    alerts = []
    ready_beds = summary.get("ready_available_beds", 6)
    standard_discharge_velocity = 5.5  # standard departures per hour

    cum_arrivals = 0
    cum_discharges = 0.0

    for pt in forecast.interval_forecasts:
        cum_arrivals += pt.predicted_arrivals
        cum_discharges += standard_discharge_velocity

        # Estimated net bed deficit if arrivals exceed cumulative departures + initial ready buffer
        net_buffer = ready_beds + int(cum_discharges)
        deficit = max(0, cum_arrivals - net_buffer)

        if deficit > 15:
            alerts.append(ProactiveCapacityAlert(
                interval_hour=pt.horizon_hour,
                timestamp=pt.timestamp,
                predicted_arrivals=pt.predicted_arrivals,
                projected_bed_deficit=deficit,
                projected_waiting_queue=deficit + 10,
                severity="CRITICAL",
                operational_action="Pre-authorize Opening of 12 Surge Observation Flex Beds & Expedite Morning Discharge Sign-Offs",
                lead_time_recommended=f"Initiate protocol at least 2 hours prior to {pt.timestamp}"
            ))
        elif deficit > 5:
            alerts.append(ProactiveCapacityAlert(
                interval_hour=pt.horizon_hour,
                timestamp=pt.timestamp,
                predicted_arrivals=pt.predicted_arrivals,
                projected_bed_deficit=deficit,
                projected_waiting_queue=deficit + 5,
                severity="WARNING",
                operational_action="Dispatch Priority EVS Cleaning Strike Team to Turn Over Dirty Inpatient Beds",
                lead_time_recommended=f"Initiate cleaning sweep 90 minutes prior to {pt.timestamp}"
            ))

    return alerts


def translate_forecast_to_simulator_parameters(
    forecast: ForecastResult,
    ct_scanner_status: str = "Available",
    discharge_delay: bool = False
) -> Dict[str, Any]:
    """
    Translates a generated arrival forecast into structured parameters
    for direct execution in run_scenario_step_by_step().
    
    Prevents double-counting:
    - Isolates the incremental surge above baseline: additional_arrivals = incremental_surge
    - Matches simulation duration to forecast horizon
    """
    return {
        "duration_hours": forecast.horizon_hours,
        "additional_arrivals_total": forecast.incremental_surge_above_baseline,
        "ct_scanner_status": ct_scanner_status,
        "discharge_delay": discharge_delay,
        "forecast_total": forecast.total_predicted_arrivals,
        "forecast_mean_hourly": forecast.mean_arrivals_per_hour,
        "forecast_peak_hour": forecast.peak_predicted_hour,
        "forecast_peak_count": forecast.peak_predicted_count
    }
