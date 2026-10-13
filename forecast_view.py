"""
=============================================================================
 MediFlow AI — Patient Arrival Forecasting UI View
=============================================================================
 Module: forecast_view.py
 Purpose: Renders the complete Phase 4 Patient Arrival Forecasting page:
          1. Forecast Controls (horizon, category, confidence interval)
          2. Forecast Visualization (history + prediction + uncertainty ribbons)
          3. Summary Metrics & Surge Warning Alerts
          4. Chronological Holdout Quality Evaluation (MAE, RMSE vs Naive)
          5. Direct What-If Simulator Integration ("Use Forecast in Simulator")
          6. Phase 3 Proactive Capacity Risk & Actionable Recommendations
=============================================================================
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
from forecast_engine import (
    generate_historical_arrival_dataset,
    generate_patient_arrival_forecast,
    evaluate_forecast_model_performance,
    evaluate_forecast_driven_capacity_risk,
    translate_forecast_to_simulator_parameters,
    FORECAST_THRESHOLDS,
    ForecastResult,
    ForecastEvaluationMetrics
)
from simulator import run_scenario_step_by_step, ScenarioResult


def render_patient_forecasting(
    summary: dict,
    df_departments: pd.DataFrame,
    df_resources: pd.DataFrame
):
    """
    Renders the complete Phase 4 Patient Arrival Forecasting interface.
    """
    # -------------------------------------------------------------------------
    # Brand Header
    # -------------------------------------------------------------------------
    st.markdown("""
    <div class="brand-header-box">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
            <div>
                <div class="brand-title-text">🏥 MediFlow AI</div>
                <div class="brand-subtitle-text">Intelligent Hospital Resource Optimization</div>
            </div>
            <div>
                <span class="facility-badge">📍 Metro General Hospital</span>
                <span class="live-telemetry-badge">🔮 Phase 4: Arrival Forecasting</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_title_l, col_title_r = st.columns([3, 1])
    with col_title_l:
        st.subheader("🔮 Intelligent Patient Arrival Forecasting")
        st.caption("Statistical time-series arrival projections with empirical uncertainty bounds, holdout validation, and simulation integration.")
    with col_title_r:
        st.markdown('<div style="text-align: right; padding-top: 6px;"><span class="badge-demo">PREDICTIVE ANALYTICS</span></div>', unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 1. Historical Data Loading (Cached & Reproducible)
    # -------------------------------------------------------------------------
    @st.cache_data
    def get_cached_historical_data():
        return generate_historical_arrival_dataset(start_date="2026-08-18", num_days=56, seed=42)

    df_history = get_cached_historical_data()

    # -------------------------------------------------------------------------
    # 2. Interactive Forecast Controls
    # -------------------------------------------------------------------------
    with st.container(border=True):
        col_ctrl1, col_ctrl2, col_ctrl3, col_ctrl4 = st.columns(4)

        with col_ctrl1:
            horizon_choice = st.selectbox(
                "Forecast Horizon",
                options=[4, 8, 12, 24],
                index=2,
                format_func=lambda h: f"Next {h} Hours",
                key="fc_horizon",
                help="Select the forward-looking projection window."
            )

        with col_ctrl2:
            category_choice = st.selectbox(
                "Triage / Department Category",
                options=["All Facility", "Emergency (Red)", "Urgent (Yellow)", "Routine (Green)"],
                index=0,
                key="fc_category",
                help="Filter forecasts by patient triage acuity classification."
            )

        with col_ctrl3:
            ci_level = st.selectbox(
                "Uncertainty Interval",
                options=["80% Prediction Interval", "95% Prediction Interval"],
                index=0,
                key="fc_ci",
                help="Empirical confidence ribbon derived from historical variance."
            )

        with col_ctrl4:
            st.markdown('<div style="height: 28px;"></div>', unsafe_allow_html=True)
            refresh_clicked = st.button("🔄 Refresh Forecast", type="primary", use_container_width=True)

    horizon_hours = int(horizon_choice)
    target_category = str(category_choice)
    is_95_ci = ("95%" in ci_level)

    # Generate Forecast
    forecast = generate_patient_arrival_forecast(
        df_history=df_history,
        horizon_hours=horizon_hours,
        target_category=target_category,
        recent_trend_factor=1.05  # Slight afternoon active momentum
    )

    # Holdout Model Evaluation (Strict chronological split)
    @st.cache_data
    def get_cached_evaluation(test_days: int = 7):
        return evaluate_forecast_model_performance(df_history, test_days=test_days)

    eval_metrics = get_cached_evaluation(test_days=7)

    # Capacity alerts
    capacity_alerts = evaluate_forecast_driven_capacity_risk(forecast, summary, df_departments)

    st.markdown("<br>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 3. Forecast Summary KPIs & Surge Alert Banner
    # -------------------------------------------------------------------------
    col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
    with col_kpi1:
        st.metric(
            label=f"Expected Total Inflow ({horizon_hours}h)",
            value=f"{forecast.total_predicted_arrivals} Patients",
            delta=f"+{forecast.incremental_surge_above_baseline} vs Baseline Buffer" if forecast.incremental_surge_above_baseline > 0 else "Nominal buffer",
            delta_color="inverse" if forecast.incremental_surge_above_baseline > 0 else "off"
        )
    with col_kpi2:
        st.metric(
            label="Mean Arrival Velocity",
            value=f"{forecast.mean_arrivals_per_hour} / hour",
            delta=f"Target Intake: ~10/hr",
            delta_color="off"
        )
    with col_kpi3:
        st.metric(
            label="Peak Arrival Interval",
            value=f"{forecast.peak_predicted_count} Patients",
            delta=f"at {forecast.peak_predicted_hour}",
            delta_color="inverse" if forecast.peak_predicted_count >= FORECAST_THRESHOLDS["HOURLY_SURGE_PEAK"] else "off"
        )
    with col_kpi4:
        holdout_mae = eval_metrics.mae
        st.metric(
            label="Holdout MAE (Unseen Test)",
            value=f"±{holdout_mae} / hour",
            delta=f"{eval_metrics.error_reduction_pct}% Gain vs Naive",
            delta_color="normal"
        )

    # Demand Surge Banner
    if forecast.is_surge_warning_active:
        st.markdown(f"""
        <div style="background-color: #fff1f2; border: 1px solid #fecdd3; border-left: 5px solid #e11d48; padding: 16px 20px; border-radius: 12px; margin: 16px 0;">
            <div style="font-weight: 700; font-size: 1.05rem; color: #9f1239; margin-bottom: 4px;">
                {forecast.surge_warning_message}
            </div>
            <div style="font-size: 0.9rem; color: #4c0519;">
                ⚠️ <strong>Operational Action Required:</strong> Inpatient beds and triage stretchers should be pre-cleared before the arrival peak at {forecast.peak_predicted_hour}.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background-color: #f0fdf4; border: 1px solid #bbf7d0; border-left: 5px solid #16a34a; padding: 14px 18px; border-radius: 12px; margin: 16px 0;">
            <div style="font-weight: 700; font-size: 1rem; color: #166534;">
                {forecast.surge_warning_message}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 4. Interactive Forecast Visualization (History + Forecast + Uncertainty)
    # -------------------------------------------------------------------------
    st.subheader(f"📈 Patient Arrival Trajectory: Historical Observations & {horizon_hours}-Hour Forecast")
    st.caption("Solid line displays actual observed demonstration records (past 48 hours). Dashed line and shaded band display forward projection with empirical uncertainty intervals.")

    # Slice past 48 hours of history for readable visualization context
    recent_history = df_history.iloc[-48:].copy()

    col_map = {
        "All Facility": "total_arrivals",
        "Emergency (Red)": "emergency_red",
        "Urgent (Yellow)": "urgent_yellow",
        "Routine (Green)": "routine_green"
    }
    hist_col = col_map.get(target_category, "total_arrivals")

    fig = go.Figure()

    # 1. Historical Observed Trajectory
    fig.add_trace(go.Scatter(
        x=recent_history["timestamp"],
        y=recent_history[hist_col],
        mode="lines+markers",
        name=f"Observed History ({target_category})",
        line=dict(color="#1e3a8a", width=2.5),
        marker=dict(size=4, color="#1e3a8a"),
        hovertemplate="Observed: %{y} patients<br>Time: %{x}<extra></extra>"
    ))

    # Prepare forecast coordinate arrays (bridge last observed point to first forecast point)
    last_hist_x = recent_history["timestamp"].iloc[-1]
    last_hist_y = recent_history[hist_col].iloc[-1]

    fc_x = [last_hist_x] + [p.timestamp for p in forecast.interval_forecasts]
    fc_y = [last_hist_y] + [p.predicted_arrivals for p in forecast.interval_forecasts]

    if is_95_ci:
        fc_lower = [last_hist_y] + [p.lower_ci_95 for p in forecast.interval_forecasts]
        fc_upper = [last_hist_y] + [p.upper_ci_95 for p in forecast.interval_forecasts]
        ci_label = "95% Prediction Interval"
    else:
        fc_lower = [last_hist_y] + [p.lower_ci_80 for p in forecast.interval_forecasts]
        fc_upper = [last_hist_y] + [p.upper_ci_80 for p in forecast.interval_forecasts]
        ci_label = "80% Prediction Interval"

    # 2. Upper Prediction Interval Boundary (transparent line)
    fig.add_trace(go.Scatter(
        x=fc_x,
        y=fc_upper,
        mode="lines",
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip"
    ))

    # 3. Lower Prediction Interval Boundary with shaded fill
    fig.add_trace(go.Scatter(
        x=fc_x,
        y=fc_lower,
        mode="lines",
        line=dict(width=0),
        fill="tonexty",
        fillcolor="rgba(14, 165, 233, 0.22)",
        name=ci_label,
        hoverinfo="skip"
    ))

    # 4. Forecast Point Estimates
    fig.add_trace(go.Scatter(
        x=fc_x,
        y=fc_y,
        mode="lines+markers",
        name=f"Forecast ({horizon_hours}h Horizon)",
        line=dict(color="#0284c7", width=3, dash="dash"),
        marker=dict(size=6, color="#0284c7", symbol="diamond"),
        hovertemplate="Forecast: %{y} patients<br>Time: %{x}<extra></extra>"
    ))

    # Reference Line: Surge Threshold
    fig.add_hline(
        y=FORECAST_THRESHOLDS["HOURLY_SURGE_PEAK"],
        line_dash="dot",
        line_color="#ef4444",
        annotation_text="Surge Threshold (14/hr)",
        annotation_position="top left"
    )

    # Vertical line separating history and forecast
    fig.add_vline(
        x=last_hist_x,
        line_width=1.5,
        line_dash="solid",
        line_color="#64748b",
        annotation_text="Forecast Horizon Start",
        annotation_position="top right"
    )

    fig.update_layout(
        height=420,
        xaxis_title="Timeline (Date & Hour)",
        yaxis_title="Hourly Patient Inflow Count",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=30, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)

    # -------------------------------------------------------------------------
    # 5. Tabbed Detailed Analysis
    # -------------------------------------------------------------------------
    tab_dt, tab_eval, tab_sim, tab_risk = st.tabs([
        "📋 1. Hourly Forecast Table",
        "🎯 2. Holdout Quality Evaluation",
        "⚡ 3. Simulator Integration",
        "🛡️ 4. Proactive Capacity Risk Alerts"
    ])

    # --- TAB 1: Hourly Breakdown Table ---
    with tab_dt:
        st.markdown(f"##### Detailed Interval Estimates ({horizon_hours}-Hour Window)")
        table_rows = []
        for p in forecast.interval_forecasts:
            table_rows.append({
                "Interval": f"Hour +{p.horizon_hour}",
                "Timestamp": p.timestamp,
                "Day": p.day_name,
                "Hour": f"{p.hour_of_day:02d}:00",
                "Predicted Total": p.predicted_arrivals,
                "80% CI Range": f"[{p.lower_ci_80} – {p.upper_ci_80}]",
                "95% CI Range": f"[{p.lower_ci_95} – {p.upper_ci_95}]",
                "Emergency (Red)": p.category_breakdown.get("Emergency", 0),
                "Urgent (Yellow)": p.category_breakdown.get("Urgent", 0),
                "Routine (Green)": p.category_breakdown.get("Routine", 0),
            })
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

    # --- TAB 2: Model Evaluation & Holdout Quality ---
    with tab_eval:
        st.markdown("##### 🎯 Chronological Holdout Validation (Zero Temporal Leakage)")
        st.caption("The model is evaluated strictly on unseen chronological test hours that occurred after the training window. Future records are never mixed into training data.")

        col_ev1, col_ev2, col_ev3, col_ev4 = st.columns(4)
        with col_ev1:
            st.metric("Test Period Hours", f"{eval_metrics.test_observations_count} Hours", delta="7-Day Holdout", delta_color="off")
        with col_ev2:
            st.metric("Seasonal Model MAE", f"{eval_metrics.mae:.2f} patients/hr", delta="Mean Absolute Error", delta_color="off")
        with col_ev3:
            st.metric("Seasonal Model RMSE", f"{eval_metrics.rmse:.2f} patients/hr", delta="Penalizes Surge Spikes", delta_color="off")
        with col_ev4:
            st.metric("Naive Baseline MAE", f"{eval_metrics.baseline_mae:.2f} patients/hr", delta=f"{eval_metrics.error_reduction_pct}% Error Reduction", delta_color="normal")

        st.markdown(f"""
        <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 14px 18px; margin: 14px 0; font-size: 0.92rem; color: #334155; line-height: 1.5;">
            <strong>Plain-Language Accuracy Interpretation:</strong><br>
            {eval_metrics.interpretation}
            <br><br>
            <strong>Count-Data Error Metric Definitions:</strong><br>
            • <strong>MAE (Mean Absolute Error):</strong> The average discrepancy between predicted and actual patient counts per hour. An MAE of {eval_metrics.mae} means the forecast is typically off by approximately {round(eval_metrics.mae)} patients in any given hour.<br>
            • <strong>RMSE (Root Mean Squared Error):</strong> Gives heavier weight to large unexpected surge misses, providing a safety measure against severe underprediction during flash surges.<br>
            • <strong>Benchmark Comparison:</strong> Compares the Seasonal Profile against a 24-Hour Lag Persistence Naive baseline (yesterday's volume at the exact same hour).
        </div>
        """, unsafe_allow_html=True)

        with st.expander("ℹ️ Model Assumptions, Mathematics & Documented Limitations", expanded=False):
            st.markdown(f"""
            **Model Name:** {forecast.model_name}
            
            **Mathematical Formulation:**
            $$\\hat{{y}}_{{d, h}} = \\mu_{{d, h}} \\times \\tau_{{recent}}$$
            where $\\mu_{{d, h}}$ is the empirical conditional expectation for day-of-week $d$ and hour-of-day $h$, and $\\tau_{{recent}}$ is the short-term trend multiplier.
            
            **Assumptions:**
            {"".join(f"- {a}\\n" for a in forecast.assumptions)}
            
            **Known Limitations:**
            {"".join(f"- {l}\\n" for l in forecast.limitations)}
            """)

    # --- TAB 3: Integration with What-If Simulator ---
    with tab_sim:
        st.markdown("##### ⚡ Connect Forecast Directly into What-If Simulator")
        st.caption("Seamlessly transfer forecasted demand into the hospital capacity simulator to evaluate inpatient bed, CT scanner, and triage queue outcomes.")

        sim_params = translate_forecast_to_simulator_parameters(
            forecast=forecast,
            ct_scanner_status="Unavailable" if "Degraded" in str(summary) else "Available",
            discharge_delay=True
        )

        col_s_l, col_s_r = st.columns([2, 1])
        with col_s_l:
            st.markdown(f"""
            <div style="background: #f0f9ff; border: 1px solid #bae6fd; border-radius: 12px; padding: 18px 22px;">
                <h4 style="margin-top: 0; color: #0369a1;">Translated Simulation Parameters:</h4>
                <ul style="color: #334155; font-size: 0.95rem; margin-bottom: 12px;">
                    <li><strong>Simulation Duration:</strong> {sim_params['duration_hours']} Hours (matches forecast horizon)</li>
                    <li><strong>Forecasted Total Patient Arrivals:</strong> {sim_params['forecast_total']} Patients</li>
                    <li><strong>Incremental Surge Above Baseline Buffer:</strong> +{sim_params['additional_arrivals_total']} Patients</li>
                    <li><strong>Forecasted Peak Arrival Window:</strong> {sim_params['forecast_peak_hour']} ({sim_params['forecast_peak_count']} arrivals/hr)</li>
                </ul>
                <p style="font-size: 0.85rem; color: #64748b; margin-bottom: 0;">
                    🔒 <strong>Double-Counting Prevention:</strong> Only the net incremental volume above nominal intake (+{sim_params['additional_arrivals_total']}) is passed as additional surge into the simulator. Baseline demand is accounted for within standard hospital turnover.
                </p>
            </div>
            """, unsafe_allow_html=True)

        with col_s_r:
            st.markdown('<div style="height: 10px;"></div>', unsafe_allow_html=True)
            use_forecast_btn = st.button("🚀 Use Forecast in Simulator", type="secondary", use_container_width=True)
            if use_forecast_btn:
                st.session_state["sim_forecast_applied"] = True
                st.session_state["sim_forecast_surge"] = sim_params["additional_arrivals_total"]
                st.session_state["sim_forecast_duration"] = sim_params["duration_hours"]
                st.session_state["main_nav"] = "⚡ What-If Simulator"
                st.success("✅ Forecast transferred successfully! Navigating to What-If Simulator...")
                st.rerun()

            use_di_btn = st.button("🎯 Send to Decision Intelligence", type="primary", use_container_width=True)
            if use_di_btn:
                st.session_state["di_horizon"] = sim_params["duration_hours"]
                st.session_state["main_nav"] = "🎯 Decision Intelligence"
                st.success("✅ Forecast transferred to Decision Intelligence Engine! Navigating...")
                st.rerun()

        # On-Page Simulator Preview
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("##### 🔍 Forecast-Driven Simulation Impact Preview")
        
        # Run deterministic simulation with forecast inputs
        sim_preview = run_scenario_step_by_step(
            name="Forecasted Operational Stress",
            scenario_type="disrupted",
            description=f"Simulated operational response to {horizon_hours}h forecasted demand.",
            assumptions=[f"+{sim_params['additional_arrivals_total']} forecasted surge arrivals", f"{horizon_hours}h horizon"],
            duration_hours=sim_params["duration_hours"],
            total_beds=280,
            initial_occupied=238,
            initial_waiting=34,
            initial_ct_queue=6,
            additional_arrivals_total=sim_params["additional_arrivals_total"],
            discharge_rate_hourly=2.4,  # conservative clearance
            max_admission_intake_hourly=12,
            ct_capacity_hourly=8.0,
            ct_status_label="Available"
        )

        col_pr1, col_pr2, col_pr3, col_pr4 = st.columns(4)
        with col_pr1:
            st.metric("Projected Inpatient Occupancy", f"{sim_preview.final_occupancy_pct}%", delta=f"{sim_preview.final_occupied}/280 Beds", delta_color="off")
        with col_pr2:
            st.metric("Projected Available Beds", f"{sim_preview.final_available} Beds Open", delta="Diminishing buffer" if sim_preview.final_available < 15 else "Adequate", delta_color="inverse" if sim_preview.final_available < 15 else "normal")
        with col_pr3:
            st.metric("Projected Waiting in Triage", f"{sim_preview.final_waiting} Patients", delta=f"Peak: {sim_preview.peak_waiting} waiting", delta_color="inverse")
        with col_pr4:
            st.metric("Total Discharges Needed", f"{sim_preview.total_discharges} Discharges", delta="To avoid gridlock", delta_color="off")

    # --- TAB 4: Proactive Decision Support Alerts ---
    with tab_risk:
        st.markdown("##### 🛡️ Proactive Early-Warning Capacity Alerts (Phase 3 Integration)")
        st.caption("Identify anticipated capacity deficits before arrivals occur so hospital coordinators can activate flex beds, EVS turnovers, and staff rebalancing in advance.")

        if not capacity_alerts:
            st.success("✅ Anticipated arrival trajectory does not indicate a critical bed deficit over the selected horizon window.")
        else:
            for al in capacity_alerts[:4]:
                badge_sev = '<span style="background: #fee2e2; color: #991b1b; padding: 2px 8px; border-radius: 9999px; font-size: 0.75rem; font-weight: 700;">CRITICAL LEAD TIME</span>' if al.severity == "CRITICAL" else '<span style="background: #fef3c7; color: #92400e; padding: 2px 8px; border-radius: 9999px; font-size: 0.75rem; font-weight: 700;">PREVENTATIVE ACTION</span>'
                st.markdown(f"""
                <div style="background: #ffffff; border: 1px solid #e2e8f0; border-left: 5px solid {'#ef4444' if al.severity == 'CRITICAL' else '#f59e0b'}; border-radius: 12px; padding: 16px 20px; margin-bottom: 12px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <strong style="font-size: 1.05rem; color: #0f172a;">Anticipated Capacity Deficit at {al.timestamp} (Interval +{al.interval_hour}h)</strong>
                        {badge_sev}
                    </div>
                    <div style="margin-top: 6px; font-size: 0.92rem; color: #334155;">
                        <strong>Projected Impact:</strong> Inflow rate of {al.predicted_arrivals} arrivals/hr exceeds ready bed buffer, projecting a net deficit of <span style="color: #ef4444; font-weight: 700;">{al.projected_bed_deficit} beds</span> with {al.projected_waiting_queue} patients waiting in triage.
                    </div>
                    <div style="margin-top: 8px; font-size: 0.9rem; color: #0284c7;">
                        🎯 <strong>Proactive Operational Recommendation:</strong> {al.operational_action}
                    </div>
                    <div style="margin-top: 4px; font-size: 0.8rem; color: #64748b;">
                        ⏱️ <strong>Recommended Action Lead-Time:</strong> {al.lead_time_recommended}
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # Bottom Disclaimer
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.caption("🔒 **MediFlow AI Predictive Analytics Notice:** All arrival forecasts, uncertainty intervals, and proactive alerts are statistical estimates derived from synthetic demonstration data. MediFlow AI does not process real clinical records and does not automate patient treatment, triage, or admission decisions.")
