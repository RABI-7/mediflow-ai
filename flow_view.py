"""
=============================================================================
 MediFlow AI — Patient Flow & Arrival Intelligence
=============================================================================
 Dedicated workspace for monitoring emergency department patient inflow,
 empirical arrival demand forecasting, and proactive early-warning capacity alerts.
=============================================================================
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, Any

from forecast_engine import (
    generate_historical_arrival_dataset,
    generate_patient_arrival_forecast,
    evaluate_forecast_model_performance,
    evaluate_forecast_driven_capacity_risk,
    translate_forecast_to_simulator_parameters
)


def render_patient_flow(
    summary: Dict[str, Any],
    df_departments: pd.DataFrame,
    df_resources: pd.DataFrame
):
    """
    Renders the dedicated Patient Flow & Arrival Intelligence page.
    """
    # -------------------------------------------------------------------------
    # 1. Page Header
    # -------------------------------------------------------------------------
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <div style="font-size: 1.5rem; font-weight: 800; color: #0f172a; letter-spacing: -0.02em;">
            Patient Flow & Arrival Intelligence
        </div>
        <div style="font-size: 0.88rem; color: #475569; margin-top: 2px;">
            Emergency patient intake velocity, empirical arrival forecasting with uncertainty intervals, and proactive capacity alerts.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Cached historical dataset
    @st.cache_data
    def get_flow_history():
        return generate_historical_arrival_dataset(start_date="2026-08-18", num_days=56, seed=42)

    df_history = get_flow_history()

    # -------------------------------------------------------------------------
    # 2. Interactive Forecast Controls
    # -------------------------------------------------------------------------
    with st.container(border=True):
        col_c1, col_c2, col_c3, col_c4 = st.columns(4)
        with col_c1:
            horizon_hours = st.selectbox("Forecast Projection Window", [4, 8, 12, 24], index=2, format_func=lambda h: f"{h} Hours", key="pf_hor")
        with col_c2:
            triage_cat = st.selectbox("Acuity Classification", ["All Facility", "Emergency (Red)", "Urgent (Yellow)", "Routine (Green)"], index=0, key="pf_cat")
        with col_c3:
            ci_choice = st.selectbox("Prediction Interval", ["95% Prediction Interval", "80% Prediction Interval"], index=0, key="pf_ci")
        with col_c4:
            st.markdown('<div style="height: 28px;"></div>', unsafe_allow_html=True)
            refresh_btn = st.button("🔄 Update Projections", type="primary", use_container_width=True)

    # Generate active forecast
    forecast = generate_patient_arrival_forecast(
        df_history=df_history,
        horizon_hours=int(horizon_hours),
        target_category=str(triage_cat),
        recent_trend_factor=1.05
    )

    # 7-day chronological holdout evaluation
    @st.cache_data
    def get_cached_metrics():
        return evaluate_forecast_model_performance(df_history, test_days=7)

    eval_metrics = get_cached_metrics()

    # Proactive capacity alerts
    alerts = evaluate_forecast_driven_capacity_risk(forecast, summary, df_departments)

    # -------------------------------------------------------------------------
    # 3. Forecast Metrics Row
    # -------------------------------------------------------------------------
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric(
            label="Projected Arrivals",
            value=f"{forecast.total_predicted_arrivals} Patients",
            delta=f"Over {forecast.horizon_hours} Hours",
            delta_color="off"
        )
    with m2:
        st.metric(
            label="Average Hourly Rate",
            value=f"{forecast.mean_arrivals_per_hour:.1f} / hr",
            delta="Inflow Velocity",
            delta_color="off"
        )
    with m3:
        st.metric(
            label="Anticipated Peak Hour",
            value=f"{forecast.peak_predicted_hour}",
            delta=f"{forecast.peak_predicted_count} Arrivals",
            delta_color="inverse"
        )
    with m4:
        st.metric(
            label="Forecast Error (MAE)",
            value=f"{eval_metrics.mae:.2f} pts/hr",
            delta=f"{eval_metrics.error_reduction_pct:.1f}% vs Lag-24 Naive",
            delta_color="normal"
        )
    with m5:
        st.metric(
            label="Incremental Surge",
            value=f"+{forecast.incremental_surge_above_baseline} Above Base",
            delta="Simulation Surge Delta",
            delta_color="inverse" if forecast.incremental_surge_above_baseline > 15 else "off"
        )

    # -------------------------------------------------------------------------
    # 4. Inflow Projections Chart with Prediction Interval Ribbon
    # -------------------------------------------------------------------------
    with st.container(border=True):
        st.markdown(f"""
        <div style="font-weight: 700; color: #0f172a; font-size: 0.98rem; margin-bottom: 2px;">
            Patient Inflow Trajectory & Empirical Uncertainty Band ({ci_choice})
        </div>
        <div style="font-size: 0.82rem; color: #64748b; margin-bottom: 12px;">
            Recent historical arrival observations merged seamlessly with forward statistical projections
        </div>
        """, unsafe_allow_html=True)

        # Plotly chart
        fig = go.Figure()

        # Last 24 hours of history
        if df_history is None or df_history.empty:
            st.info("Historical arrival observations are currently unavailable for display.")
        else:
            recent_hist = df_history.iloc[-24:].copy()

            # Safe extraction of timestamp column
            if "timestamp" in recent_hist.columns:
                hist_timestamps = recent_hist["timestamp"].tolist()
            elif "datetime" in recent_hist.columns:
                hist_timestamps = recent_hist["datetime"].dt.strftime("%Y-%m-%d %H:00").tolist()
            elif "ds" in recent_hist.columns:
                hist_timestamps = recent_hist["ds"].tolist()
            else:
                hist_timestamps = [f"T-{24 - i}h" for i in range(len(recent_hist))]

            if triage_cat == "Emergency (Red)" and "emergency_red" in recent_hist.columns:
                hist_vals = recent_hist["emergency_red"].tolist()
            elif triage_cat == "Urgent (Yellow)" and "urgent_yellow" in recent_hist.columns:
                hist_vals = recent_hist["urgent_yellow"].tolist()
            elif triage_cat == "Routine (Green)" and "routine_green" in recent_hist.columns:
                hist_vals = recent_hist["routine_green"].tolist()
            elif "total_arrivals" in recent_hist.columns:
                hist_vals = recent_hist["total_arrivals"].tolist()
            else:
                hist_vals = []

            if hist_timestamps and hist_vals:
                fig.add_trace(go.Scatter(
                    x=hist_timestamps,
                    y=hist_vals,
                    mode="lines+markers",
                    name="Recent Historical Arrivals",
                    line=dict(color="#64748b", width=2),
                    marker=dict(size=4)
                ))

        # Forward Forecast
        fc_times = [p.timestamp for p in forecast.interval_forecasts]
        fc_preds = [p.predicted_arrivals for p in forecast.interval_forecasts]
        is_95 = ("95%" in ci_choice)
        fc_lower = [p.lower_ci_95 if is_95 else p.lower_ci_80 for p in forecast.interval_forecasts]
        fc_upper = [p.upper_ci_95 if is_95 else p.upper_ci_80 for p in forecast.interval_forecasts]

        # Upper bound
        fig.add_trace(go.Scatter(
            x=fc_times,
            y=fc_upper,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip"
        ))

        # Lower bound with fill
        fig.add_trace(go.Scatter(
            x=fc_times,
            y=fc_lower,
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(2, 132, 199, 0.15)",
            name=f"{'95%' if is_95 else '80%'} Empirical Prediction Band",
            hoverinfo="skip"
        ))

        # Point Forecast
        fig.add_trace(go.Scatter(
            x=fc_times,
            y=fc_preds,
            mode="lines+markers",
            name="Predicted Arrival Inflow",
            line=dict(color="#0284c7", width=3),
            marker=dict(size=6, color="#0284c7")
        ))

        fig.update_layout(
            height=340,
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            xaxis=dict(title="Timeline", showgrid=True, gridcolor="#f1f5f9"),
            yaxis=dict(title="Arrivals / Hour", showgrid=True, gridcolor="#f1f5f9"),
            template="plotly_white",
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)

    # -------------------------------------------------------------------------
    # 5. Proactive Capacity Early-Warning Alerts
    # -------------------------------------------------------------------------
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Proactive Capacity Deficit Alerts")
    st.caption("Early-warning notifications triggered when projected arrival rate exceeds ready bed intake and discharge velocity.")

    if not alerts:
        st.success("✅ Projected inflow trajectory remains within ready buffer over the selected horizon window.")
    else:
        for al in alerts[:4]:
            border_c = "#ef4444" if al.severity == "CRITICAL" else "#f59e0b"
            badge_bg = "#fee2e2" if al.severity == "CRITICAL" else "#fef3c7"
            badge_fg = "#991b1b" if al.severity == "CRITICAL" else "#92400e"

            st.markdown(f"""
            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-left: 4px solid {border_c}; border-radius: 8px; padding: 12px 16px; margin-bottom: 10px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <strong style="color: #0f172a; font-size: 0.95rem;">Anticipated Capacity Deficit at {al.timestamp} (Interval +{al.interval_hour}h)</strong>
                    <span style="background: {badge_bg}; color: {badge_fg}; font-size: 0.75rem; font-weight: 700; padding: 2px 8px; border-radius: 9999px;">
                        {al.severity} LEAD TIME
                    </span>
                </div>
                <div style="font-size: 0.86rem; color: #334155; margin-top: 4px;">
                    <strong>Projected Inflow:</strong> {al.predicted_arrivals} arrivals/hr exceeds discharge clearance velocity, projecting a net deficit of <span style="color: #ef4444; font-weight: 700;">{al.projected_bed_deficit} beds</span> with {al.projected_waiting_queue} patients queued.
                </div>
                <div style="font-size: 0.83rem; color: #0284c7; margin-top: 4px;">
                    🎯 <strong>Proactive Operational Directive:</strong> {al.operational_action}
                </div>
                <div style="font-size: 0.78rem; color: #64748b; margin-top: 3px;">
                    ⏱️ <strong>Recommended Action Lead-Time:</strong> {al.lead_time_recommended}
                </div>
            </div>
            """, unsafe_allow_html=True)
