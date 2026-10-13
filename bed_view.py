"""
=============================================================================
 MediFlow AI — Bed Management & Capacity Recovery
=============================================================================
 Dedicated workspace for real-time bed tracking, partition breakdown,
 and operational hidden capacity recovery.
=============================================================================
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, Any

from bottleneck_engine import evaluate_hidden_capacity, HiddenCapacityResult


def render_bed_management(
    summary: Dict[str, Any],
    df_departments: pd.DataFrame,
    df_resources: pd.DataFrame
):
    """
    Renders the dedicated Bed Management & Capacity Recovery page.
    """
    # -------------------------------------------------------------------------
    # 1. Page Header & Operational Summary
    # -------------------------------------------------------------------------
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <div style="font-size: 1.5rem; font-weight: 800; color: #0f172a; letter-spacing: -0.02em;">
            Bed Management & Capacity Recovery
        </div>
        <div style="font-size: 0.88rem; color: #475569; margin-top: 2px;">
            Real-time inpatient bed state tracking, turnover blocker identification, and operational capacity recovery.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Compute bed state partition metrics
    hc_result: HiddenCapacityResult = evaluate_hidden_capacity(summary, None)
    bd = hc_result.breakdown

    tot_beds = bd.total_physical_capacity
    occ_beds = bd.occupied_beds
    disch_beds = bd.awaiting_discharge_coordination
    clean_beds = bd.awaiting_cleaning_evs
    verif_beds = bd.awaiting_readiness_verification
    ready_beds = bd.ready_available_beds
    maint_beds = bd.unavailable_maintenance
    blocked_beds = hc_result.recoverable_beds_total

    # -------------------------------------------------------------------------
    # 2. Key Operational Bed Metrics Row
    # -------------------------------------------------------------------------
    c1, c2, c3, c4, c5, c6 = st.columns(6)

    with c1:
        st.metric(
            label="Licensed Beds",
            value=f"{tot_beds}",
            delta="Physical Envelope",
            delta_color="off"
        )
    with c2:
        occ_pct = round((occ_beds / tot_beds) * 100, 1)
        st.metric(
            label="Inpatient Occupied",
            value=f"{occ_beds}",
            delta=f"{occ_pct}% Occupancy",
            delta_color="inverse" if occ_pct >= 85 else "off"
        )
    with c3:
        st.metric(
            label="Genuinely Ready",
            value=f"{ready_beds}",
            delta="Intake Cleared",
            delta_color="normal" if ready_beds > 10 else "inverse"
        )
    with c4:
        st.metric(
            label="Awaiting Cleaning (EVS)",
            value=f"{clean_beds}",
            delta="Dirty Beds",
            delta_color="inverse" if clean_beds > 8 else "off"
        )
    with c5:
        st.metric(
            label="Pending Discharge Handoff",
            value=f"{disch_beds}",
            delta="Departure Delayed",
            delta_color="inverse" if disch_beds > 10 else "off"
        )
    with c6:
        st.metric(
            label="Recoverable Buffer",
            value=f"{blocked_beds}",
            delta="Turnover Unlock",
            delta_color="normal"
        )

    # -------------------------------------------------------------------------
    # 3. Bed State Partition Visualization
    # -------------------------------------------------------------------------
    with st.container(border=True):
        st.markdown("""
        <div style="font-weight: 700; color: #0f172a; font-size: 0.98rem; margin-bottom: 4px;">
            Inpatient Bed Allocation Breakdown
        </div>
        <div style="font-size: 0.84rem; color: #64748b; margin-bottom: 12px;">
            Strict physical invariant: Occupied + Discharge Handoff + EVS Cleaning + Verification + Ready + Maintenance == Total Licensed Beds (280)
        </div>
        """, unsafe_allow_html=True)

        fig_bar = go.Figure()
        categories = ["Licensed Capacity (280)"]

        fig_bar.add_trace(go.Bar(name="Occupied Beds", y=categories, x=[occ_beds], orientation="h", marker_color="#0284c7"))
        fig_bar.add_trace(go.Bar(name="Awaiting Discharge", y=categories, x=[disch_beds], orientation="h", marker_color="#f59e0b"))
        fig_bar.add_trace(go.Bar(name="Awaiting Cleaning (EVS)", y=categories, x=[clean_beds], orientation="h", marker_color="#ef4444"))
        fig_bar.add_trace(go.Bar(name="Awaiting Verification", y=categories, x=[verif_beds], orientation="h", marker_color="#8b5cf6"))
        fig_bar.add_trace(go.Bar(name="Ready Available", y=categories, x=[ready_beds], orientation="h", marker_color="#10b981"))
        fig_bar.add_trace(go.Bar(name="Maintenance", y=categories, x=[maint_beds], orientation="h", marker_color="#94a3b8"))

        fig_bar.update_layout(
            barmode="stack",
            height=120,
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="left", x=0),
            xaxis=dict(showgrid=True, gridcolor="#f1f5f9", title="Total Beds"),
            yaxis=dict(showticklabels=False),
            template="plotly_white"
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # -------------------------------------------------------------------------
    # 4. Department-Level Bed Utilization Table
    # -------------------------------------------------------------------------
    col_dept_l, col_dept_r = st.columns([1.6, 1.4])

    with col_dept_l:
        with st.container(border=True):
            st.markdown("""
            <div style="font-weight: 700; color: #0f172a; font-size: 0.98rem; margin-bottom: 2px;">
                Departmental Bed Census & Availability
            </div>
            <div style="font-size: 0.82rem; color: #64748b; margin-bottom: 12px;">
                Current inpatient distribution across specialty and acute clinical units
            </div>
            """, unsafe_allow_html=True)

            dept_display = df_departments[["Department", "Capacity", "Occupied", "Available", "Occupancy %", "Status"]].copy()
            dept_display.columns = ["Department", "Capacity", "Occupied", "Open", "Occupancy %", "Flow Status"]
            st.dataframe(dept_display, use_container_width=True, hide_index=True)

    with col_dept_r:
        with st.container(border=True):
            st.markdown("""
            <div style="font-weight: 700; color: #0f172a; font-size: 0.98rem; margin-bottom: 2px;">
                Operational Capacity Recovery Actions
            </div>
            <div style="font-size: 0.82rem; color: #64748b; margin-bottom: 12px;">
                Specific task-level interventions to transition blocked beds into intake-ready status
            </div>
            """, unsafe_allow_html=True)

            for act in hc_result.ordered_actions:
                badge_bg = "#fee2e2" if act.priority == 1 else "#fef3c7" if act.priority == 2 else "#eff6ff"
                badge_fg = "#991b1b" if act.priority == 1 else "#92400e" if act.priority == 2 else "#1e40af"
                st.markdown(f"""
                <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px 14px; margin-bottom: 8px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <strong style="color: #0f172a; font-size: 0.92rem;">Priority {act.priority}: {act.title}</strong>
                        <span style="background: {badge_bg}; color: {badge_fg}; font-weight: 700; font-size: 0.75rem; padding: 2px 8px; border-radius: 9999px;">
                            +{act.recoverable_beds_estimated} Ready Beds
                        </span>
                    </div>
                    <div style="font-size: 0.84rem; color: #334155; margin-top: 4px;">
                        <strong>Blocker:</strong> {act.primary_blocker}
                    </div>
                    <div style="font-size: 0.82rem; color: #0284c7; margin-top: 3px;">
                        <strong>Action Protocol:</strong> {act.action_protocol}
                    </div>
                    <div style="font-size: 0.78rem; color: #64748b; margin-top: 4px; display: flex; justify-content: space-between;">
                        <span>⏱️ Est. Unlock: <strong>{act.estimated_unlock_time}</strong></span>
                        <span>Resource: <strong>{act.required_resources}</strong></span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 5. Readiness Safeguard Notice
    # -------------------------------------------------------------------------
    st.markdown("""
    <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-left: 4px solid #0284c7; padding: 10px 14px; border-radius: 6px; margin-top: 10px; font-size: 0.82rem; color: #334155;">
        🔒 <strong>Readiness Safeguard:</strong> MediFlow AI enforces strict clinical bed partition invariants. Beds categorized under cleaning, verification, or discharge coordination are never counted as intake-ready until all environmental sanitization and nursing handoffs are verified.
    </div>
    """, unsafe_allow_html=True)
