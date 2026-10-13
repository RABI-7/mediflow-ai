"""
=============================================================================
 MediFlow AI — Operational Reports & Shift Turnover Briefings
=============================================================================
 Dedicated workspace for generating administrative shift turnover summaries,
 census audit logs, and printable operational status reports.
=============================================================================
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from typing import Dict, Any

from bottleneck_engine import evaluate_hidden_capacity


def render_reports(
    summary: Dict[str, Any],
    df_departments: pd.DataFrame,
    df_resources: pd.DataFrame
):
    """
    Renders the dedicated Operational Reports workspace.
    """
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <div style="font-size: 1.5rem; font-weight: 800; color: #0f172a; letter-spacing: -0.02em;">
            Operational Reports & Shift Turnover
        </div>
        <div style="font-size: 0.88rem; color: #475569; margin-top: 2px;">
            Standardized shift handover briefing, departmental census verification, and printable capacity audit reports.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 1. Report Selector & Export Header
    with st.container(border=True):
        col_ctl1, col_ctl2, col_ctl3 = st.columns([2, 1, 1])
        with col_ctl1:
            report_type = st.selectbox(
                "Select Report Type",
                options=[
                    "Shift Turnover Briefing (08:00 - 16:00 Day Shift)",
                    "Hospital Census & Capacity Audit",
                    "Critical Bottleneck & Diagnostic Incident Summary"
                ]
            )
        with col_ctl2:
            st.selectbox("Format", ["Executive Summary View", "Detailed Departmental Log", "CSV Raw Export"])
        with col_ctl3:
            st.markdown('<div style="height: 28px;"></div>', unsafe_allow_html=True)
            st.button("📥 Download PDF Briefing", type="primary", use_container_width=True)

    # 2. Render Shift Turnover Briefing Document
    st.markdown("<br>", unsafe_allow_html=True)
    current_time = datetime.now().strftime("%B %d, %Y • %H:%M EST")

    with st.container(border=True):
        st.markdown(f"""
        <div style="display: flex; justify-content: space-between; border-bottom: 2px solid #0f172a; padding-bottom: 12px; margin-bottom: 16px;">
            <div>
                <h3 style="margin: 0; color: #0f172a;">Metro General Hospital — Clinical Operations Briefing</h3>
                <div style="font-size: 0.88rem; color: #64748b; margin-top: 2px;">Shift: {summary.get('shift_name', 'Day Shift (08:00 - 16:00)')} | Site: Main Hospital Campus</div>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 0.88rem; font-weight: 700; color: #0f172a;">Generated: {current_time}</div>
                <div style="font-size: 0.78rem; color: #0284c7;">Incident Command Lead: Dr. Eleanor Vance</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Section 1: Executive Census
        tot_beds = summary.get("total_beds", 280)
        occ_beds = summary.get("occupied_beds", 238)
        occ_pct = summary.get("occupancy_rate_pct", 85.0)
        wait_pts = summary.get("patients_waiting", 34)
        ready_beds = summary.get("ready_available_beds", 6)
        staff_on_duty = summary.get("staff_on_duty", 64)

        hc = evaluate_hidden_capacity(summary, None)

        st.markdown("#### 1. Facility Census & Bed Status")
        col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
        with col_m1:
            st.metric("Total Licensed Beds", f"{tot_beds}")
        with col_m2:
            st.metric("Active Inpatients", f"{occ_beds} ({occ_pct}%)")
        with col_m3:
            st.metric("Genuinely Ready", f"{ready_beds} beds")
        with col_m4:
            st.metric("Triage Boarding Queue", f"{wait_pts} patients")
        with col_m5:
            st.metric("Recoverable Capacity", f"{hc.recoverable_beds_total} beds")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 2. Departmental Unit Census")
        dept_summary = df_departments[["Department", "Capacity", "Occupied", "Available", "Occupancy %", "Staff", "Status"]].copy()
        st.dataframe(dept_summary, use_container_width=True, hide_index=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 3. Active Diagnostic Backlogs & Equipment Outages")
        res_summary = df_resources[["Resource / Unit", "Availability Status", "Queue Length", "Est. Wait Time", "Operational Health"]].copy()
        st.dataframe(res_summary, use_container_width=True, hide_index=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 4. Charge Nurse & Bed Placement Shift Directives")
        st.markdown("""
        1. **Emergency Triage Boarding Mitigation:** 34 patients currently boarding in ER triage. Prioritize admission placement into ready General Medicine beds.
        2. **EVS Cleaning Surge Sweep:** 12 dirty beds awaiting terminal cleaning. Coordinate 90-minute strike cleaning protocol to unlock 10 beds before evening surge.
        3. **Diagnostic Prioritization:** CT Scanner 2 degraded (95m wait). Divert routine outpatients to regional partner clinic to reserve CT Scanner 1 for trauma intake.
        4. **Staffing Float:** Reassign 2 cross-trained Med-Surg RNs from Pediatrics and Cardiology to General Medicine. Donor floor minimum safety floors are preserved.
        """)

        st.markdown("""
        <div style="border-top: 1px solid #e2e8f0; padding-top: 10px; margin-top: 20px; font-size: 0.78rem; color: #64748b; text-align: center;">
            MediFlow AI • Clinical Operations Intelligence Platform • Standard Shift Handover Log • Synthetic Demo Telemetry
        </div>
        """, unsafe_allow_html=True)
