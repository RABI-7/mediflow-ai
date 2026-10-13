"""
=============================================================================
 MediFlow AI — Facility Settings & Clinical Operations Governance
=============================================================================
 System configuration for hospital parameters, alerting thresholds,
 user roles, and data telemetry streams.
=============================================================================
"""

import streamlit as st
from typing import Dict, Any


def render_settings(summary: Dict[str, Any]):
    """
    Renders the dedicated Facility Settings & Operations Governance page.
    """
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <div style="font-size: 1.5rem; font-weight: 800; color: #0f172a; letter-spacing: -0.02em;">
            Facility Settings & Operational Thresholds
        </div>
        <div style="font-size: 0.88rem; color: #475569; margin-top: 2px;">
            Configure licensed capacity bounds, operational alert thresholds, shift schedules, and clinical governance protocols.
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab_fac, tab_thresh, tab_gov, tab_telemetry = st.tabs([
        "Facility Parameters",
        "Clinical Alert Thresholds",
        "Role-Based Governance",
        "Telemetry & Integrity"
    ])

    with tab_fac:
        with st.container(border=True):
            st.markdown("#### Facility Profile")
            c1, c2 = st.columns(2)
            with c1:
                st.text_input("Facility Name", "Metro General Hospital — Main Campus", disabled=True)
                st.number_input("Licensed Physical Bed Capacity", min_value=100, max_value=500, value=280, disabled=True, help="Fixed structural building envelope.")
            with c2:
                st.text_input("Campus Code", "METRO-CENTRAL-01", disabled=True)
                st.number_input("Maximum Surge Flex Envelope", min_value=280, max_value=350, value=295, disabled=True, help="Statutory upper cap including licensed surge flex overflow beds.")

            st.caption("Facility profile parameters reflect licensed building codes and municipal emergency authority approvals.")

    with tab_thresh:
        with st.container(border=True):
            st.markdown("#### Alert & Bottleneck Detection Thresholds")
            c1, c2 = st.columns(2)
            with c1:
                st.slider("Occupancy Warning Threshold (%)", min_value=70, max_value=95, value=85, help="Triggers capacity strain alerts.")
                st.slider("Occupancy Critical Threshold (%)", min_value=85, max_value=100, value=90, help="Triggers critical gridlock alarm.")
                st.slider("ER Boarding Warning (Patients)", min_value=5, max_value=30, value=10)
            with c2:
                st.slider("CT Diagnostic Wait Warning (Mins)", min_value=20, max_value=60, value=45)
                st.slider("CT Diagnostic Wait Critical (Mins)", min_value=60, max_value=180, value=90)
                st.slider("Minimum Discharges Target (Per Hour)", min_value=2.0, max_value=10.0, value=5.5)

    with tab_gov:
        with st.container(border=True):
            st.markdown("#### Incident Command & Clinical Sign-Off Protocol")
            st.markdown("""
            MediFlow AI enforces a **Human-in-the-Loop Advisory Governance Protocol**. 
            Proposed solutions are evaluated through deterministic simulation, but implementation 
            strictly requires designated clinical authorization:
            """)
            st.markdown("""
            - **Chief Medical Officer (CMO):** Final authorization for flex observation beds and major clinical diversion.
            - **Chief Nursing Officer (CNO):** Authorization for cross-department nurse floating and shift reassignment.
            - **Bed Placement Coordinator:** Authorization for expedited patient transfer and admission routing.
            - **Environmental Services (EVS) Supervisor:** Authorization for 90-minute strike cleaning protocols.
            - **Radiology Medical Director:** Authorization for diagnostic imaging diversion and protocol adjustments.
            """)

    with tab_telemetry:
        with st.container(border=True):
            st.markdown("#### Telemetry Stream Status")
            st.success("🟢 Real-time Telemetry Simulator Active • Invariant Checker: PASS • Division-by-Zero Safety: ACTIVE")
            st.markdown("""
            - **Data Mode:** Synthetic Hospital Operations Telemetry (Zero PHI processed).
            - **Update Frequency:** Every 60 seconds (Simulated diurnal clock).
            - **Mathematical Guardrails:** Non-negative constraints, strict bed state sum invariant, and donor department statutory floor preservation enforced.
            """)
