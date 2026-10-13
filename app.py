"""
=============================================================================
 MediFlow AI — Hospital Resource & Flow Optimization
=============================================================================
 Hackathon Track: Overcrowded Hospitals, Limited Resources
 Phase 1: Project Setup and Basic Dashboard

 Technologies:
 - Python
 - Streamlit (UI framework)
 - Pandas (Data processing)
 - Plotly (Interactive visualizations)

 All data generated in this app is 100% synthetic for demonstration purposes.
=============================================================================
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from simulator import run_hospital_what_if_simulation, ScenarioResult
from bottleneck_engine import (
    detect_hospital_bottlenecks,
    generate_recommendations,
    simulate_specific_intervention,
    THRESHOLDS,
    BottleneckFinding,
    OperationalRecommendation,
    InterventionComparison
)
from bottleneck_view import render_bottleneck_analysis
from flow_view import render_patient_flow
from decision_view import render_decision_intelligence
from bed_view import render_bed_management
from staff_resources_view import render_staff_resources
from reports_view import render_reports
from settings_view import render_settings
from simulator_solutions import (
    ProposedSolution,
    SolutionImpactMetrics,
    RankedSolutionEvaluation,
    generate_scenario_tailored_solutions,
    calculate_impact_metrics,
    simulate_and_rank_solutions
)

# -----------------------------------------------------------------------------
# 1. Page Configuration & Theme
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="MediFlow AI — Intelligent Hospital Resource Optimization",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished, modern enterprise healthcare product UI
st.markdown("""
<style>
    /* Global Typography & Font Smoothing */
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }
    
    /* Global Brand Header */
    .brand-header-box {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 18px 24px;
        margin-bottom: 20px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
    }
    .brand-title-text {
        font-size: 1.85rem;
        font-weight: 800;
        color: #0f172a;
        letter-spacing: -0.025em;
        margin: 0;
        line-height: 1.2;
    }
    .brand-subtitle-text {
        font-size: 0.98rem;
        font-weight: 600;
        color: #0284c7;
        margin-top: 3px;
        letter-spacing: 0.01em;
    }
    
    /* System & Facility Badges */
    .facility-badge {
        display: inline-block;
        background-color: #f1f5f9;
        color: #334155;
        font-size: 0.8rem;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 9999px;
        border: 1px solid #e2e8f0;
        margin-right: 6px;
    }
    .live-telemetry-badge {
        display: inline-block;
        background-color: #f0fdf4;
        color: #166534;
        font-size: 0.8rem;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 9999px;
        border: 1px solid #bbf7d0;
    }
    .badge-demo {
        display: inline-block;
        background-color: #e0f2fe;
        color: #0369a1;
        font-size: 0.78rem;
        font-weight: 600;
        padding: 3px 9px;
        border-radius: 9999px;
        border: 1px solid #bae6fd;
    }
    
    /* Metric Cards */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        padding: 16px 20px;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
        transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px -2px rgba(0, 0, 0, 0.06);
        border-color: #cbd5e1;
    }
    
    /* Simulator Feature Spotlight Card on Dashboard */
    .sim-spotlight-card {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        border: 1px solid #bae6fd;
        border-left: 5px solid #0284c7;
        padding: 18px 22px;
        border-radius: 12px;
        margin-bottom: 22px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .sim-spotlight-badge {
        display: inline-block;
        background-color: #0284c7;
        color: #ffffff;
        font-size: 0.72rem;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 9999px;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
        text-transform: uppercase;
    }
    .sim-spotlight-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #0c4a6e;
        margin: 0;
    }
    .sim-spotlight-desc {
        font-size: 0.9rem;
        color: #334155;
        margin-top: 3px;
        margin-bottom: 0;
    }
    
    /* Sidebar promo box */
    .sidebar-promo-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 12px 14px;
        margin-top: 14px;
    }
    
    /* Simulator Bottleneck Box */
    .bottleneck-box {
        background-color: #fff1f2;
        border-left: 5px solid #e11d48;
        padding: 16px 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        color: #9f1239;
    }
    .bottleneck-box-stable {
        background-color: #f0fdf4;
        border-left: 5px solid #16a34a;
        padding: 16px 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        color: #166534;
    }
    
    /* Scenario comparison headers */
    .scenario-header {
        padding: 10px 14px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 0.95rem;
        margin-bottom: 12px;
        text-align: center;
    }
    .scenario-baseline {
        background-color: #eff6ff;
        color: #1e40af;
        border: 1px solid #bfdbfe;
    }
    .scenario-disrupted {
        background-color: #fff1f2;
        color: #9f1239;
        border: 1px solid #fecdd3;
    }
    .scenario-response {
        background-color: #f0fdf4;
        color: #166534;
        border: 1px solid #bbf7d0;
    }
    
    /* Product roadmap placeholder card */
    .future-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        padding: 36px 30px;
        border-radius: 16px;
        text-align: center;
        margin-top: 16px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
    }
    
    /* Solution Recommendation Cards & Badges */
    .sol-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px 18px;
        margin-bottom: 14px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02);
    }
    .sol-feasible-border {
        border-left: 5px solid #10b981;
    }
    .sol-infeasible-border {
        border-left: 5px solid #ef4444;
        background: #fffbfa;
    }
    .badge-feasible {
        display: inline-block;
        background: #ecfdf5;
        color: #065f46;
        font-size: 0.72rem;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 9999px;
        border: 1px solid #a7f3d0;
    }
    .badge-infeasible {
        display: inline-block;
        background: #fef2f2;
        color: #991b1b;
        font-size: 0.72rem;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 9999px;
        border: 1px solid #fecaca;
    }
    .badge-category {
        display: inline-block;
        background: #f1f5f9;
        color: #334155;
        font-size: 0.72rem;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 9999px;
        border: 1px solid #e2e8f0;
        margin-right: 6px;
    }
    .badge-evidence {
        display: inline-block;
        background: #eff6ff;
        color: #1e40af;
        font-size: 0.72rem;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 9999px;
        border: 1px solid #bfdbfe;
    }
    .best-option-banner {
        background: linear-gradient(135deg, #f0fdf4 0%, #ffffff 100%);
        border: 2px solid #86efac;
        border-radius: 12px;
        padding: 18px 22px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px -1px rgba(16, 185, 129, 0.08);
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. Synthetic Data Generators (Pure Python & Pandas)
# -----------------------------------------------------------------------------
@st.cache_data
def load_synthetic_hospital_data():
    """
    Generates realistic, consistent synthetic hospital data for Phase 1.
    No real patient or clinical data is used.
    """
    # 1. Summary Metrics
    summary = {
        "total_beds": 280,
        "occupied_beds": 238,
        "available_beds": 42,
        "occupancy_rate_pct": 85.0,
        "patients_waiting": 34,
        "staff_on_duty": 64,
        "shift_name": "Day Shift (08:00 - 16:00)",
        "shift_start": "08:00",
        "shift_end": "16:00",
        "awaiting_discharge_beds": 14,
        "awaiting_cleaning_beds": 12,
        "awaiting_verification_beds": 6,
        "ready_available_beds": 6,
        "unavailable_maintenance_beds": 4
    }

    # 2. Synthetic Patient Arrivals by Hour (24-hour cycle)
    hours = [f"{h:02d}:00" for h in range(24)]
    # Realistic surge pattern: peaks in early afternoon and evening
    emergency_arrivals = [3, 2, 2, 1, 1, 2, 4, 6, 8, 11, 14, 15, 13, 12, 14, 16, 15, 12, 9, 8, 6, 5, 4, 3]
    urgent_arrivals =    [2, 1, 1, 0, 1, 2, 3, 5, 7, 9, 10, 12, 11, 10, 11, 12, 11, 9, 7, 5, 4, 3, 2, 1]
    routine_arrivals =   [0, 0, 0, 0, 0, 1, 2, 4, 8, 12, 15, 14, 10, 9, 11, 10, 7, 4, 2, 1, 1, 0, 0, 0]

    df_arrivals = pd.DataFrame({
        "Hour": hours,
        "Emergency (Red)": emergency_arrivals,
        "Urgent (Yellow)": urgent_arrivals,
        "Routine (Green)": routine_arrivals
    })
    df_arrivals["Total Arrivals"] = (
        df_arrivals["Emergency (Red)"] +
        df_arrivals["Urgent (Yellow)"] +
        df_arrivals["Routine (Green)"]
    )

    # 3. Departmental Bed Capacity and Utilization
    department_data = [
        {"Department": "Emergency Dept (ER)", "Capacity": 45, "Occupied": 42, "Staff": 14, "Status": "Critical Load", "MinStaffFloor": 10, "AcuityWeight": 1.5, "WorkloadUnits": 63.0, "Qualifications": "Emergency BLS, ACLS, Trauma Triage", "EligibleCrossUnits": "None (High Inflow)"},
        {"Department": "Intensive Care (ICU)", "Capacity": 30, "Occupied": 27, "Staff": 12, "Status": "Near Capacity", "MinStaffFloor": 10, "AcuityWeight": 2.0, "WorkloadUnits": 54.0, "Qualifications": "Critical Care, CCRN, ACLS", "EligibleCrossUnits": "None (Critical Locked)"},
        {"Department": "General Medicine",   "Capacity": 95, "Occupied": 86, "Staff": 16, "Status": "Normal Flow", "MinStaffFloor": 10, "AcuityWeight": 1.0, "WorkloadUnits": 86.0, "Qualifications": "Med-Surg RN, BLS", "EligibleCrossUnits": "None (Recipient)"},
        {"Department": "Surgery & Post-Op",   "Capacity": 55, "Occupied": 51, "Staff": 11, "Status": "High Load", "MinStaffFloor": 8, "AcuityWeight": 1.1, "WorkloadUnits": 56.1, "Qualifications": "PACU / Surgical RN, BLS", "EligibleCrossUnits": "None"},
        {"Department": "Pediatrics",          "Capacity": 35, "Occupied": 18, "Staff": 6,  "Status": "Normal Flow", "MinStaffFloor": 3, "AcuityWeight": 1.0, "WorkloadUnits": 18.0, "Qualifications": "Pediatric RN, PALS, Med-Surg Eligible", "EligibleCrossUnits": "General Medicine, Surgery & Post-Op"},
        {"Department": "Cardiology Ward",     "Capacity": 20, "Occupied": 14, "Staff": 5,  "Status": "Normal Flow", "MinStaffFloor": 2, "AcuityWeight": 1.0, "WorkloadUnits": 14.0, "Qualifications": "Telemetry RN, ACLS, Med-Surg Eligible", "EligibleCrossUnits": "General Medicine, Emergency Dept (ER)"}
    ]
    df_departments = pd.DataFrame(department_data)
    df_departments["Available"] = df_departments["Capacity"] - df_departments["Occupied"]
    df_departments["Occupancy %"] = ((df_departments["Occupied"] / df_departments["Capacity"]) * 100).round(1)

    # 4. Critical Hospital Resources & Diagnostic Equipment
    resources_data = [
        {
            "Resource / Unit": "CT Scanner 1 (Main Radiology)",
            "Category": "Diagnostics",
            "Availability Status": "Operational",
            "Queue Length": "6 Patients",
            "Est. Wait Time": "45 mins",
            "Operational Health": "Normal"
        },
        {
            "Resource / Unit": "CT Scanner 2 (Trauma Wing)",
            "Category": "Diagnostics",
            "Availability Status": "Maintenance / Degraded",
            "Queue Length": "11 Patients",
            "Est. Wait Time": "95 mins",
            "Operational Health": "High Bottleneck Risk"
        },
        {
            "Resource / Unit": "MRI Scanner",
            "Category": "Diagnostics",
            "Availability Status": "Operational",
            "Queue Length": "4 Patients",
            "Est. Wait Time": "60 mins",
            "Operational Health": "Normal"
        },
        {
            "Resource / Unit": "Operating Rooms (OR 1-4)",
            "Category": "Surgical",
            "Availability Status": "3 of 4 Active",
            "Queue Length": "3 Elective Cases",
            "Est. Wait Time": "Scheduled",
            "Operational Health": "Optimal"
        },
        {
            "Resource / Unit": "Patient Transport Staff",
            "Category": "Support Staff",
            "Availability Status": "6 Active Transporters",
            "Queue Length": "8 Transfer Requests",
            "Est. Wait Time": "25 mins",
            "Operational Health": "Minor Delay"
        }
    ]
    df_resources = pd.DataFrame(resources_data)

    return summary, df_arrivals, df_departments, df_resources


# -----------------------------------------------------------------------------
# 3. Healthcare SaaS Navigation Shell
# -----------------------------------------------------------------------------
def render_top_navbar(page_title: str, breadcrumb_item: str = "Operations", urgent_alerts_count: int = 3):
    """
    Renders a consistent, modern healthcare SaaS top navigation bar.
    Displays current breadcrumb, facility selector, live telemetry status, alert badge, and user profile.
    """
    st.markdown(f"""
    <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 12px 18px; margin-bottom: 20px; box-shadow: 0 1px 2px rgba(0,0,0,0.02);">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <span style="font-size: 1.1rem; font-weight: 800; color: #0f172a; letter-spacing: -0.02em;">MediFlow AI</span>
                <span style="color: #cbd5e1; font-size: 0.85rem;">/</span>
                <span style="font-size: 0.8rem; font-weight: 600; color: #64748b; text-transform: uppercase; letter-spacing: 0.04em;">{breadcrumb_item}</span>
                <span style="color: #cbd5e1; font-size: 0.85rem;">/</span>
                <span style="font-size: 0.88rem; font-weight: 700; color: #0284c7;">{page_title}</span>
            </div>
            <div style="display: flex; align-items: center; gap: 8px; flex-wrap: wrap;">
                <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; padding: 4px 10px; font-size: 0.78rem; font-weight: 600; color: #334155;">
                    📍 Metro General Hospital — Main Campus
                </div>
                <div style="background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 6px; padding: 4px 10px; font-size: 0.78rem; font-weight: 600; color: #166534;">
                    ● Live Telemetry (Updated 2m ago)
                </div>
                <div style="background: #fff1f2; border: 1px solid #fecdd3; border-radius: 6px; padding: 4px 10px; font-size: 0.78rem; font-weight: 600; color: #9f1239;">
                    🔔 {urgent_alerts_count} Active Alerts
                </div>
                <div style="background: #f1f5f9; border: 1px solid #e2e8f0; border-radius: 6px; padding: 4px 10px; font-size: 0.78rem; font-weight: 600; color: #0f172a;">
                    👤 Dr. E. Vance (Operations Lead)
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar():
    """
    Renders the left sidebar containing navigation and facility metadata.
    """
    with st.sidebar:
        st.markdown("""
        <div style="padding: 10px 4px 16px 4px;">
            <div style="font-size: 1.35rem; font-weight: 800; color: #0f172a; letter-spacing: -0.02em;">
                🏥 MediFlow AI
            </div>
            <div style="font-size: 0.8rem; font-weight: 600; color: #0284c7; margin-top: 1px;">
                Hospital Operations Platform
            </div>
        </div>
        """, unsafe_allow_html=True)

        nav_options = [
            "📊 Overview",
            "🛏️ Bed Management",
            "📈 Patient Flow",
            "⚠️ Bottleneck Intelligence",
            "🎯 Decision Intelligence",
            "⚡ What-If Simulator",
            "👥 Staff & Resources",
            "📄 Reports",
            "⚙️ Settings"
        ]

        legacy_map = {
            "📊 Hospital Dashboard": "📊 Overview",
            "⚠️ Bottleneck Analysis": "⚠️ Bottleneck Intelligence",
            "🔮 Patient Forecasting": "📈 Patient Flow"
        }
        if "nav_goto" in st.session_state and st.session_state["nav_goto"] in legacy_map:
            st.session_state["nav_goto"] = legacy_map[st.session_state["nav_goto"]]

        if "nav_goto" in st.session_state and st.session_state["nav_goto"] in nav_options:
            st.session_state["main_nav"] = st.session_state.pop("nav_goto")
        elif "main_nav" not in st.session_state or st.session_state["main_nav"] not in nav_options:
            st.session_state["main_nav"] = nav_options[0]

        page = st.radio(
            "Navigation",
            options=nav_options,
            key="main_nav",
            label_visibility="collapsed"
        )

        st.markdown("---")
        st.markdown("""
        <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px; font-size: 0.8rem; color: #475569;">
            <div style="font-weight: 700; color: #0f172a; margin-bottom: 6px;">Facility Telemetry</div>
            <div style="margin-bottom: 3px;"><strong>Site:</strong> Metro General Main</div>
            <div style="margin-bottom: 3px;"><strong>Shift:</strong> Day (08:00 - 16:00)</div>
            <div><strong>Status:</strong> <span style="color: #16a34a; font-weight: 600;">● Online</span></div>
        </div>
        """, unsafe_allow_html=True)

    return page


# -----------------------------------------------------------------------------
# 4. View: Hospital Operations Overview (Executive Workspace)
# -----------------------------------------------------------------------------
def render_overview(summary, df_arrivals, df_departments, df_resources):
    """
    Renders the executive Hospital Operations Overview page.
    Displays:
    - 6 core clinical & operational metrics
    - 2 operational charts (Arrival velocity by triage category, Departmental census & capacity)
    - 2 actionable sections (Urgent operational alerts, Detected bottlenecks & actionable directives)
    """
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <div style="font-size: 1.5rem; font-weight: 800; color: #0f172a; letter-spacing: -0.02em;">
            Hospital Operations Overview
        </div>
        <div style="font-size: 0.88rem; color: #475569; margin-top: 2px;">
            Real-time situational awareness across licensed bed census, emergency influx velocity, and active operational constraints.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ---------------------------------------------------------
    # 1. Metric Summary Cards (Compact 6-metric Row)
    # ---------------------------------------------------------
    c1, c2, c3, c4, c5, c6 = st.columns(6)

    with c1:
        st.metric(
            label="Total Licensed Beds",
            value=f"{summary['total_beds']}",
            delta="Physical Envelope",
            delta_color="off"
        )
    with c2:
        occ_pct = summary["occupancy_rate_pct"]
        st.metric(
            label="Occupied Beds",
            value=f"{summary['occupied_beds']}",
            delta=f"{occ_pct}% Occupancy",
            delta_color="inverse" if occ_pct >= 85 else "off"
        )
    with c3:
        ready_beds = summary.get("ready_available_beds", 6)
        st.metric(
            label="Ready & Available",
            value=f"{ready_beds}",
            delta="Intake Cleared",
            delta_color="normal" if ready_beds > 10 else "inverse"
        )
    with c4:
        st.metric(
            label="Patients Waiting",
            value=f"{summary['patients_waiting']}",
            delta="Triage / Boarding",
            delta_color="inverse"
        )
    with c5:
        blocked_delay_beds = (
            summary.get("awaiting_discharge_beds", 14) +
            summary.get("awaiting_cleaning_beds", 12) +
            summary.get("awaiting_verification_beds", 6)
        )
        st.metric(
            label="Blocked by Delays",
            value=f"{blocked_delay_beds}",
            delta="Recoverable Turnover",
            delta_color="inverse"
        )
    with c6:
        st.metric(
            label="Staff on Duty",
            value=f"{summary['staff_on_duty']}",
            delta="2 High Strain Units",
            delta_color="inverse"
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # 2. Operational Visualizations Row
    # ---------------------------------------------------------
    col_chart_left, col_chart_right = st.columns([1, 1])

    with col_chart_left:
        st.markdown("##### 📈 Patient Arrivals Over 24 Hours")
        st.caption("Hourly influx velocity categorized by clinical triage severity")

        fig_arrivals = go.Figure()
        fig_arrivals.add_trace(go.Scatter(
            x=df_arrivals["Hour"],
            y=df_arrivals["Emergency (Red)"],
            mode="lines+markers",
            name="Emergency (Critical)",
            line=dict(color="#ef4444", width=2.5),
            stackgroup="one"
        ))
        fig_arrivals.add_trace(go.Scatter(
            x=df_arrivals["Hour"],
            y=df_arrivals["Urgent (Yellow)"],
            mode="lines+markers",
            name="Urgent",
            line=dict(color="#f59e0b", width=2),
            stackgroup="one"
        ))
        fig_arrivals.add_trace(go.Scatter(
            x=df_arrivals["Hour"],
            y=df_arrivals["Routine (Green)"],
            mode="lines+markers",
            name="Routine",
            line=dict(color="#10b981", width=2),
            stackgroup="one"
        ))

        fig_arrivals.update_layout(
            height=340,
            margin=dict(l=20, r=20, t=25, b=20),
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis_title="Time of Day",
            yaxis_title="Arrivals / Hour",
            template="plotly_white"
        )
        st.plotly_chart(fig_arrivals, use_container_width=True)

    with col_chart_right:
        st.markdown("##### 🛏️ Department Census & Available Capacity")
        st.caption("Occupied inpatient beds versus remaining available capacity")

        fig_dept = go.Figure()
        fig_dept.add_trace(go.Bar(
            y=df_departments["Department"],
            x=df_departments["Occupied"],
            name="Occupied Beds",
            orientation="h",
            marker=dict(color="#2563eb")
        ))
        fig_dept.add_trace(go.Bar(
            y=df_departments["Department"],
            x=df_departments["Available"],
            name="Available Beds",
            orientation="h",
            marker=dict(color="#94a3b8")
        ))

        fig_dept.update_layout(
            barmode="stack",
            height=340,
            margin=dict(l=20, r=20, t=25, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis_title="Number of Beds",
            yaxis_title="",
            template="plotly_white"
        )
        st.plotly_chart(fig_dept, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # 3. Actionable Operational Alerts & Directives Row
    # ---------------------------------------------------------
    col_alert_left, col_action_right = st.columns([1, 1])

    with col_alert_left:
        st.markdown("##### 🚨 Operational Alerts (Ranked by Urgency)")
        st.caption("Active operational bottlenecks prioritized by clinical impact")

        alerts = [
            {
                "level": "CRITICAL",
                "badge_color": "#ef4444",
                "bg_color": "#fef2f2",
                "border_color": "#fecaca",
                "title": "Emergency Department Boarding Strain",
                "detail": "34 patients currently waiting in triage/boarding queue. ER census at 93.3% (42/45 beds occupied).",
                "action": "Expedite inpatient admission transfers to General Med."
            },
            {
                "level": "HIGH",
                "badge_color": "#f59e0b",
                "bg_color": "#fffbeb",
                "border_color": "#fde68a",
                "title": "CT Scanner 2 Diagnostic Degradation",
                "detail": "Trauma scanner operating in degraded maintenance mode. Queue at 11 patients (est. wait 95 min).",
                "action": "Re-route eligible non-trauma diagnostic scans to Main Radiology Scanner 1."
            },
            {
                "level": "HIGH",
                "badge_color": "#f59e0b",
                "bg_color": "#fffbeb",
                "border_color": "#fde68a",
                "title": "Bed Turnover Clearance Delay",
                "detail": "32 beds physically unoccupied but blocked from intake (14 discharge hold, 12 dirty EVS, 6 verification).",
                "action": "Deploy rapid-turnover EVS response team to recover ready capacity."
            },
            {
                "level": "MODERATE",
                "badge_color": "#3b82f6",
                "bg_color": "#eff6ff",
                "border_color": "#bfdbfe",
                "title": "Nurse Workload Acuity Imbalance",
                "detail": "General Medicine and Surgery exceeding 5.0:1 ratio. Pediatrics & Cardiology have surplus coverage.",
                "action": "Evaluate qualified float-eligible RN cross-allocation to General Medicine."
            }
        ]

        for alt in alerts:
            st.markdown(f"""
            <div style="background: {alt['bg_color']}; border: 1px solid {alt['border_color']}; border-left: 4px solid {alt['badge_color']}; border-radius: 8px; padding: 10px 14px; margin-bottom: 10px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 3px;">
                    <span style="font-weight: 700; font-size: 0.88rem; color: #0f172a;">{alt['title']}</span>
                    <span style="font-size: 0.72rem; font-weight: 700; color: {alt['badge_color']}; background: #ffffff; border: 1px solid {alt['border_color']}; border-radius: 4px; padding: 1px 6px;">{alt['level']}</span>
                </div>
                <div style="font-size: 0.82rem; color: #475569; margin-bottom: 4px;">{alt['detail']}</div>
                <div style="font-size: 0.8rem; font-weight: 600; color: #0369a1;">➔ Directive: {alt['action']}</div>
            </div>
            """, unsafe_allow_html=True)

    with col_action_right:
        st.markdown("##### ⚡ Flow Bottlenecks & Quick Directives")
        st.caption("Direct operational actions to unblock clinical patient flow")

        with st.container(border=True):
            st.markdown("<strong style='color: #0f172a;'>1. Recover 32 Blocked Inpatient Beds</strong>", unsafe_allow_html=True)
            st.markdown("<span style='font-size: 0.84rem; color: #475569;'>Discharge handoffs (14 beds) and EVS sanitization (12 beds) represent the fastest potential capacity recovery.</span>", unsafe_allow_html=True)
            if st.button("🛏️ Open Bed Management & Capacity Recovery", use_container_width=True, key="ov_btn_bed"):
                st.session_state["nav_goto"] = "🛏️ Bed Management"
                st.rerun()

        with st.container(border=True):
            st.markdown("<strong style='color: #0f172a;'>2. Stress-Test Upcoming Influx Surges</strong>", unsafe_allow_html=True)
            st.markdown("<span style='font-size: 0.84rem; color: #475569;'>Simulate sudden ED patient surges, CT outages, and discharge delays side-by-side with tested mitigations.</span>", unsafe_allow_html=True)
            if st.button("⚡ Open What-If Capacity Simulator", use_container_width=True, key="ov_btn_sim"):
                st.session_state["nav_goto"] = "⚡ What-If Simulator"
                st.rerun()

        with st.container(border=True):
            st.markdown("<strong style='color: #0f172a;'>3. Review Multi-Objective Decision Package</strong>", unsafe_allow_html=True)
            st.markdown("<span style='font-size: 0.84rem; color: #475569;'>Inspect ranked candidate interventions, objective scores, risk trade-offs, and record clinical approval.</span>", unsafe_allow_html=True)
            if st.button("🎯 Launch Decision Intelligence Workspace", type="primary", use_container_width=True, key="ov_btn_di"):
                st.session_state["nav_goto"] = "🎯 Decision Intelligence"
                st.rerun()


# -----------------------------------------------------------------------------
# 5. View: Hospital What-If Simulator
# -----------------------------------------------------------------------------
def render_what_if_simulator(summary=None, df_departments=None, df_resources=None):
    """
    Renders the interactive Hospital What-If Simulator with solution recommendations and comparisons.
    Simulates acute operational disruptions, proposes tailored interventions, and enforces feasibility.
    """
    if summary is None or df_departments is None:
        summary, _, df_departments, df_resources = load_synthetic_hospital_data()

    # Clean Page Header Row
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <div style="font-size: 1.5rem; font-weight: 800; color: #0f172a; letter-spacing: -0.02em;">
            Hospital What-If Capacity Simulator
        </div>
        <div style="font-size: 0.88rem; color: #475569; margin-top: 2px;">
            Deterministic stress-testing: evaluate sudden patient arrival surges, diagnostic outages, and bed clearance delays under identical baselines.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Check if forecast parameters were passed from Phase 4
    forecast_applied = st.session_state.get("sim_forecast_applied", False)
    default_surge = int(st.session_state.get("sim_forecast_surge", 30))
    default_surge = max(0, min(60, default_surge))
    default_duration = int(st.session_state.get("sim_forecast_duration", 12))
    duration_options = [4, 8, 12, 24]
    duration_idx = duration_options.index(default_duration) if default_duration in duration_options else 2

    if forecast_applied:
        st.markdown(f"""
        <div style="background-color: #f0f9ff; border: 1px solid #bae6fd; border-left: 5px solid #0284c7; padding: 14px 18px; border-radius: 10px; margin-bottom: 16px;">
            <div style="font-weight: 700; color: #0369a1; font-size: 0.98rem; margin-bottom: 2px;">
                🔮 Arrival Forecast Parameters Applied
            </div>
            <div style="font-size: 0.88rem; color: #334155;">
                Pre-populated controls with <strong>+{default_surge} forecasted surge arrivals</strong> across a <strong>{default_duration}-hour window</strong> based on the Intelligent Arrival Forecasting module.
                You can review or adjust any scenario parameters below before executing the stress-test.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 1. Interactive Control Panel
    # -------------------------------------------------------------------------
    st.subheader("⚙️ Configure Disruption Scenario")
    with st.container(border=True):
        col_ctrl1, col_ctrl2, col_ctrl3, col_ctrl4 = st.columns(4)

        with col_ctrl1:
            additional_arrivals = st.slider(
                "Additional Patient Arrivals",
                min_value=0,
                max_value=60,
                value=default_surge,
                step=5,
                help="Simulate an unscheduled surge of incoming emergency/urgent patients (0 to 60 patients)."
            )

        with col_ctrl2:
            ct_status_choice = st.selectbox(
                "CT Scanner Status",
                options=[
                    "Unavailable (1 Scanner Down - 50% Outage)",
                    "Available (Both Units Operational)"
                ],
                index=0,
                help="Simulate diagnostic equipment failure in the trauma/radiology wing."
            )

        with col_ctrl3:
            discharge_delay_choice = st.selectbox(
                "Inpatient Discharge Delay",
                options=[
                    "On (Bed Clearances & Handoffs Delayed)",
                    "Off (Standard Discharge Flow)"
                ],
                index=0,
                help="Simulate hospital-wide delays in patient discharge summaries, transport, or pharmacy clearances."
            )

        with col_ctrl4:
            duration_choice = st.selectbox(
                "Simulation Duration",
                options=duration_options,
                index=duration_idx,
                format_func=lambda h: f"{h} Hours",
                help="Time window over which the simulation model evaluates resource strain."
            )

        col_btn_left, col_btn_right = st.columns([1, 3])
        with col_btn_left:
            run_btn = st.button("🚀 Run Simulation", type="primary", use_container_width=True)
        with col_btn_right:
            st.caption("💡 Adjust the sliders and dropdowns above, then click **Run Simulation** (or explore results updated below).")

    # Determine standardized parameter flags
    ct_scanner_status = "Unavailable" if "Unavailable" in ct_status_choice else "Available"
    discharge_delay = True if "On" in discharge_delay_choice else False
    duration_hours = int(duration_choice)

    # -------------------------------------------------------------------------
    # 2. Run Simulation Model (via simulator.py)
    # -------------------------------------------------------------------------
    baseline_res, disrupted_res, response_res, meta = run_hospital_what_if_simulation(
        additional_arrivals=additional_arrivals,
        ct_scanner_status=ct_scanner_status,
        discharge_delay=discharge_delay,
        duration_hours=duration_hours
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 3. Main Bottleneck Summary Banner
    # -------------------------------------------------------------------------
    if disrupted_res.final_waiting > 0 or "Saturation" in meta["primary_disruption_bottleneck"] or "Diagnostic" in meta["primary_disruption_bottleneck"]:
        box_class = "bottleneck-box"
        icon = "🚨"
    else:
        box_class = "bottleneck-box-stable"
        icon = "✅"

    waiting_diff = meta["waiting_difference"]
    mitigated_count = meta["waiting_mitigated"]

    st.markdown(f"""
    <div class="{box_class}">
        <h3 style="margin-top: 0; margin-bottom: 8px;">{icon} Main Bottleneck Identified: {meta['primary_disruption_bottleneck']}</h3>
        <p style="margin-bottom: 8px; font-size: 1rem;">
            <strong>Disruption Impact:</strong> Under the active disruption scenario (+{additional_arrivals} surge arrivals, CT {ct_scanner_status}, Discharge Delays {'Active' if discharge_delay else 'Inactive'} over {duration_hours}h),
            inpatient bed occupancy reaches <strong>{disrupted_res.final_occupancy_pct}%</strong> with <strong>{disrupted_res.final_waiting} patients left waiting</strong> for an open bed.
            Diagnostic backlog accumulates to <strong>{disrupted_res.final_ct_queue} patients waiting for CT</strong> with an estimated wait time of <strong>{disrupted_res.est_ct_wait_mins} minutes</strong>.
        </p>
        <p style="margin-bottom: 0; font-size: 0.95rem;">
            <strong>Operational Mitigation:</strong> The illustrative proposed response strategy absorbs the surge, mitigating <strong>{mitigated_count} waiting patients</strong>
            and stabilizing diagnostic wait times to <strong>{response_res.est_ct_wait_mins} minutes</strong>.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 4. Scenario A & B Reference Overview
    # -------------------------------------------------------------------------
    st.subheader("📊 Operational Scenarios: Baseline vs. Active Disruption")
    st.caption("Compare baseline hospital performance against the unmitigated stress-test disruption before evaluating solutions.")

    col_scen_a, col_scen_b = st.columns(2)

    # --- Scenario A: Baseline ---
    with col_scen_a:
        st.markdown('<div class="scenario-header scenario-baseline">🟢 Scenario A: Baseline (Normal Operations)</div>', unsafe_allow_html=True)
        st.metric(
            label="Occupied Beds",
            value=f"{baseline_res.final_occupied} / {baseline_res.total_beds}",
            delta=f"{baseline_res.final_occupancy_pct}% Occupancy",
            delta_color="off"
        )
        st.metric(
            label="Available Beds",
            value=f"{baseline_res.final_available} beds open",
            delta="Normal buffer",
            delta_color="off"
        )
        st.metric(
            label="Patients Waiting for Bed (Unmet Demand)",
            value=f"{baseline_res.final_waiting}",
            delta=f"Peak: {baseline_res.peak_waiting} in queue",
            delta_color="off"
        )
        st.metric(
            label="CT Scanner Queue & Wait",
            value=f"{baseline_res.final_ct_queue} patients",
            delta=f"{baseline_res.est_ct_wait_mins} mins wait",
            delta_color="off"
        )
        st.caption(f"**Total Discharges:** {baseline_res.total_discharges} | **Projected Bed Demand:** {baseline_res.projected_bed_demand} beds")

    # --- Scenario B: Disrupted ---
    with col_scen_b:
        st.markdown('<div class="scenario-header scenario-disrupted">🔴 Scenario B: Disrupted Operations (Active Stress-Test)</div>', unsafe_allow_html=True)
        st.metric(
            label="Occupied Beds",
            value=f"{disrupted_res.final_occupied} / {disrupted_res.total_beds}",
            delta=f"{disrupted_res.final_occupancy_pct}% Occupancy",
            delta_color="inverse"
        )
        delta_avail = disrupted_res.final_available - baseline_res.final_available
        st.metric(
            label="Available Beds",
            value=f"{disrupted_res.final_available} beds open",
            delta=f"{delta_avail} vs Baseline",
            delta_color="normal" if delta_avail >= 0 else "inverse"
        )
        delta_waiting = disrupted_res.final_waiting - baseline_res.final_waiting
        st.metric(
            label="Patients Waiting for Bed (Unmet Demand)",
            value=f"{disrupted_res.final_waiting}",
            delta=f"+{delta_waiting} surge backlog" if delta_waiting > 0 else "Stable",
            delta_color="inverse" if delta_waiting > 0 else "off"
        )
        delta_ct_wait = disrupted_res.est_ct_wait_mins - baseline_res.est_ct_wait_mins
        st.metric(
            label="CT Scanner Queue & Wait",
            value=f"{disrupted_res.final_ct_queue} patients",
            delta=f"+{delta_ct_wait} mins delay" if delta_ct_wait > 0 else "Normal",
            delta_color="inverse" if delta_ct_wait > 0 else "off"
        )
        st.caption(f"**Total Discharges:** {disrupted_res.total_discharges} | **Projected Bed Demand:** {disrupted_res.projected_bed_demand} beds")

    st.markdown("<br>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 5. Recommended Solutions Section (Tailored to Active Calculated Bottlenecks)
    # -------------------------------------------------------------------------
    st.subheader("💡 Recommended Solutions & Actionable Interventions")
    st.caption("Context-aware operational strategies generated dynamically from active simulation bottlenecks. Check one or more solutions to simulate and compare against the disrupted baseline.")

    # Generate tailored solution candidates based on active calculated bottlenecks
    available_solutions = generate_scenario_tailored_solutions(
        disrupted_res=disrupted_res,
        summary=summary,
        df_departments=df_departments,
        additional_arrivals=additional_arrivals,
        ct_scanner_status=ct_scanner_status,
        discharge_delay=discharge_delay,
        duration_hours=duration_hours
    )

    feasible_solutions = [s for s in available_solutions if s.feasibility.is_feasible]
    infeasible_solutions = [s for s in available_solutions if not s.feasibility.is_feasible]

    # Render Feasible Solution Selection Cards
    selected_solution_ids = []
    
    col_cards = st.columns(min(3, max(1, len(feasible_solutions))))
    for idx, sol in enumerate(feasible_solutions):
        target_col = col_cards[idx % len(col_cards)]
        with target_col:
            # Default first two solutions checked for immediate interactive comparison
            is_default = (idx < 2)
            chk_val = st.checkbox(
                f"**Test: {sol.title}**",
                key=f"chk_sol_{sol.id}",
                value=is_default,
                help=f"Select to simulate and compare {sol.title} under identical initial conditions."
            )
            if chk_val:
                selected_solution_ids.append(sol.id)

            st.markdown(f"""
            <div class="sol-card sol-feasible-border">
                <div style="margin-bottom: 6px;">
                    <span class="badge-category">{sol.category}</span>
                    <span class="badge-feasible">🟢 FEASIBLE</span>
                    <span class="badge-evidence">{sol.evidence_quality}</span>
                </div>
                <div style="font-weight: 700; color: #0f172a; font-size: 0.95rem; margin-bottom: 4px;">
                    🎯 Addresses: {sol.target_bottleneck}
                </div>
                <div style="font-size: 0.85rem; color: #334155; margin-bottom: 8px;">
                    {sol.description}
                </div>
                <div style="font-size: 0.8rem; color: #64748b; margin-bottom: 4px;">
                    <strong>Why it helps:</strong> {sol.operational_rationale}
                </div>
            </div>
            """, unsafe_allow_html=True)

            with st.expander(f"🔍 Details & Evidence: {sol.title}", expanded=False):
                st.markdown(f"**Target Bottleneck:** {sol.target_bottleneck}")
                st.markdown(f"**Supporting Evidence:** {sol.evidence_summary}")
                st.markdown(f"**Action Plan:**\n{sol.action_plan}")
                st.markdown("**Model Assumptions:**")
                for assump in sol.assumptions:
                    st.markdown(f"- {assump}")
                st.markdown(f"**Evidence Quality Label:** `{sol.evidence_quality}`")
                st.markdown(f"**Missing Data / Model Limitations:** {sol.limitations_or_missing_data}")
                st.markdown(f"**Required Human Approval:** `{sol.required_human_approval}`")
                st.markdown(f"**Clinical Priority Safeguard:** {sol.clinical_priority_safeguard}")

    # Operational Safety Tests Expander (Infeasible candidates to demonstrate safety enforcement)
    with st.expander("🛡️ Operational Safety Tests (Verify Rejection of Infeasible Actions)", expanded=False):
        st.caption("Demonstrates the safety engine: Attempting actions that strip nurse staffing below statutory floors or exceed licensed physical capacity will be rejected with explicit rationale instead of being simulated.")
        col_inf1, col_inf2 = st.columns(2)
        for idx, sol in enumerate(infeasible_solutions):
            target_inf_col = col_inf1 if idx % 2 == 0 else col_inf2
            with target_inf_col:
                chk_inf = st.checkbox(
                    f"**Test Safety Rejection: {sol.title}**",
                    key=f"chk_inf_{sol.id}",
                    value=False,
                    help="Attempting to simulate this intervention will trigger an operational safety rejection."
                )
                if chk_inf:
                    selected_solution_ids.append(sol.id)

                reasons_str = "<br>• ".join(sol.feasibility.rejection_reasons)
                st.markdown(f"""
                <div class="sol-card sol-infeasible-border">
                    <div style="margin-bottom: 6px;">
                        <span class="badge-category">{sol.category}</span>
                        <span class="badge-infeasible">🔴 INFEASIBLE</span>
                    </div>
                    <div style="font-weight: 700; color: #991b1b; font-size: 0.92rem; margin-bottom: 4px;">
                        {sol.title}
                    </div>
                    <div style="font-size: 0.83rem; color: #7f1d1d; margin-bottom: 6px;">
                        <strong>Rejection Reasons:</strong><br>• {reasons_str}
                    </div>
                    <div style="font-size: 0.78rem; color: #450a0a;">
                        <strong>Regulatory Rule:</strong> {sol.clinical_priority_safeguard}
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 6. Simulate, Enforce Feasibility & Rank Selected Solutions
    # -------------------------------------------------------------------------
    st.markdown("<br>", unsafe_allow_html=True)
    col_cmp_btn1, col_cmp_btn2 = st.columns([1, 3])
    with col_cmp_btn1:
        run_cmp_btn = st.button("🚀 Run Scenario Comparison", type="primary", use_container_width=True)
    with col_cmp_btn2:
        st.caption(f"Evaluating **{len(selected_solution_ids)} selected scenario(s)** under identical initial conditions (280 licensed beds, 238 occupied, 34 waiting, 6 CT queue) across {duration_hours} hours.")

    ranked_evaluations, best_evaluation = simulate_and_rank_solutions(
        selected_solution_ids=selected_solution_ids,
        available_solutions=available_solutions,
        duration_hours=duration_hours,
        additional_arrivals=additional_arrivals,
        ct_scanner_status=ct_scanner_status,
        discharge_delay=discharge_delay,
        disrupted_res=disrupted_res
    )

    # Render Infeasibility Warnings for any rejected actions
    for ev in ranked_evaluations:
        if not ev.is_feasible:
            st.markdown(f"""
            <div style="background-color: #fef2f2; border: 1px solid #fecaca; border-left: 5px solid #dc2626; padding: 14px 18px; border-radius: 8px; margin-bottom: 14px;">
                <strong style="color: #991b1b; font-size: 0.95rem;">🚨 Operational Safeguard Enforced: Solution Rejected</strong><br>
                <span style="font-size: 0.88rem; color: #450a0a;">
                    <strong>{ev.solution.title}</strong> cannot be simulated. <strong>Reason:</strong> {ev.rejection_explanation}<br>
                    <em>Minimum staffing regulations and physical licensed capacity limits strictly prevent executing this intervention.</em>
                </span>
            </div>
            """, unsafe_allow_html=True)

    # Render Best Feasible Option Banner (Requirement 6)
    if best_evaluation:
        imp = best_evaluation.impact
        st.markdown(f"""
        <div class="best-option-banner">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; margin-bottom: 6px;">
                <div style="font-weight: 800; font-size: 1.15rem; color: #065f46;">
                    🏆 Recommended Best Feasible Option: {best_evaluation.solution.title}
                </div>
                <div>
                    <span class="badge-feasible">RANK #1 FEASIBLE</span>
                    <span style="background: #dcfce7; color: #166534; font-size: 0.75rem; font-weight: 700; padding: 2px 8px; border-radius: 9999px; border: 1px solid #86efac;">Score: {best_evaluation.objective_score}</span>
                </div>
            </div>
            <div style="font-size: 0.92rem; color: #1e293b; margin-bottom: 8px;">
                <strong>Selection Rationale:</strong> {best_evaluation.selection_rationale}
            </div>
            <div style="display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 10px; font-size: 0.88rem; color: #0f172a;">
                <span>• <strong>Waiting Queue Relief:</strong> <span style="color: #16a34a; font-weight: 700;">{imp.delta_waiting} ({imp.pct_waiting:.1f}%)</span></span>
                <span>• <strong>Available Beds Restored:</strong> <span style="color: #16a34a; font-weight: 700;">+{imp.delta_available} ({imp.pct_available:.1f}%)</span></span>
                <span>• <strong>CT Wait Time:</strong> <span style="color: #16a34a; font-weight: 700;">{imp.delta_ct_wait} min ({imp.pct_ct_wait:.1f}%)</span></span>
                <span>• <strong>Total Discharges:</strong> <span style="color: #16a34a; font-weight: 700;">+{imp.delta_discharges}</span></span>
            </div>
            <div style="font-size: 0.82rem; color: #047857; background: #ffffff; padding: 8px 12px; border-radius: 6px; border: 1px solid #bbf7d0;">
                🔒 <strong>Advisory Decision Support Notice:</strong> This ranking is calculated using operational flow objectives (queue minimization and bed availability) subject to synthetic model constraints. Authorized clinical and administrative leadership review is required. No patient transfers, discharges, or treatment alterations are automated.
            </div>
        </div>
        """, unsafe_allow_html=True)
    elif not any(e.is_feasible for e in ranked_evaluations):
        st.info("ℹ️ No feasible intervention is currently selected. Select one or more feasible operational interventions above to test their simulated impact.")

    # -------------------------------------------------------------------------
    # 7. Comprehensive Scenario Comparison Table (Requirement 2 & 3)
    # -------------------------------------------------------------------------
    feasible_evals = [e for e in ranked_evaluations if e.is_feasible and e.sim_result is not None]

    st.subheader("📋 Scenario Impact Comparison Table")
    st.caption("Side-by-side comparison of baseline, disrupted, and each simulated intervention showing absolute values, absolute changes (Δ), and percentage changes (%) vs Disrupted.")

    # Build comparison table dictionary
    table_rows = [
        "Occupied Beds",
        "Bed Occupancy Rate (%)",
        "Available Beds",
        "Patients Waiting for Bed (Unmet Demand)",
        "CT Scanner Queue (Patients)",
        "Estimated CT Wait Time (Minutes)",
        "Total Patient Discharges",
        "Operational Feasibility Status"
    ]

    comp_dict: Dict[str, List[Any]] = {
        "Metric / Indicator": table_rows,
        "Baseline (Normal)": [
            f"{baseline_res.final_occupied} / {baseline_res.total_beds}",
            f"{baseline_res.final_occupancy_pct}%",
            f"{baseline_res.final_available}",
            f"{baseline_res.final_waiting}",
            f"{baseline_res.final_ct_queue}",
            f"{baseline_res.est_ct_wait_mins} min",
            f"{baseline_res.total_discharges}",
            "Feasible (Standard)"
        ],
        "Disrupted (Active)": [
            f"{disrupted_res.final_occupied} / {disrupted_res.total_beds}",
            f"{disrupted_res.final_occupancy_pct}%",
            f"{disrupted_res.final_available}",
            f"{disrupted_res.final_waiting}",
            f"{disrupted_res.final_ct_queue}",
            f"{disrupted_res.est_ct_wait_mins} min",
            f"{disrupted_res.total_discharges}",
            "Disrupted"
        ]
    }

    for ev in feasible_evals:
        sim = ev.sim_result
        imp = ev.impact
        short_title = ev.solution.title.split(":")[0] if ":" in ev.solution.title else ev.solution.title
        
        # Format metric values with Δ and % change vs Disrupted
        col_name = f"{short_title}"
        col_vals = [
            f"{sim.final_occupied} / {sim.total_beds} ({'+' if imp.delta_occupied >= 0 else ''}{imp.delta_occupied}, {imp.pct_occupied:+.1f}%)",
            f"{sim.final_occupancy_pct}%",
            f"{sim.final_available} ({'+' if imp.delta_available >= 0 else ''}{imp.delta_available}, {imp.pct_available:+.1f}%)",
            f"{sim.final_waiting} ({imp.delta_waiting:+d}, {imp.pct_waiting:+.1f}%)",
            f"{sim.final_ct_queue} ({imp.delta_ct_queue:+d}, {imp.pct_ct_queue:+.1f}%)",
            f"{sim.est_ct_wait_mins} min ({imp.delta_ct_wait:+d} min, {imp.pct_ct_wait:+.1f}%)",
            f"{sim.total_discharges} ({'+' if imp.delta_discharges >= 0 else ''}{imp.delta_discharges}, {imp.pct_discharges:+.1f}%)",
            f"FEASIBLE (Rank #{ev.rank})"
        ]
        comp_dict[col_name] = col_vals

    df_comp = pd.DataFrame(comp_dict)
    st.dataframe(df_comp, use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 8. Interactive Visualizations (Across All Tested Scenarios)
    # -------------------------------------------------------------------------
    st.subheader("📈 Visual Comparison Across Tested Scenarios")

    tab_vis1, tab_vis2, tab_vis3 = st.tabs([
        "🛏️ Bed Capacity & Unmet Demand",
        "⏳ Patient Waiting Queue Over Time",
        "🔬 Diagnostic & CT Scanner Workload"
    ])

    with tab_vis1:
        st.caption("Comparison of Occupied Beds, Available Beds, and Unmet Demand (Patients Waiting) across all tested scenarios")
        fig_beds = go.Figure()
        
        scen_labels = ["Baseline (Normal)", "Disrupted"]
        occ_vals = [baseline_res.final_occupied, disrupted_res.final_occupied]
        avail_vals = [baseline_res.final_available, disrupted_res.final_available]
        wait_vals = [baseline_res.final_waiting, disrupted_res.final_waiting]

        for ev in feasible_evals:
            lbl = ev.solution.title.split(":")[0] if ":" in ev.solution.title else ev.solution.title
            scen_labels.append(lbl)
            occ_vals.append(ev.sim_result.final_occupied)
            avail_vals.append(ev.sim_result.final_available)
            wait_vals.append(ev.sim_result.final_waiting)

        fig_beds.add_trace(go.Bar(
            name="Occupied Beds",
            x=scen_labels,
            y=occ_vals,
            marker_color="#2563eb",
            text=occ_vals,
            textposition="auto"
        ))
        fig_beds.add_trace(go.Bar(
            name="Available Beds",
            x=scen_labels,
            y=avail_vals,
            marker_color="#10b981",
            text=avail_vals,
            textposition="auto"
        ))
        fig_beds.add_trace(go.Bar(
            name="Unmet Demand (Waiting for Bed)",
            x=scen_labels,
            y=wait_vals,
            marker_color="#ef4444",
            text=wait_vals,
            textposition="auto"
        ))

        fig_beds.update_layout(
            barmode="stack",
            height=400,
            xaxis_title="Operational Scenario",
            yaxis_title="Total Patients / Beds",
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig_beds, use_container_width=True)

    with tab_vis2:
        st.caption(f"Hour-by-hour trajectory of patients waiting for an inpatient bed over {duration_hours} hours across scenarios")
        fig_queue = go.Figure()

        fig_queue.add_trace(go.Scatter(
            x=baseline_res.hourly_df["Hour"],
            y=baseline_res.hourly_df["Patients Waiting"],
            mode="lines+markers",
            name="Baseline (Normal)",
            line=dict(color="#2563eb", width=2.5)
        ))
        fig_queue.add_trace(go.Scatter(
            x=disrupted_res.hourly_df["Hour"],
            y=disrupted_res.hourly_df["Patients Waiting"],
            mode="lines+markers",
            name="Disrupted (Active)",
            line=dict(color="#ef4444", width=3, dash="solid")
        ))

        line_colors = ["#10b981", "#8b5cf6", "#06b6d4", "#f59e0b", "#ec4899"]
        for idx, ev in enumerate(feasible_evals):
            col_line = line_colors[idx % len(line_colors)]
            lbl = ev.solution.title.split(":")[0] if ":" in ev.solution.title else ev.solution.title
            fig_queue.add_trace(go.Scatter(
                x=ev.sim_result.hourly_df["Hour"],
                y=ev.sim_result.hourly_df["Patients Waiting"],
                mode="lines+markers",
                name=lbl,
                line=dict(color=col_line, width=2.5, dash="dot")
            ))

        fig_queue.update_layout(
            height=400,
            xaxis_title="Simulation Timeline (Hours)",
            yaxis_title="Patients Waiting in Queue",
            hovermode="x unified",
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig_queue, use_container_width=True)

    with tab_vis3:
        st.caption("Diagnostic CT Scanner backlog and estimated wait times across tested scenarios")
        col_diag_l, col_diag_r = st.columns(2)

        with col_diag_l:
            fig_ct_q = go.Figure()
            fig_ct_q.add_trace(go.Scatter(
                x=baseline_res.hourly_df["Hour"],
                y=baseline_res.hourly_df["CT Queue"],
                mode="lines+markers",
                name="Baseline Queue",
                line=dict(color="#2563eb", width=2)
            ))
            fig_ct_q.add_trace(go.Scatter(
                x=disrupted_res.hourly_df["Hour"],
                y=disrupted_res.hourly_df["CT Queue"],
                mode="lines+markers",
                name="Disrupted Queue",
                line=dict(color="#ef4444", width=2.5)
            ))

            for idx, ev in enumerate(feasible_evals):
                col_line = line_colors[idx % len(line_colors)]
                lbl = ev.solution.title.split(":")[0] if ":" in ev.solution.title else ev.solution.title
                fig_ct_q.add_trace(go.Scatter(
                    x=ev.sim_result.hourly_df["Hour"],
                    y=ev.sim_result.hourly_df["CT Queue"],
                    mode="lines+markers",
                    name=f"{lbl} Queue",
                    line=dict(color=col_line, width=2)
                ))

            fig_ct_q.update_layout(
                title="CT Scan Queue Length (Patients)",
                height=340,
                xaxis_title="Simulation Timeline",
                yaxis_title="Patients in CT Queue",
                template="plotly_white",
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_ct_q, use_container_width=True)

        with col_diag_r:
            ct_scen_labels = ["Baseline", "Disrupted"]
            ct_wait_vals = [baseline_res.est_ct_wait_mins, disrupted_res.est_ct_wait_mins]
            ct_bar_colors = ["#2563eb", "#ef4444"]

            for idx, ev in enumerate(feasible_evals):
                lbl = ev.solution.title.split(":")[0] if ":" in ev.solution.title else ev.solution.title
                ct_scen_labels.append(lbl)
                ct_wait_vals.append(ev.sim_result.est_ct_wait_mins)
                ct_bar_colors.append(line_colors[idx % len(line_colors)])

            fig_ct_wait = go.Figure()
            fig_ct_wait.add_trace(go.Bar(
                x=ct_scen_labels,
                y=ct_wait_vals,
                marker_color=ct_bar_colors,
                text=[f"{v} min" for v in ct_wait_vals],
                textposition="auto"
            ))
            fig_ct_wait.update_layout(
                title="Estimated CT Scan Wait Time (Minutes)",
                height=340,
                yaxis_title="Minutes",
                template="plotly_white",
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_ct_wait, use_container_width=True)

    # -------------------------------------------------------------------------
    # 7. Detailed Hour-by-Hour Simulation Data
    # -------------------------------------------------------------------------
    with st.expander("📄 View Hourly Projection Data Table (Disrupted Scenario)", expanded=False):
        st.dataframe(
            disrupted_res.hourly_df[[
                "Hour", "Occupied Beds", "Available Beds", "Patients Waiting",
                "Bed Demand (New)", "Discharges", "CT Queue", "CT Wait Time (min)"
            ]],
            use_container_width=True,
            hide_index=True
        )

    # Bottom Notice
    st.markdown("---")
    st.caption("🔒 **MediFlow AI Data Notice:** All values presented in the What-If Simulator are synthetic demonstration estimates. MediFlow AI does not process real clinical records or real patient health information (PHI).")


# -----------------------------------------------------------------------------
# 6. Main Application Controller
# -----------------------------------------------------------------------------
def main():
    # Load synthetic demonstration data
    summary, df_arrivals, df_departments, df_resources = load_synthetic_hospital_data()

    # Render Sidebar navigation
    selected_page = render_sidebar()

    # Clean active page title for top navbar
    page_title_clean = selected_page.split(" ", 1)[-1]
    render_top_navbar(page_title_clean, "Operations")

    # Route to selected page
    if "Overview" in selected_page:
        render_overview(summary, df_arrivals, df_departments, df_resources)

    elif "Bed Management" in selected_page:
        render_bed_management(summary, df_departments, df_resources)

    elif "Patient Flow" in selected_page:
        render_patient_flow(summary, df_departments, df_resources)

    elif "Bottleneck" in selected_page:
        render_bottleneck_analysis(summary, df_departments, df_resources)

    elif "Decision Intelligence" in selected_page:
        render_decision_intelligence(summary, df_departments, df_resources)

    elif "Simulator" in selected_page:
        render_what_if_simulator(summary, df_departments, df_resources)

    elif "Staff" in selected_page:
        render_staff_resources(summary, df_departments, df_resources)

    elif "Reports" in selected_page:
        render_reports(summary, df_departments, df_resources)

    elif "Settings" in selected_page:
        render_settings(summary)

    else:
        render_overview(summary, df_arrivals, df_departments, df_resources)


if __name__ == "__main__":
    main()

