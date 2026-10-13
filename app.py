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
from forecast_view import render_patient_forecasting
from decision_view import render_decision_intelligence
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
# 3. Sidebar Navigation & Global Controls
# -----------------------------------------------------------------------------
def render_sidebar():
    """
    Renders the left sidebar containing navigation and facility metadata.
    """
    with st.sidebar:
        st.markdown("## 🏥 MediFlow AI")
        st.markdown(
            "**Intelligent Hospital Resource Optimization**"
        )
        st.markdown("---")

        st.markdown("### 📌 Navigation")
        nav_options = [
            "📊 Hospital Dashboard",
            "🎯 Decision Intelligence",
            "⚡ What-If Simulator",
            "⚠️ Bottleneck Analysis",
            "🔮 Patient Forecasting"
        ]
        if "nav_goto" in st.session_state and st.session_state["nav_goto"] in nav_options:
            st.session_state["main_nav"] = st.session_state.pop("nav_goto")
        elif "main_nav" not in st.session_state:
            st.session_state["main_nav"] = nav_options[0]

        page = st.radio(
            "Select Page",
            options=nav_options,
            key="main_nav",
            label_visibility="collapsed"
        )

        st.markdown("""
        <div class="sidebar-promo-box">
            <div style="font-weight: 700; color: #7c3aed; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.04em;">🎯 Phase 5 Engine</div>
            <div style="font-size: 0.82rem; color: #334155; margin-top: 3px;">
                <strong>Decision Intelligence:</strong> 7-step pipeline from monitoring to feasible simulation and human review.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 🏥 Facility Status")
        st.info("🕒 **Shift:** Day Shift (08:00 - 16:00)\n\n📍 **Site:** Metro General Hospital\n\n🟢 **Status:** Live (Demo Telemetry)")

    return page


# -----------------------------------------------------------------------------
# 4. View: Main Hospital Dashboard (Phase 1 Working Page)
# -----------------------------------------------------------------------------
def render_dashboard(summary, df_arrivals, df_departments, df_resources):
    """
    Renders the active Hospital Dashboard with KPIs, Plotly charts, and tables.
    """
    # Global Brand Header
    st.markdown("""
    <div class="brand-header-box">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
            <div>
                <div class="brand-title-text">🏥 MediFlow AI</div>
                <div class="brand-subtitle-text">Intelligent Hospital Resource Optimization</div>
            </div>
            <div>
                <span class="facility-badge">📍 Metro General Hospital</span>
                <span class="live-telemetry-badge">● Operational Telemetry Active</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Clean Page Header Row
    col_dash_left, col_dash_right = st.columns([3, 1])
    with col_dash_left:
        st.subheader("Hospital Operations Dashboard")
        st.caption("Continuous departmental census, patient inflow rate, and capacity utilization.")
    with col_dash_right:
        st.markdown('<div style="text-align: right; padding-top: 6px;"><span class="badge-demo">SYNTHETIC DATA ENGINE</span></div>', unsafe_allow_html=True)

    # Small Optional Information Section (Replaces long problem-statement banner)
    with st.expander("ℹ️ About MediFlow AI & Operational Scope", expanded=False):
        st.markdown("""
        **MediFlow AI** provides real-time situational awareness across hospital beds, triage queues, 
        and critical diagnostic equipment. Clinical operations teams use MediFlow AI to identify flow bottlenecks 
        early and evaluate resource balancing strategies before bed shortages impact patient care.
        """)

    # Prominent Simulator & Decision Intelligence Spotlight Cards
    col_spot1, col_spot2 = st.columns(2)
    with col_spot1:
        st.markdown("""
        <div class="sim-spotlight-card" style="height: 100%;">
            <div>
                <span class="sim-spotlight-badge">⚡ Core Simulation Engine</span>
                <div class="sim-spotlight-title">Hospital What-If Simulator</div>
                <div class="sim-spotlight-desc">
                    Stress-test hospital capacity under acute shocks: simulate surges, CT scanner outages, 
                    and discharge delays side-by-side with prospective response strategies.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("⚡ Open What-If Simulator", use_container_width=True, key="dash_btn_sim"):
            st.session_state["nav_goto"] = "⚡ What-If Simulator"
            st.rerun()
    with col_spot2:
        st.markdown("""
        <div class="sim-spotlight-card" style="border-left: 5px solid #8b5cf6; background: linear-gradient(135deg, #fdf4ff 0%, #ffffff 100%); height: 100%;">
            <div>
                <span class="sim-spotlight-badge" style="background-color: #8b5cf6;">🎯 Phase 5 Engine</span>
                <div class="sim-spotlight-title" style="color: #581c87;">Integrated Decision Intelligence</div>
                <div class="sim-spotlight-desc">
                    7-step decision pipeline: Monitor ➔ Detect bottlenecks & chains ➔ Forecast arrival demand ➔ Propose feasible responses ➔ Simulate trade-offs ➔ Human review.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🚀 Launch Decision Intelligence Engine", type="primary", use_container_width=True, key="dash_btn_di"):
            st.session_state["nav_goto"] = "🎯 Decision Intelligence"
            st.rerun()

    # ---------------------------------------------------------
    # Metric Summary Cards
    # ---------------------------------------------------------
    st.subheader("Key Operational Metrics")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Occupied Beds",
            value=f"{summary['occupied_beds']} / {summary['total_beds']}",
            delta=f"{summary['occupancy_rate_pct']}% Occupancy Rate",
            delta_color="inverse"
        )
    with col2:
        st.metric(
            label="Available Beds",
            value=f"{summary['available_beds']}",
            delta="-4 from last hour",
            delta_color="normal"
        )
    with col3:
        st.metric(
            label="Patients Waiting (Triage/Queue)",
            value=f"{summary['patients_waiting']}",
            delta="+7 surge alert",
            delta_color="inverse"
        )
    with col4:
        st.metric(
            label="Medical Staff on Duty",
            value=f"{summary['staff_on_duty']}",
            delta="5 shifts active",
            delta_color="off"
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # Interactive Visualizations
    # ---------------------------------------------------------
    col_chart_left, col_chart_right = st.columns([1, 1])

    with col_chart_left:
        st.subheader("📈 Patient Arrivals Over 24 Hours")
        st.caption("Hourly influx categorized by clinical triage severity")

        # Plotly Line/Area Chart for Arrivals
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
            height=380,
            margin=dict(l=20, r=20, t=30, b=20),
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis_title="Time of Day",
            yaxis_title="Patient Influx Count",
            template="plotly_white"
        )
        st.plotly_chart(fig_arrivals, use_container_width=True)

    with col_chart_right:
        st.subheader("🛏️ Bed Capacity & Occupancy by Department")
        st.caption("Comparison of occupied beds versus remaining capacity")

        # Plotly Bar Chart for Departmental Bed Occupancy
        fig_dept = go.Figure()
        fig_dept.add_trace(go.Bar(
            y=df_departments["Department"],
            x=df_departments["Occupied"],
            name="Occupied Beds",
            orientation="h",
            marker=dict(color="#3b82f6")
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
            height=380,
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis_title="Number of Beds",
            yaxis_title="",
            template="plotly_white"
        )
        st.plotly_chart(fig_dept, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # Hospital Resource & Equipment Table
    # ---------------------------------------------------------
    st.subheader("📋 Hospital Department & Critical Equipment Status")
    st.caption("Synthetic snapshot of departmental loads and diagnostic equipment availability")

    tab1, tab2 = st.tabs(["🏥 Department Bed Summary", "🔬 Critical Equipment & Staffing"])

    with tab1:
        # Format Department table cleanly
        dept_display = df_departments[["Department", "Capacity", "Occupied", "Available", "Occupancy %", "Staff", "Status"]]
        st.dataframe(
            dept_display,
            use_container_width=True,
            hide_index=True
        )

    with tab2:
        # Critical Equipment and Diagnostics
        st.dataframe(
            df_resources,
            use_container_width=True,
            hide_index=True
        )

    # Bottom Notice
    st.markdown("---")
    st.caption("🔒 **MediFlow AI Data Notice:** All values presented above are simulated synthetic values for demonstration purposes. MediFlow AI does not store, transmit, or process protected patient health information (PHI).")


# -----------------------------------------------------------------------------
# 5. View: Hospital What-If Simulator (Phase 2 Working Page)
# -----------------------------------------------------------------------------
def render_what_if_simulator(summary=None, df_departments=None, df_resources=None):
    """
    Renders the interactive Hospital What-If Simulator with solution recommendations and comparisons.
    Simulates acute operational disruptions, proposes tailored interventions, and enforces feasibility.
    """
    if summary is None or df_departments is None:
        summary, _, df_departments, df_resources = load_synthetic_hospital_data()
    # Global Brand Header
    st.markdown("""
    <div class="brand-header-box">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
            <div>
                <div class="brand-title-text">🏥 MediFlow AI</div>
                <div class="brand-subtitle-text">Intelligent Hospital Resource Optimization</div>
            </div>
            <div>
                <span class="facility-badge">📍 Metro General Hospital</span>
                <span class="live-telemetry-badge">⚡ Core Simulation Engine Active</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Clean Page Header Row
    col_sim_left, col_sim_right = st.columns([3, 1])
    with col_sim_left:
        st.subheader("Hospital What-If Simulator")
        st.caption("Interactive capacity stress-testing: evaluate sudden patient surges, diagnostic outages, and discharge interventions.")
    with col_sim_right:
        st.markdown('<div style="text-align: right; padding-top: 6px;"><span class="badge-demo">SCENARIO STRESS-TEST</span></div>', unsafe_allow_html=True)

    # Small Optional Information Section
    with st.expander("ℹ️ About Simulation Methodology & Assumptions", expanded=False):
        st.markdown("""
        The simulator executes a deterministic hour-by-hour resource flow model that strictly respects physical bed limits, 
        admission turnover constraints, and diagnostic throughput. It compares **Baseline (Normal Operations)**, 
        **Disrupted**, and **Proposed Response** scenarios side-by-side to assist clinical leaders in capacity planning.
        """)

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
    st.caption("🔒 **MediFlow AI Data Notice:** All values presented in the What-If Simulator are synthetic demonstration estimates for hackathon evaluation. MediFlow AI does not process real clinical records or real patient health information (PHI).")


# -----------------------------------------------------------------------------
# 6. Views: Future Phase Placeholders
# -----------------------------------------------------------------------------
def render_future_phase(page_name, phase_number, feature_description, planned_capabilities):
    """
    Renders a modern product roadmap card for upcoming capabilities.
    """
    # Global Brand Header
    st.markdown("""
    <div class="brand-header-box">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
            <div>
                <div class="brand-title-text">🏥 MediFlow AI</div>
                <div class="brand-subtitle-text">Intelligent Hospital Resource Optimization</div>
            </div>
            <div>
                <span class="facility-badge">📍 Metro General Hospital</span>
                <span class="live-telemetry-badge">🔮 Product Roadmap</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_fut_l, col_fut_r = st.columns([3, 1])
    with col_fut_l:
        st.subheader(page_name)
    with col_fut_r:
        st.markdown(f'<div style="text-align: right; padding-top: 6px;"><span class="badge-demo">SCHEDULED • PHASE {phase_number}</span></div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="future-card">
        <h2 style="color: #0284c7; margin-bottom: 8px;">🚀 Planned for Phase {phase_number}</h2>
        <p style="font-size: 1.05rem; color: #475569; max-width: 700px; margin: 0 auto 24px auto;">
            {feature_description}
        </p>
        <div style="display: inline-block; background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px; padding: 20px; text-align: left; max-width: 600px;">
            <h4 style="margin-top: 0; color: #0f172a;">Planned Capabilities for this Module:</h4>
            <ul>
                {"".join(f"<li style='margin-bottom: 8px;'>{cap}</li>" for cap in planned_capabilities)}
            </ul>
        </div>
        <p style="margin-top: 24px; color: #64748b; font-size: 0.9rem;">
            <em>Status: Module under active roadmap development. Available in upcoming version rollout.</em>
        </p>
    </div>
    """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 7. Main Application Controller
# -----------------------------------------------------------------------------
def main():
    # Load synthetic demonstration data
    summary, df_arrivals, df_departments, df_resources = load_synthetic_hospital_data()

    # Render Sidebar navigation
    selected_page = render_sidebar()

    # Route to selected page
    if "Dashboard" in selected_page:
        render_dashboard(summary, df_arrivals, df_departments, df_resources)

    elif "Decision Intelligence" in selected_page:
        render_decision_intelligence(summary, df_departments, df_resources)

    elif "Simulator" in selected_page:
        render_what_if_simulator(summary, df_departments, df_resources)

    elif "Bottleneck" in selected_page:
        render_bottleneck_analysis(summary, df_departments, df_resources)

    elif "Forecasting" in selected_page:
        render_patient_forecasting(summary, df_departments, df_resources)


if __name__ == "__main__":
    main()
