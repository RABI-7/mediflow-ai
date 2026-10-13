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

# -----------------------------------------------------------------------------
# 1. Page Configuration & Theme
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="MediFlow AI — Hospital Resource Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished hackathon presentation styling
st.markdown("""
<style>
    /* Metric card styling */
    div[data-testid="stMetric"] {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        padding: 14px 18px;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    /* Header badge */
    .badge-demo {
        display: inline-block;
        background-color: #e0f2fe;
        color: #0369a1;
        font-size: 0.82rem;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 9999px;
        border: 1px solid #bae6fd;
        margin-bottom: 8px;
    }
    
    /* Mission Callout */
    .mission-card {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        border-left: 5px solid #0284c7;
        padding: 16px 20px;
        border-radius: 8px;
        margin-bottom: 24px;
        color: #0f172a;
    }
    
    /* Future Phase placeholder card */
    .future-card {
        background: #f8fafc;
        border: 2px dashed #cbd5e1;
        padding: 40px 30px;
        border-radius: 16px;
        text-align: center;
        margin-top: 20px;
    }
    
    /* Simulator styling */
    .bottleneck-box {
        background-color: #fff1f2;
        border-left: 5px solid #e11d48;
        padding: 16px 20px;
        border-radius: 8px;
        margin-bottom: 20px;
        color: #9f1239;
    }
    .bottleneck-box-stable {
        background-color: #f0fdf4;
        border-left: 5px solid #16a34a;
        padding: 16px 20px;
        border-radius: 8px;
        margin-bottom: 20px;
        color: #166534;
    }
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
        "staff_on_duty": 58,
        "shift_name": "Day Shift (08:00 - 16:00)"
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
        {"Department": "Emergency Dept (ER)", "Capacity": 45, "Occupied": 42, "Staff": 14, "Status": "Critical Load"},
        {"Department": "Intensive Care (ICU)", "Capacity": 30, "Occupied": 27, "Staff": 12, "Status": "Near Capacity"},
        {"Department": "General Medicine",   "Capacity": 95, "Occupied": 82, "Staff": 15, "Status": "Normal Flow"},
        {"Department": "Surgery & Post-Op",   "Capacity": 55, "Occupied": 48, "Staff": 10, "Status": "High Load"},
        {"Department": "Pediatrics",          "Capacity": 35, "Occupied": 24, "Staff": 4,  "Status": "Normal Flow"},
        {"Department": "Cardiology Ward",     "Capacity": 20, "Occupied": 15, "Staff": 3,  "Status": "Normal Flow"}
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
    Renders the left sidebar containing navigation and demo metadata.
    """
    with st.sidebar:
        st.markdown("## 🏥 MediFlow AI")
        st.markdown(
            "*Hospital Resource & Patient Flow Optimization Platform*"
        )
        st.markdown("---")

        st.markdown("### 📌 Navigation")
        page = st.radio(
            "Select Page",
            options=[
                "📊 Hospital Dashboard",
                "🧪 What-If Simulator",
                "⚠️ Bottleneck Analysis",
                "🔮 Patient Forecasting"
            ],
            index=0,
            key="main_nav",
            label_visibility="collapsed"
        )

        st.markdown("---")
        st.markdown("### ⚙️ Demo Configuration")
        st.info("🕒 **Active Shift:** Day Shift (08:00 - 16:00)\n\n📍 **Facility:** Metro General Hospital (Demo)")

        st.markdown("---")
        st.caption("🛡️ **Mode:** Synthetic Data Engine Active")
        st.caption("Version: Phase 2 (What-If Simulator Active)")

    return page


# -----------------------------------------------------------------------------
# 4. View: Main Hospital Dashboard (Phase 1 Working Page)
# -----------------------------------------------------------------------------
def render_dashboard(summary, df_arrivals, df_departments, df_resources):
    """
    Renders the active Phase 1 Dashboard with KPIs, Plotly charts, and tables.
    """
    # Header Section
    st.markdown('<span class="badge-demo">SYNTHETIC DEMONSTRATION DATA</span>', unsafe_allow_html=True)
    st.title("Hospital Resource & Patient Flow Dashboard")

    # Short Explanation of Purpose
    st.markdown("""
    <div class="mission-card">
        <strong>💡 Problem & Purpose:</strong>
        Hospitals face acute challenges when handling sudden patient surges alongside limited physical beds,
        stretched staff, and diagnostic delays. <strong>MediFlow AI</strong> delivers immediate situational awareness
        across critical units so hospital administrators can identify flow constraints early and optimize resource utilization.
    </div>
    """, unsafe_allow_html=True)

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
def render_what_if_simulator():
    """
    Renders the interactive Phase 2 Hospital What-If Simulator.
    Simulates acute operational disruptions and evaluates response strategies.
    """
    st.markdown('<span class="badge-demo">HOSPITAL WHAT-IF STRESS TESTING • PHASE 2</span>', unsafe_allow_html=True)
    st.title("Hospital What-If Simulator")

    # Short Explanation of Purpose
    st.markdown("""
    <div class="mission-card">
        <strong>🧪 Simulation Purpose:</strong>
        Explore the operational consequences of sudden hospital disruptions—such as unexpected patient surges,
        diagnostic equipment outages, and discharge delays—and evaluate illustrative response strategies
        before testing decisions in live hospital environments.
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
                value=30,
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
                options=[4, 8, 12, 24],
                index=2,
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
    # 4. Side-by-Side Scenario Comparison Cards
    # -------------------------------------------------------------------------
    st.subheader("📊 Scenario Comparison: Normal vs. Disrupted vs. Proposed Response")
    st.caption("Compare key resource outcomes across normal operations, the stress-test disruption, and prospective interventions.")

    col_scen_a, col_scen_b, col_scen_c = st.columns(3)

    # --- Scenario A: Baseline ---
    with col_scen_a:
        st.markdown('<div class="scenario-header scenario-baseline">🟢 Scenario A: Baseline (Normal)</div>', unsafe_allow_html=True)
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
            label="Patients Waiting for Bed",
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
        st.caption(f"**Total Discharges:** {baseline_res.total_discharges} | **Demand:** {baseline_res.projected_bed_demand} beds")

    # --- Scenario B: Disrupted ---
    with col_scen_b:
        st.markdown('<div class="scenario-header scenario-disrupted">🔴 Scenario B: Disrupted Operations</div>', unsafe_allow_html=True)
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
            label="Patients Waiting for Bed",
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
        st.caption(f"**Total Discharges:** {disrupted_res.total_discharges} | **Demand:** {disrupted_res.projected_bed_demand} beds")

    # --- Scenario C: Proposed Response ---
    with col_scen_c:
        st.markdown('<div class="scenario-header scenario-response">🔵 Scenario C: Proposed Response</div>', unsafe_allow_html=True)
        flex_note = f" (+{meta['flex_beds_activated']} flex)" if meta['flex_beds_activated'] > 0 else ""
        st.metric(
            label=f"Occupied Beds{flex_note}",
            value=f"{response_res.final_occupied} / {response_res.total_beds}",
            delta=f"{response_res.final_occupancy_pct}% Occupancy",
            delta_color="off"
        )
        st.metric(
            label="Available Beds",
            value=f"{response_res.final_available} beds open",
            delta=f"Maintained buffer",
            delta_color="off"
        )
        st.metric(
            label="Patients Waiting for Bed",
            value=f"{response_res.final_waiting}",
            delta=f"-{mitigated_count} mitigated" if mitigated_count > 0 else "Clear",
            delta_color="normal" if mitigated_count > 0 else "off"
        )
        st.metric(
            label="CT Scanner Queue & Wait",
            value=f"{response_res.final_ct_queue} patients",
            delta=f"{response_res.est_ct_wait_mins} mins wait",
            delta_color="off"
        )
        st.caption(f"**Total Discharges:** {response_res.total_discharges} | **Demand:** {response_res.projected_bed_demand} beds")

    # -------------------------------------------------------------------------
    # 5. Modeling Assumptions & Explanations (Transparent & Beginner Friendly)
    # -------------------------------------------------------------------------
    with st.expander("📋 View Assumptions and Constraints for Each Scenario", expanded=False):
        c_assump1, c_assump2, c_assump3 = st.columns(3)
        with c_assump1:
            st.markdown("**🟢 Scenario A Assumptions:**")
            for a in baseline_res.assumptions:
                st.markdown(f"- {a}")
        with c_assump2:
            st.markdown("**🔴 Scenario B Assumptions:**")
            for a in disrupted_res.assumptions:
                st.markdown(f"- {a}")
        with c_assump3:
            st.markdown("**🔵 Scenario C Assumptions:**")
            for a in response_res.assumptions:
                st.markdown(f"- {a}")
        st.info("⚠️ **Modeling Notice:** Scenario C represents an illustrative exploratory response (e.g. activating discharge lounges, deploying flex surge beds, and prioritizing diagnostic triage). It is designed to assist operational discussions and does not guarantee clinical outcomes.")

    st.markdown("<br>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 6. Interactive Visualizations
    # -------------------------------------------------------------------------
    st.subheader("📈 Visual Comparison Across Scenarios")

    tab_vis1, tab_vis2, tab_vis3 = st.tabs([
        "🛏️ Bed Capacity & Unmet Demand",
        "⏳ Patient Waiting Queue Over Time",
        "🔬 Diagnostic & CT Scanner Workload"
    ])

    with tab_vis1:
        st.caption("Comparison of Occupied Beds, Available Beds, and Unmet Demand (Patients Waiting) across scenarios")
        # Bar chart comparing bed status
        fig_beds = go.Figure()
        scenarios = ["Baseline (Normal)", "Disrupted", "Proposed Response"]
        occupied_vals = [baseline_res.final_occupied, disrupted_res.final_occupied, response_res.final_occupied]
        available_vals = [baseline_res.final_available, disrupted_res.final_available, response_res.final_available]
        waiting_vals = [baseline_res.final_waiting, disrupted_res.final_waiting, response_res.final_waiting]

        fig_beds.add_trace(go.Bar(
            name="Occupied Beds",
            x=scenarios,
            y=occupied_vals,
            marker_color="#2563eb",
            text=occupied_vals,
            textposition="auto"
        ))
        fig_beds.add_trace(go.Bar(
            name="Available Beds",
            x=scenarios,
            y=available_vals,
            marker_color="#10b981",
            text=available_vals,
            textposition="auto"
        ))
        fig_beds.add_trace(go.Bar(
            name="Unmet Demand (Waiting for Bed)",
            x=scenarios,
            y=waiting_vals,
            marker_color="#ef4444",
            text=waiting_vals,
            textposition="auto"
        ))

        fig_beds.update_layout(
            barmode="stack",
            height=400,
            xaxis_title="Simulation Scenario",
            yaxis_title="Total Number of Patients / Beds",
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig_beds, use_container_width=True)

    with tab_vis2:
        st.caption(f"Hour-by-hour trajectory of patients waiting for an inpatient bed over {duration_hours} hours")
        # Line chart of waiting queue over time
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
            name="Disrupted",
            line=dict(color="#ef4444", width=3, dash="solid")
        ))
        fig_queue.add_trace(go.Scatter(
            x=response_res.hourly_df["Hour"],
            y=response_res.hourly_df["Patients Waiting"],
            mode="lines+markers",
            name="Proposed Response",
            line=dict(color="#10b981", width=2.5, dash="dot")
        ))

        fig_queue.update_layout(
            height=400,
            xaxis_title="Simulation Timeline",
            yaxis_title="Patients Waiting in Queue",
            hovermode="x unified",
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig_queue, use_container_width=True)

    with tab_vis3:
        st.caption("Diagnostic CT Scanner backlog and estimated wait times across the simulation hours")
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
            fig_ct_q.add_trace(go.Scatter(
                x=response_res.hourly_df["Hour"],
                y=response_res.hourly_df["CT Queue"],
                mode="lines+markers",
                name="Response Queue",
                line=dict(color="#10b981", width=2)
            ))
            fig_ct_q.update_layout(
                title="CT Scan Queue Length",
                height=340,
                xaxis_title="Simulation Timeline",
                yaxis_title="Patients in CT Queue",
                template="plotly_white",
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_ct_q, use_container_width=True)

        with col_diag_r:
            fig_ct_wait = go.Figure()
            fig_ct_wait.add_trace(go.Bar(
                x=["Baseline", "Disrupted", "Proposed Response"],
                y=[baseline_res.est_ct_wait_mins, disrupted_res.est_ct_wait_mins, response_res.est_ct_wait_mins],
                marker_color=["#2563eb", "#ef4444", "#10b981"],
                text=[f"{baseline_res.est_ct_wait_mins} min", f"{disrupted_res.est_ct_wait_mins} min", f"{response_res.est_ct_wait_mins} min"],
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
    Renders a friendly placeholder for upcoming phases (Bottlenecks, Forecasting).
    """
    st.title(page_name)
    st.markdown(
        f'<span class="badge-demo">SCHEDULED FOR PHASE {phase_number}</span>',
        unsafe_allow_html=True
    )

    st.markdown(f"""
    <div class="future-card">
        <h2 style="color: #0284c7; margin-bottom: 8px;">🚀 Coming in Phase {phase_number}</h2>
        <p style="font-size: 1.1rem; color: #475569; max-width: 700px; margin: 0 auto 24px auto;">
            {feature_description}
        </p>
        <div style="display: inline-block; background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 20px; text-align: left; max-width: 600px;">
            <h4 style="margin-top: 0; color: #0f172a;">Planned Capabilities for this Module:</h4>
            <ul>
                {"".join(f"<li style='margin-bottom: 8px;'>{cap}</li>" for cap in planned_capabilities)}
            </ul>
        </div>
        <p style="margin-top: 24px; color: #64748b; font-size: 0.9rem;">
            <em>Current status: Phase 2 (What-If Simulator) is active. This feature will be introduced when transitioning to Phase {phase_number}.</em>
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
    if selected_page == "📊 Hospital Dashboard":
        render_dashboard(summary, df_arrivals, df_departments, df_resources)

    elif selected_page == "🧪 What-If Simulator":
        render_what_if_simulator()

    elif selected_page == "⚠️ Bottleneck Analysis":
        render_future_phase(
            page_name="Bottleneck Detection & Recommendations",
            phase_number="3",
            feature_description="Algorithmic identification of hospital flow constraints and actionable recommendations to relieve unit pressure.",
            planned_capabilities=[
                "Automated detection of acute capacity bottlenecks across wards and diagnostic units.",
                "Workload balancing suggestions for nursing and medical staff.",
                "Actionable recommendations (e.g., fast-tracking low-acuity discharge, diverting elective scans).",
                "Root-cause drilldowns for delays in patient transfer times."
            ]
        )

    elif selected_page == "🔮 Patient Forecasting":
        render_future_phase(
            page_name="Patient Arrival Forecasting",
            phase_number="4",
            feature_description="Proactive arrival projections to help hospital management prepare bed and staffing capacity ahead of time.",
            planned_capabilities=[
                "Short-term patient arrival predictions based on historical synthetic trends.",
                "Seasonal and day-of-week surge pattern indicators.",
                "Proactive staffing and bed reserve alerts before peak arrival hours.",
                "Interactive scenario-based forecast adjustments."
            ]
        )


if __name__ == "__main__":
    main()
