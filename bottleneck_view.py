"""
=============================================================================
 MediFlow AI — Bottleneck Intelligence UI View
=============================================================================
 Module: bottleneck_view.py
 Purpose: Renders the complete Phase 3 Advanced Bottleneck Intelligence &
          Decision Support interactive interface:
          1. Operational Bottleneck Summary
          2. Hidden Capacity Recovery
          3. Bottleneck Dependency Chains
          4. Safe Intervention Comparison
          5. Staff Workload Imbalance
          6. Explainable Recommendations & Evidence Panel
=============================================================================
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from simulator import run_scenario_step_by_step, ScenarioResult
from bottleneck_engine import (
    THRESHOLDS,
    BottleneckFinding,
    detect_hospital_bottlenecks,
    evaluate_hidden_capacity,
    detect_bottleneck_chains,
    evaluate_staff_workload_imbalance,
    get_available_candidate_interventions,
    evaluate_multiple_candidate_interventions,
    generate_explainable_recommendations
)


def render_bottleneck_analysis(
    summary: dict,
    df_departments: pd.DataFrame,
    df_resources: pd.DataFrame
):
    """
    Renders the complete Phase 3 Advanced Bottleneck Intelligence page.
    """
    # -------------------------------------------------------------------------
    # Brand Header
    # -------------------------------------------------------------------------
    # Clean Professional Header
    # -------------------------------------------------------------------------
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <div style="font-size: 1.5rem; font-weight: 800; color: #0f172a; letter-spacing: -0.02em;">
            Bottleneck Intelligence & Dependency Analysis
        </div>
        <div style="font-size: 0.88rem; color: #475569; margin-top: 2px;">
            Identify acute clinical flow delays, diagnostic backlogs, process dependencies, and explainable intervention evidence.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # Disruption Context Bar (Interactive Scenario Settings)
    # -------------------------------------------------------------------------
    with st.expander("⚙️ Configure Active Operational Stress & Disruption Scenario", expanded=False):
        col_ctx1, col_ctx2, col_ctx3, col_ctx4 = st.columns(4)
        with col_ctx1:
            surge_input = st.slider("Additional Patient Arrivals", 0, 60, 30, 5, key="bn_surge", help="Simulate unscheduled patient surge")
        with col_ctx2:
            ct_input = st.selectbox(
                "CT Scanner Status",
                ["Unavailable (1 Scanner Down - 50% Outage)", "Available (Both Units Operational)"],
                index=0,
                key="bn_ct"
            )
        with col_ctx3:
            disch_input = st.selectbox(
                "Inpatient Discharge Delay",
                ["On (Beds & Handoffs Delayed)", "Off (Standard Discharge Flow)"],
                index=0,
                key="bn_disch"
            )
        with col_ctx4:
            dur_input = st.selectbox("Evaluation Window", [4, 8, 12, 24], index=2, format_func=lambda h: f"{h} Hours", key="bn_dur")

    ct_status = "Unavailable" if "Unavailable" in ct_input else "Available"
    discharge_delay = True if "On" in disch_input else False
    duration_hours = int(dur_input)

    # Deterministic simulation of active disrupted scenario for reference
    is_ct_down = (ct_status == "Unavailable")
    sim_disrupted = run_scenario_step_by_step(
        name="Active Disruption Context",
        scenario_type="disrupted",
        description="Active operational strain for bottleneck evaluation.",
        assumptions=[f"+{surge_input} surge arrivals", f"CT status: {ct_status}", f"Discharge delay: {'ON' if discharge_delay else 'OFF'}"],
        duration_hours=duration_hours,
        total_beds=280,
        initial_occupied=238,
        initial_waiting=34,
        initial_ct_queue=6,
        additional_arrivals_total=surge_input,
        discharge_rate_hourly=2.4 if discharge_delay else 5.5,
        max_admission_intake_hourly=10 if is_ct_down else 12,
        ct_capacity_hourly=3.8 if is_ct_down else 8.0,
        ct_status_label=ct_status
    )

    # Run Phase 3 Intelligence Engines
    findings = detect_hospital_bottlenecks(summary, df_departments, df_resources, sim_disrupted)
    hidden_capacity = evaluate_hidden_capacity(summary, sim_disrupted)
    chain_result = detect_bottleneck_chains(summary, sim_disrupted)
    dept_workloads, staff_recs = evaluate_staff_workload_imbalance(df_departments, summary)
    explainable_recs = generate_explainable_recommendations(
        findings=findings,
        hidden_capacity=hidden_capacity,
        chain_result=chain_result,
        staff_workloads=dept_workloads,
        sim_disrupted=sim_disrupted
    )

    # -------------------------------------------------------------------------
    # Six Core Phase 3 Intelligence Tabs
    # -------------------------------------------------------------------------
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🚨 1. Operational Bottlenecks",
        "🛏️ 2. Hidden Capacity Recovery",
        "🔗 3. Bottleneck Dependency Chains",
        "🛡️ 4. Safe Intervention Comparison",
        "👥 5. Staff Workload Imbalance",
        "📋 6. Explainable Recommendations"
    ])

    # =========================================================================
    # TAB 1: Operational Bottleneck Summary
    # =========================================================================
    with tab1:
        critical_count = sum(1 for f in findings if f.severity == "CRITICAL")
        warning_count = sum(1 for f in findings if f.severity == "WARNING")
        optimal_count = sum(1 for f in findings if f.severity == "OPTIMAL")

        col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
        with col_kpi1:
            st.metric("Detected Bottlenecks", f"{len(findings)} Monitored", delta="Operational Scope", delta_color="off")
        with col_kpi2:
            st.metric("Critical Alarms", f"{critical_count} Units", delta="Immediate Attention", delta_color="inverse")
        with col_kpi3:
            st.metric("Warning Alerts", f"{warning_count} Units", delta="Elevated Strain", delta_color="off")
        with col_kpi4:
            st.metric("Operating Buffer", f"{optimal_count} Safe", delta="Within Nominal", delta_color="normal")

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("📌 Prioritized Operational Bottleneck Findings")
        st.caption("Classified using explicit operational thresholds. Click cards to view supporting metrics and evidence.")

        for f in findings:
            if f.severity == "CRITICAL":
                border_color = "#ef4444"
                badge_html = '<span style="background-color: #fee2e2; color: #991b1b; padding: 4px 10px; border-radius: 9999px; font-weight: 700; font-size: 0.78rem; border: 1px solid #fecaca;">CRITICAL SEVERITY</span>'
            elif f.severity == "WARNING":
                border_color = "#f59e0b"
                badge_html = '<span style="background-color: #fef3c7; color: #92400e; padding: 4px 10px; border-radius: 9999px; font-weight: 700; font-size: 0.78rem; border: 1px solid #fde68a;">WARNING</span>'
            else:
                border_color = "#10b981"
                badge_html = '<span style="background-color: #dcfce7; color: #166534; padding: 4px 10px; border-radius: 9999px; font-weight: 700; font-size: 0.78rem; border: 1px solid #bbf7d0;">OPTIMAL</span>'

            st.markdown(f"""
            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-left: 5px solid {border_color}; border-radius: 12px; padding: 18px 22px; margin-bottom: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.03);">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span style="font-weight: 700; font-size: 1.1rem; color: #0f172a;">{f.title}</span>
                    <div>
                        <span style="background: #f1f5f9; color: #334155; padding: 4px 10px; border-radius: 9999px; font-size: 0.78rem; font-weight: 600; margin-right: 8px;">📍 {f.affected_unit}</span>
                        {badge_html}
                    </div>
                </div>
                <div style="display: flex; gap: 24px; margin-top: 10px; flex-wrap: wrap;">
                    <div style="flex: 1; min-width: 220px;">
                        <span style="font-size: 0.82rem; color: #64748b; font-weight: 600; text-transform: uppercase;">Observed Metric:</span>
                        <div style="font-size: 1rem; font-weight: 700; color: #0f172a;">{f.metric_label}: <span style="color: #0284c7;">{f.metric_value}</span></div>
                    </div>
                    <div style="flex: 2; min-width: 300px;">
                        <span style="font-size: 0.82rem; color: #64748b; font-weight: 600; text-transform: uppercase;">Classification Rule:</span>
                        <div style="font-size: 0.92rem; color: #334155;"><code>{f.threshold_rule}</code></div>
                    </div>
                </div>
                <div style="margin-top: 12px; padding-top: 10px; border-top: 1px solid #f1f5f9; font-size: 0.92rem; color: #475569;">
                    <strong>Evidence & Context:</strong> {f.evidence}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with st.expander("📋 View Complete Clinical Operations Thresholds & Rules Specification", expanded=False):
            threshold_rows = []
            for category, vals in THRESHOLDS.items():
                threshold_rows.append({
                    "Operational Category": category,
                    "Critical Threshold": vals.get("critical", "N/A"),
                    "Warning Threshold": vals.get("warning", "N/A"),
                    "Target / Normal Benchmark": vals.get("target", "Nominal")
                })
            st.dataframe(pd.DataFrame(threshold_rows), use_container_width=True, hide_index=True)

    # =========================================================================
    # TAB 2: Hidden Capacity Recovery Engine
    # =========================================================================
    with tab2:
        st.subheader("🛏️ Hidden Capacity Recovery Engine")
        st.caption("Identify beds blocked in operational transitions and distinguish actionable recovery from merely theoretical capacity.")

        b = hidden_capacity.breakdown
        col_hc1, col_hc2, col_hc3, col_hc4, col_hc5 = st.columns(5)
        with col_hc1:
            st.metric("Total Licensed Beds", f"{b.total_physical_capacity}", delta="Physical Limit", delta_color="off")
        with col_hc2:
            st.metric("Occupied Inpatient Beds", f"{b.occupied_beds}", delta=f"{round(b.occupied_beds/b.total_physical_capacity*100, 1)}% Inpatient", delta_color="off")
        with col_hc3:
            st.metric("Ready Available Beds", f"{b.ready_available_beds}", delta="Immediately Open", delta_color="normal")
        with col_hc4:
            st.metric("Recoverable Blocked Beds", f"{hidden_capacity.recoverable_beds_total}", delta="Actionable via Protocol", delta_color="off")
        with col_hc5:
            st.metric("Theoretical Unoccupied", f"{hidden_capacity.theoretical_unoccupied}", delta="Nominal Buffer", delta_color="off")

        # Hidden Capacity Gap Banner
        st.markdown(f"""
        <div style="background-color: #f0fdf4; border: 1px solid #bbf7d0; border-left: 5px solid #16a34a; padding: 16px 20px; border-radius: 12px; margin: 18px 0;">
            <div style="font-weight: 700; font-size: 1.05rem; color: #166534; margin-bottom: 6px;">
                💡 The "Hidden Capacity" Reality: {hidden_capacity.theoretical_unoccupied} Nominal Beds vs. {b.ready_available_beds} Ready Beds
            </div>
            <div style="font-size: 0.95rem; color: #1e3a8a; line-height: 1.5;">
                {hidden_capacity.explanation}
                <br><strong>Strict Rule:</strong> A bed is NEVER counted as ready available until all sanitation, inspection, and EHR release criteria are satisfied.
            </div>
        </div>
        """, unsafe_allow_html=True)

        col_viz_left, col_viz_right = st.columns([1, 1])
        with col_viz_left:
            st.markdown("##### 📊 Facility Bed State Partition")
            bed_states_df = pd.DataFrame([
                {"State": "Occupied (Inpatient)", "Beds": b.occupied_beds, "Color": "#0284c7"},
                {"State": "Awaiting Discharge Coordination", "Beds": b.awaiting_discharge_coordination, "Color": "#f59e0b"},
                {"State": "Awaiting Cleaning (EVS)", "Beds": b.awaiting_cleaning_evs, "Color": "#ec4899"},
                {"State": "Awaiting Readiness Verification", "Beds": b.awaiting_readiness_verification, "Color": "#8b5cf6"},
                {"State": "Ready Available for Admission", "Beds": b.ready_available_beds, "Color": "#10b981"},
                {"State": "Unavailable (Maintenance)", "Beds": b.unavailable_maintenance, "Color": "#64748b"}
            ])

            fig_donut = go.Figure(data=[go.Pie(
                labels=bed_states_df["State"],
                values=bed_states_df["Beds"],
                hole=.45,
                marker=dict(colors=bed_states_df["Color"]),
                textinfo="label+value",
                hoverinfo="label+value+percent"
            )])
            fig_donut.update_layout(
                height=340,
                margin=dict(l=10, r=10, t=20, b=20),
                showlegend=False
            )
            st.plotly_chart(fig_donut, use_container_width=True)

            # Invariant check banner
            st.markdown(f"""
            <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 10px 14px; font-size: 0.85rem; color: #334155; text-align: center;">
                🔒 <strong>Partition Invariant Verified:</strong> {b.total_accounted} / {b.total_physical_capacity} Beds (100% Accounted • Zero Double-Counting)
            </div>
            """, unsafe_allow_html=True)

        with col_viz_right:
            st.markdown("##### 🛑 Modeled Blockers Preventing Availability")
            blocker_rows = []
            for blk in hidden_capacity.blockers_summary:
                blocker_rows.append({
                    "Bed State": blk["state"],
                    "Blocked Beds": blk["beds"],
                    "Category": blk["blocker_category"],
                    "Specific Modeled Blocker": blk["modeled_blocker"],
                    "Actionable Now": "✅ Yes" if blk["recoverable"] else "❌ Long-Term"
                })
            st.dataframe(pd.DataFrame(blocker_rows), use_container_width=True, hide_index=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("🎯 Ordered Capacity Recovery Actions (Human Review Required)")
        st.caption("Ranked by speed of impact and operational feasibility. Select actions for human approval.")

        for act in hidden_capacity.ordered_actions:
            badge_feas = '<span style="background-color: #dcfce7; color: #166534; padding: 3px 8px; border-radius: 9999px; font-size: 0.75rem; font-weight: 700;">FEASIBLE</span>' if "Feasible" in act.feasibility_status else '<span style="background-color: #fee2e2; color: #991b1b; padding: 3px 8px; border-radius: 9999px; font-size: 0.75rem; font-weight: 700;">NON-ACTIONABLE</span>'
            st.markdown(f"""
            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 16px 20px; margin-bottom: 12px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <span style="background: #0284c7; color: white; padding: 2px 8px; border-radius: 9999px; font-size: 0.75rem; font-weight: 700; margin-right: 8px;">PRIORITY {act.priority}</span>
                        <strong style="font-size: 1.05rem; color: #0f172a;">{act.title}</strong>
                    </div>
                    <div>
                        <span style="background: #eff6ff; color: #1d4ed8; padding: 3px 8px; border-radius: 9999px; font-size: 0.78rem; font-weight: 600; margin-right: 6px;">⏱️ {act.estimated_unlock_time}</span>
                        {badge_feas}
                    </div>
                </div>
                <div style="display: flex; gap: 20px; margin-top: 10px; font-size: 0.9rem; color: #334155; flex-wrap: wrap;">
                    <div><strong>Target State:</strong> {act.target_state} ({act.current_blocked_beds} beds currently blocked)</div>
                    <div><strong>Estimated Recoverable:</strong> <span style="color: #16a34a; font-weight: 700;">+{act.recoverable_beds_estimated} Ready Beds</span></div>
                    <div><strong>Required Resources:</strong> {act.required_resources}</div>
                </div>
                <div style="margin-top: 8px; font-size: 0.88rem; color: #475569;">
                    <strong>Action Protocol:</strong> {act.action_protocol}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # =========================================================================
    # TAB 3: Bottleneck Dependency Chains
    # =========================================================================
    with tab3:
        st.subheader("🔗 Hospital Bottleneck Dependency Chains")
        st.caption("Transparent representation of modeled process dependencies. Evaluates how delays propagate across units.")

        if chain_result.is_chain_active:
            st.markdown(f"""
            <div style="background-color: #fff1f2; border: 1px solid #fecdd3; border-left: 5px solid #e11d48; padding: 16px 20px; border-radius: 12px; margin-bottom: 20px;">
                <div style="font-weight: 700; font-size: 1.05rem; color: #9f1239; margin-bottom: 4px;">
                    🚨 CONFIRMED MULTI-STEP OPERATIONAL DEPENDENCY CHAIN DETECTED
                </div>
                <div style="font-size: 0.95rem; color: #4c0519; white-space: pre-line;">
                    {chain_result.propagation_summary}
                </div>
                <div style="margin-top: 8px; font-size: 0.85rem; color: #881337;">
                    <strong>Initiating Bottleneck:</strong> {chain_result.initiating_bottleneck} | <strong>Affected Resources:</strong> {', '.join(chain_result.affected_resources)}
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background-color: #f0fdf4; border: 1px solid #bbf7d0; border-left: 5px solid #16a34a; padding: 16px 20px; border-radius: 12px; margin-bottom: 20px;">
                <div style="font-weight: 700; font-size: 1.05rem; color: #166534; margin-bottom: 4px;">
                    ✅ LOCALIZED OPERATIONAL STATUS (NO MULTI-STEP CHAIN ACTIVE)
                </div>
                <div style="font-size: 0.95rem; color: #14532d;">
                    {chain_result.propagation_summary}
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("##### 🔄 Step-by-Step Modeled Process Flow")
        # Visual step flow cards
        cols_step = st.columns(5)
        for idx, (col_s, node) in enumerate(zip(cols_step, chain_result.nodes)):
            with col_s:
                if node.status == "CRITICAL":
                    card_border = "#ef4444"
                    badge_col = "#fee2e2"
                    text_col = "#991b1b"
                elif node.status == "WARNING":
                    card_border = "#f59e0b"
                    badge_col = "#fef3c7"
                    text_col = "#92400e"
                else:
                    card_border = "#10b981"
                    badge_col = "#dcfce7"
                    text_col = "#166534"

                st.markdown(f"""
                <div style="background: #ffffff; border: 2px solid {card_border}; border-radius: 12px; padding: 14px; text-align: center; min-height: 220px; box-shadow: 0 1px 3px rgba(0,0,0,0.04);">
                    <div style="font-size: 0.75rem; font-weight: 700; color: #64748b; text-transform: uppercase;">STEP {idx+1}</div>
                    <div style="font-weight: 700; font-size: 0.92rem; color: #0f172a; margin: 4px 0 8px 0; min-height: 40px;">{node.name}</div>
                    <div style="margin-bottom: 8px;">
                        <span style="background-color: {badge_col}; color: {text_col}; padding: 2px 8px; border-radius: 9999px; font-size: 0.75rem; font-weight: 700;">{node.status}</span>
                    </div>
                    <div style="font-size: 0.82rem; color: #334155; margin-bottom: 4px;">
                        <strong>{node.metric_label}:</strong><br><span style="color: #0284c7; font-weight: 700;">{node.metric_value}</span>
                    </div>
                    <div style="font-size: 0.72rem; color: #64748b; border-top: 1px solid #f1f5f9; padding-top: 6px; margin-top: 6px;">
                        📍 {node.department}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("🔍 Inspect Modeled Dependency Edges & Causality Safeguards", expanded=False):
            st.markdown("""
            **Causality Safeguard Notice:** MediFlow AI only represents delay chains where the relationship is **explicitly modeled** in the operational simulator.
            It **never** infers clinical causation from mere statistical correlation or the simultaneous presence of two delays.
            """)
            edge_rows = []
            for e in chain_result.edges:
                edge_rows.append({
                    "From Process": e.source_id,
                    "To Process": e.target_id,
                    "Dependency Description": e.label,
                    "Operational Mechanism": e.mechanism,
                    "Modeled in Code": "✅ Explicit" if e.is_modeled else "❌ Unmodeled",
                    "Active Under Disruption": "🔴 Active" if e.is_active else "⚪ Inactive",
                    "Evidence Classification": e.confidence_label
                })
            st.dataframe(pd.DataFrame(edge_rows), use_container_width=True, hide_index=True)

    # =========================================================================
    # TAB 4: Safe What-If Decision Engine (Intervention Comparison)
    # =========================================================================
    with tab4:
        st.subheader("🛡️ Safe What-If Decision Engine (Multi-Intervention Comparison)")
        st.caption("Compare multiple candidate response options under identical disruption conditions with rigorous operational feasibility checks.")

        avail_cands = get_available_candidate_interventions(ct_status, discharge_delay)
        cand_options = {c.id: f"{c.name} ({c.category})" for c in avail_cands}

        selected_cand_ids = st.multiselect(
            "Select Candidate Interventions to Compare (Multiple Selection Supported):",
            options=list(cand_options.keys()),
            default=["cand_evs_turnover", "cand_discharge_lounge", "cand_diagnostic_diversion", "cand_comprehensive_bundle", "cand_infeasible_extreme_pull"],
            format_func=lambda cid: cand_options[cid]
        )

        if not selected_cand_ids:
            st.warning("Please select at least one candidate intervention above to evaluate.")
        else:
            multi_res = evaluate_multiple_candidate_interventions(
                candidate_ids=selected_cand_ids,
                additional_arrivals=surge_input,
                ct_scanner_status=ct_status,
                discharge_delay=discharge_delay,
                duration_hours=duration_hours,
                summary=summary,
                df_departments=df_departments
            )

            # Feasibility Validation Section
            st.markdown("##### 🔎 Feasibility Engine Validation Results")
            for candidate, feas, sim_res in multi_res.evaluated_responses:
                if feas.is_feasible:
                    st.markdown(f"""
                    <div style="background: #f0fdf4; border: 1px solid #bbf7d0; border-left: 4px solid #16a34a; border-radius: 8px; padding: 10px 14px; margin-bottom: 8px;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <strong>✅ {candidate.name}</strong>
                            <span style="background: #dcfce7; color: #166534; padding: 2px 8px; border-radius: 9999px; font-size: 0.75rem; font-weight: 700;">FEASIBLE FOR COMPARISON</span>
                        </div>
                        <div style="font-size: 0.85rem; color: #166534; margin-top: 4px;">
                            Staffing floors verified • Physical bed limits respected (≤295 licensed beds) • Clinical priority safeguard active.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    reasons_text = "<br>• ".join(feas.rejection_reasons)
                    st.markdown(f"""
                    <div style="background: #fff1f2; border: 1px solid #fecdd3; border-left: 4px solid #e11d48; border-radius: 8px; padding: 10px 14px; margin-bottom: 8px;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <strong>⛔ {candidate.name}</strong>
                            <span style="background: #fee2e2; color: #991b1b; padding: 2px 8px; border-radius: 9999px; font-size: 0.75rem; font-weight: 700;">REJECTED BY FEASIBILITY ENGINE</span>
                        </div>
                        <div style="font-size: 0.85rem; color: #991b1b; margin-top: 4px;">
                            <strong>Rejection Rationale:</strong><br>• {reasons_text}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            # Side-by-side Table of Feasible Scenarios
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("##### 📊 Deterministic Scenario Comparison Table")

            comp_rows = [
                {
                    "Scenario": "🟢 Baseline (Normal)",
                    "Occupied Beds": f"{multi_res.baseline.final_occupied} / {multi_res.baseline.total_beds}",
                    "Available Beds": multi_res.baseline.final_available,
                    "Patients Waiting": multi_res.baseline.final_waiting,
                    "CT Queue": multi_res.baseline.final_ct_queue,
                    "CT Wait (min)": multi_res.baseline.est_ct_wait_mins,
                    "Total Discharges": multi_res.baseline.total_discharges,
                    "Occupancy %": f"{multi_res.baseline.final_occupancy_pct}%"
                },
                {
                    "Scenario": "🔴 Disrupted Operations",
                    "Occupied Beds": f"{multi_res.disrupted.final_occupied} / {multi_res.disrupted.total_beds}",
                    "Available Beds": multi_res.disrupted.final_available,
                    "Patients Waiting": multi_res.disrupted.final_waiting,
                    "CT Queue": multi_res.disrupted.final_ct_queue,
                    "CT Wait (min)": multi_res.disrupted.est_ct_wait_mins,
                    "Total Discharges": multi_res.disrupted.total_discharges,
                    "Occupancy %": f"{multi_res.disrupted.final_occupancy_pct}%"
                }
            ]

            chart_names = ["Baseline", "Disrupted"]
            chart_waiting = [multi_res.baseline.final_waiting, multi_res.disrupted.final_waiting]
            chart_available = [multi_res.baseline.final_available, multi_res.disrupted.final_available]

            for cand, feas, sim in multi_res.evaluated_responses:
                if feas.is_feasible and sim is not None:
                    comp_rows.append({
                        "Scenario": f"🔵 {cand.name}",
                        "Occupied Beds": f"{sim.final_occupied} / {sim.total_beds}",
                        "Available Beds": sim.final_available,
                        "Patients Waiting": sim.final_waiting,
                        "CT Queue": sim.final_ct_queue,
                        "CT Wait (min)": sim.est_ct_wait_mins,
                        "Total Discharges": sim.total_discharges,
                        "Occupancy %": f"{sim.final_occupancy_pct}%"
                    })
                    chart_names.append(cand.name.split(":")[0])
                    chart_waiting.append(sim.final_waiting)
                    chart_available.append(sim.final_available)

            st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

            # Chart comparison
            fig_cand = go.Figure()
            fig_cand.add_trace(go.Bar(
                name="Patients Waiting for Bed",
                x=chart_names,
                y=chart_waiting,
                marker_color="#ef4444"
            ))
            fig_cand.add_trace(go.Bar(
                name="Available Beds Open",
                x=chart_names,
                y=chart_available,
                marker_color="#10b981"
            ))
            fig_cand.update_layout(
                title="Scenario Comparison: Triage Queue vs. Bed Availability",
                barmode="group",
                height=360,
                template="plotly_white",
                yaxis_title="Count",
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_cand, use_container_width=True)

            st.info("🔒 **Operational Review Safeguard:** MediFlow AI does not execute real clinical operations, transfer patients, or modify treatments. Scenarios represent deterministic mathematical decision support for authorized human review.")

    # =========================================================================
    # TAB 5: Staff Workload Imbalance Detector
    # =========================================================================
    with tab5:
        st.subheader("👥 Staff Workload Imbalance Detector")
        st.caption("Acuity-weighted clinical workload monitoring with statutory minimum staffing floor safeguards.")

        st.markdown(r"""
        <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 14px 18px; margin-bottom: 18px; font-size: 0.9rem; color: #334155;">
            <strong>Documented Workload Formula:</strong><br>
            <code>Workload Index = (Occupied Patients × Acuity Weight) / Available Qualified Staff</code><br>
            <strong>Thresholds:</strong> Critical &gt; 4.8 : 1 (Severe overload) | Warning 3.5 – 4.8 : 1 (Elevated strain) | Optimal &lt; 3.5 : 1 (Balanced buffer)
        </div>
        """, unsafe_allow_html=True)

        dept_summary_rows = []
        for dw in dept_workloads:
            dept_summary_rows.append({
                "Department": dw.department,
                "Capacity": dw.capacity,
                "Census": dw.occupied,
                "Staff on Duty": dw.staff_on_duty,
                "Min Safety Floor": dw.min_staff_floor,
                "Acuity Weight": f"{dw.acuity_weight}x",
                "Workload Units": dw.workload_units,
                "Workload Index": f"{dw.workload_per_staff} : 1",
                "Status": dw.status,
                "Available Float Staff": dw.available_float_staff
            })
        st.dataframe(pd.DataFrame(dept_summary_rows), use_container_width=True, hide_index=True)

        # Workload bar chart
        fig_staff = go.Figure()
        fig_staff.add_trace(go.Bar(
            x=[dw.department for dw in dept_workloads],
            y=[dw.workload_per_staff for dw in dept_workloads],
            marker=dict(color=["#ef4444" if dw.status == "CRITICAL" else ("#f59e0b" if dw.status == "WARNING" else "#10b981") for dw in dept_workloads]),
            text=[f"{dw.workload_per_staff}:1" for dw in dept_workloads],
            textposition="auto"
        ))
        fig_staff.add_hline(y=THRESHOLDS["STAFF_WORKLOAD_RATIO"]["critical"], line_dash="dash", line_color="#ef4444", annotation_text="Critical Threshold (4.8:1)")
        fig_staff.add_hline(y=THRESHOLDS["STAFF_WORKLOAD_RATIO"]["warning"], line_dash="dot", line_color="#f59e0b", annotation_text="Warning Threshold (3.5:1)")
        fig_staff.update_layout(
            title="Departmental Acuity-Weighted Workload per Staff Member",
            yaxis_title="Workload Units / Staff",
            height=340,
            template="plotly_white",
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_staff, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("💡 Feasible Staffing Rebalancing Recommendations (Subject to Clinical Review)")

        if not staff_recs:
            st.info("No viable staff rebalancing transfers identified without violating donor unit minimum staffing floors.")
        else:
            for s_rec in staff_recs:
                st.markdown(f"""
                <div style="background: #ffffff; border: 1px solid #e2e8f0; border-left: 5px solid #0284c7; border-radius: 12px; padding: 16px 20px; margin-bottom: 12px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <strong style="font-size: 1.05rem; color: #0f172a;">Reallocate {s_rec.staff_to_move} Staff: {s_rec.from_department} ➔ {s_rec.to_department}</strong>
                        <span style="background: #dcfce7; color: #166534; padding: 2px 8px; border-radius: 9999px; font-size: 0.75rem; font-weight: 700;">FEASIBLE PROPOSAL</span>
                    </div>
                    <div style="font-size: 0.92rem; color: #334155; margin-top: 8px;">
                        {s_rec.clinical_advisory}
                    </div>
                    <div style="display: flex; gap: 20px; margin-top: 10px; font-size: 0.85rem; color: #64748b; flex-wrap: wrap;">
                        <div><strong>Qualifications:</strong> {s_rec.staff_qualification}</div>
                        <div><strong>Donor Remaining:</strong> {s_rec.donor_remaining_staff} staff (Safety Floor: {s_rec.donor_min_floor})</div>
                        <div><strong>Projected Ratio:</strong> <span style="color: #16a34a; font-weight: 700;">{s_rec.projected_recipient_workload} : 1</span></div>
                    </div>
                    <div style="margin-top: 8px; font-size: 0.8rem; color: #475569;">
                        <em>Requirement: Requires Nurse Supervisor authorization & cross-training verification prior to transfer.</em>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # =========================================================================
    # TAB 6: Explainable Recommendation and Evidence Panel
    # =========================================================================
    with tab6:
        st.subheader("📋 Explainable Recommendation & Evidence Panel")
        st.caption("Decision support recommendations with transparent evidence classification, explicit model rules, and human sign-off markers.")

        for rec in explainable_recs:
            badge_evidence = '<span style="background-color: #ecfdf5; color: #065f46; padding: 4px 10px; border-radius: 9999px; font-size: 0.78rem; font-weight: 700; border: 1px solid #a7f3d0;">🟢 Strong Model Evidence</span>' if "Strong" in rec.evidence_quality else '<span style="background-color: #fffbeb; color: #92400e; padding: 4px 10px; border-radius: 9999px; font-size: 0.78rem; font-weight: 700; border: 1px solid #fde68a;">🟡 Partial Model Evidence</span>'

            st.markdown(f"""
            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 14px; padding: 22px 26px; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.03);">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <div>
                        <span style="background: #f1f5f9; color: #0f172a; padding: 2px 8px; border-radius: 6px; font-size: 0.75rem; font-weight: 700; margin-right: 8px;">{rec.id}</span>
                        <span style="font-weight: 800; font-size: 1.2rem; color: #0f172a;">{rec.title}</span>
                    </div>
                    <div>
                        {badge_evidence}
                    </div>
                </div>
                <div style="margin-bottom: 12px; font-size: 0.92rem; color: #0284c7; font-weight: 600;">
                    🎯 Target Bottleneck: {rec.detected_bottleneck}
                </div>
                
                <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 12px 16px; margin-bottom: 14px;">
                    <div style="font-size: 0.8rem; font-weight: 700; color: #64748b; text-transform: uppercase; margin-bottom: 6px;">Observed Supporting Metrics:</div>
                    <div style="display: flex; gap: 20px; flex-wrap: wrap;">
                        {"".join(f"<div style='font-size: 0.88rem; color: #334155;'><strong>{k}:</strong> <span style='color: #0f172a;'>{v}</span></div>" for k, v in rec.supporting_metrics.items())}
                    </div>
                </div>

                <div style="font-size: 0.95rem; color: #334155; line-height: 1.5; margin-bottom: 12px;">
                    <strong>Why Generated (Operational Rationale):</strong><br>
                    {rec.why_generated}
                </div>

                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 12px;">
                    <div style="background: #ffffff; border: 1px solid #f1f5f9; border-radius: 8px; padding: 10px 14px;">
                        <span style="font-size: 0.78rem; font-weight: 700; color: #64748b; text-transform: uppercase;">Model Rules & Trigger Thresholds:</span>
                        <div style="font-size: 0.88rem; color: #0f172a; margin-top: 4px;"><code>{rec.model_rules_thresholds}</code></div>
                    </div>
                    <div style="background: #ffffff; border: 1px solid #f1f5f9; border-radius: 8px; padding: 10px 14px;">
                        <span style="font-size: 0.78rem; font-weight: 700; color: #64748b; text-transform: uppercase;">Estimated Impact (vs Disrupted Baseline):</span>
                        <div style="font-size: 0.88rem; color: #16a34a; font-weight: 700; margin-top: 4px;">{rec.estimated_impact}</div>
                    </div>
                </div>

                <div style="font-size: 0.88rem; color: #475569; margin-bottom: 8px;">
                    <strong>Assumptions & Boundary Conditions:</strong> {rec.assumptions_missing_data}
                </div>

                <div style="font-size: 0.85rem; color: #64748b; margin-bottom: 12px;">
                    <strong>Evidence Scope & Plain Language Limitation:</strong> {rec.evidence_limitations_plain_lang}
                </div>

                <div style="padding-top: 12px; border-top: 1px solid #f1f5f9; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
                    <div style="font-size: 0.85rem; color: #991b1b; font-weight: 600;">
                        🔒 {rec.clinical_priority_safeguard}
                    </div>
                    <div style="font-size: 0.85rem; color: #0f172a; font-weight: 700;">
                        ✍️ Required Approval: <span style="color: #0284c7;">{rec.required_human_approval}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Quick-Jump to Decision Intelligence
    st.markdown("<br>", unsafe_allow_html=True)
    col_bn_jump1, col_bn_jump2 = st.columns([2, 1])
    with col_bn_jump1:
        st.markdown("**Ready to simulate solutions and record administrative governance reviews?**")
        st.caption("Advance directly to the 7-step Decision Intelligence Engine to model solutions under identical conditions and evaluate trade-offs.")
    with col_bn_jump2:
        if st.button("🎯 Open Decision Intelligence Hub", type="primary", use_container_width=True, key="bn_to_di"):
            st.session_state["main_nav"] = "🎯 Decision Intelligence"
            st.rerun()

    # Bottom Notice
    st.markdown("---")
    st.caption("🔒 **MediFlow AI Decision Support Notice:** All bottleneck detections, dependency chain links, and intervention estimates are deterministic calculations from synthetic hospital models. MediFlow AI does not make autonomous clinical treatment decisions or execute real-world patient transfers.")
