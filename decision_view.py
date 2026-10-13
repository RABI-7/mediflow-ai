"""
=============================================================================
 MediFlow AI — Integrated Decision Intelligence View
=============================================================================
 Module: decision_view.py
 Purpose: Phase 5 primary decision intelligence interface:
   Visualizes the complete 7-step decision-support workflow:
   1. Monitor Hospital State & Census Telemetry
   2. Detect Operational Bottlenecks, Hidden Capacity & Dependency Chains
   3. Forecast Future Patient Arrival Risks
   4. Generate Feasible Operational Response Options
   5. Simulate & Compare Scenarios Under Identical Baselines
   6. Analyze Trade-Offs, Constraints & Objective Ranking
   7. Interactive Human Review Governance & Clinical Safeguards
=============================================================================
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from typing import Dict, Any, Optional

from decision_engine import (
    run_integrated_decision_pipeline,
    validate_hospital_state,
    ReviewStatus,
    HumanReviewRecord,
    IntegratedDecisionPipelineResult
)
from forecast_engine import (
    generate_historical_arrival_dataset,
    generate_patient_arrival_forecast,
    ForecastResult
)


def render_decision_intelligence(
    summary: Dict[str, Any],
    df_departments: pd.DataFrame,
    df_resources: pd.DataFrame
):
    """
    Renders the complete Phase 5 Integrated Decision Intelligence Engine interface.
    """
    # -------------------------------------------------------------------------
    # Brand Header
    # -------------------------------------------------------------------------
    # Clean Professional Header
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <div style="font-size: 1.5rem; font-weight: 800; color: #0f172a; letter-spacing: -0.02em;">
            Decision Intelligence & Response Evaluation
        </div>
        <div style="font-size: 0.88rem; color: #475569; margin-top: 2px;">
            Closed-loop operational decision support: identify bottlenecks, evaluate candidate interventions under identical baselines, analyze trade-offs, and record clinical governance sign-offs.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # Operational Scenario Configuration Controls
    # -------------------------------------------------------------------------
    st.subheader("⚙️ Operational Context & Scenario Parameters")
    with st.container(border=True):
        col_c1, col_c2, col_c3, col_c4 = st.columns(4)

        with col_c1:
            use_forecast_surge = st.checkbox("🔮 Sync with Forecast Model", value=True, help="Auto-calculate surge arrivals from forward forecasting engine")
            if use_forecast_surge:
                # Load forecast
                forecast_horizon = st.selectbox("Projection Window", [4, 8, 12, 24], index=2, format_func=lambda h: f"{h} Hours", key="di_horizon")
                df_hist = generate_historical_arrival_dataset(start_date="2026-08-18", num_days=56, seed=42)
                active_forecast = generate_patient_arrival_forecast(df_hist, horizon_hours=forecast_horizon)
                additional_arrivals = active_forecast.incremental_surge_above_baseline
                st.caption(f"Forecast: **+{additional_arrivals} surge arrivals** (Total: {active_forecast.total_predicted_arrivals})")
            else:
                additional_arrivals = st.slider("Additional Patient Arrivals", 0, 60, 30, 5, key="di_surge_manual")
                forecast_horizon = 12
                active_forecast = None

        with col_c2:
            ct_status_choice = st.selectbox(
                "CT Scanner Status",
                ["Unavailable (1 Scanner Down - 50% Outage)", "Available (Both Units Operational)"],
                index=0,
                key="di_ct"
            )

        with col_c3:
            discharge_delay_choice = st.selectbox(
                "Inpatient Discharge Delay",
                ["On (Bed Clearances & Handoffs Delayed)", "Off (Standard Discharge Flow)"],
                index=0,
                key="di_disch"
            )

        with col_c4:
            duration_choice = forecast_horizon if use_forecast_surge else st.selectbox("Simulation Duration", [4, 8, 12, 24], index=2, format_func=lambda h: f"{h} Hours", key="di_dur")

        col_b_l, col_b_r = st.columns([1, 3])
        with col_b_l:
            run_pipeline_btn = st.button("🚀 Execute Decision Pipeline", type="primary", use_container_width=True)
        with col_b_r:
            st.caption("Executes the end-to-end 7-step optimization pipeline: detects bottlenecks, simulates solutions, evaluates trade-offs, and updates the human review audit trail.")

    ct_status = "Unavailable" if "Unavailable" in ct_status_choice else "Available"
    discharge_delay = True if "On" in discharge_delay_choice else False
    duration_hours = int(duration_choice)

    # -------------------------------------------------------------------------
    # Run Decision Pipeline
    # -------------------------------------------------------------------------
    # Retrieve human review session state
    if "human_reviews" not in st.session_state:
        st.session_state["human_reviews"] = {}

    pipeline_result = run_integrated_decision_pipeline(
        summary=summary,
        df_departments=df_departments,
        df_resources=df_resources,
        additional_arrivals=additional_arrivals,
        ct_scanner_status=ct_status,
        discharge_delay=discharge_delay,
        duration_hours=duration_hours,
        forecast=active_forecast,
        existing_reviews=st.session_state["human_reviews"]
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # Pipeline View: Focused Operational Workspaces
    # -------------------------------------------------------------------------
    tab_risks, tab_step45, tab_step6, tab_step7 = st.tabs([
        "🚨 Operational Bottlenecks & Demand Risks",
        "⚡ Candidate Responses & Simulation",
        "⚖️ Trade-Offs & Objective Ranking",
        "🛡️ Clinical Governance & Sign-Off"
    ])

    # =========================================================================
    # TAB 1: Bottlenecks, Hidden Capacity & Forward Risks
    # =========================================================================
    with tab_risks:
        st.subheader("Operational Bottlenecks, Constraints & Demand Projections")
        st.caption("Algorithmic constraint detection identifying acute bottlenecks, non-ready bed blockers, process chains, and forward arrival demand.")

        col_bn_l, col_bn_r = st.columns([1, 1])
        with col_bn_l:
            st.markdown("##### 🚨 Active Operational Bottlenecks Detected")
            for bn in pipeline_result.bottlenecks:
                sev_color = "#ef4444" if bn.severity == "CRITICAL" else "#f59e0b"
                st.markdown(f"""
                <div style="background: #ffffff; border-left: 4px solid {sev_color}; border: 1px solid #e2e8f0; border-radius: 8px; padding: 10px 14px; margin-bottom: 10px;">
                    <div style="font-weight: 700; color: #0f172a; font-size: 0.92rem;">
                        {bn.title} <span style="font-size: 0.75rem; background: #fee2e2; color: #991b1b; padding: 2px 6px; border-radius: 4px;">{bn.severity}</span>
                    </div>
                    <div style="font-size: 0.82rem; color: #475569; margin-top: 2px;">
                        <strong>Indicator:</strong> {bn.metric_label} = <strong>{bn.metric_value}</strong> ({bn.threshold_rule})
                    </div>
                    <div style="font-size: 0.78rem; color: #64748b; margin-top: 2px;">
                        <em>{bn.evidence}</em>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        with col_bn_r:
            # Hidden Capacity Recovery Summary
            hc = pipeline_result.hidden_capacity
            st.markdown("##### 🛏️ Transitional Bed Capacity Blockers")
            st.markdown(f"""
            <div style="background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 10px; padding: 14px 16px;">
                <div style="font-weight: 700; color: #0f172a; margin-bottom: 6px;">
                    Nominal Unallocated Beds: <strong>{hc.theoretical_unoccupied}</strong> | Genuinely Ready: <strong>{hc.ready_available_beds}</strong>
                </div>
                <div style="font-size: 0.85rem; color: #334155; margin-bottom: 8px;">
                    {hc.explanation}
                </div>
                <div style="font-size: 0.82rem; color: #0369a1; background: #e0f2fe; padding: 8px 12px; border-radius: 6px;">
                    ⚡ <strong>Recoverable Buffer:</strong> {hc.recoverable_beds_total} beds blocked in turnover states can be unlocked via EVS & Discharge coordination without adding licensed capacity.
                </div>
            </div>
            """, unsafe_allow_html=True)

            with st.expander("📋 View Ordered Bed Recovery Protocols", expanded=False):
                for act in hc.ordered_actions:
                    st.markdown(f"**Priority {act.priority}: {act.title}** ({act.target_state})")
                    st.markdown(f"- Currently Blocked: {act.current_blocked_beds} beds | Estimated Recoverable: **+{act.recoverable_beds_estimated} beds**")
                    st.markdown(f"- Blocker: {act.primary_blocker}")
                    st.markdown(f"- Protocol: {act.action_protocol}")
                    st.markdown(f"- Unlock Timeline: `{act.estimated_unlock_time}` | Status: `{act.feasibility_status}`")
                    st.markdown("---")

        # Bottleneck Chain Detector
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("##### 🔗 Modeled Bottleneck Dependency Chain")
        bc = pipeline_result.bottleneck_chains
        st.markdown(f"""
        <div style="background: #fff7ed; border-left: 5px solid #f97316; border-radius: 8px; padding: 14px 18px; margin-bottom: 12px;">
            <div style="font-weight: 700; color: #9a3412; font-size: 0.95rem;">
                Initiating Bottleneck: {bc.initiating_bottleneck}
            </div>
            <div style="font-size: 0.86rem; color: #7c2d12; margin-top: 4px; white-space: pre-line;">
                {bc.propagation_summary}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # =========================================================================
    # TAB 2: Solutions & Multi-Scenario Simulation
    # =========================================================================
    with tab_step45:
        st.subheader("Candidate Operational Responses & Comparative Simulation")
        st.caption("Context-aware response options simulated deterministically under identical initial conditions (280 licensed beds, 238 occupied, 34 waiting, 6 CT queue) and duration.")

        # Candidate Solutions Overview
        st.markdown("##### Proposed Feasible Response Options")
        col_sol_cards = st.columns(min(3, max(1, len(pipeline_result.available_solutions))))
        for idx, sol in enumerate([s for s in pipeline_result.available_solutions if s.feasibility.is_feasible]):
            target_col = col_sol_cards[idx % len(col_sol_cards)]
            with target_col:
                st.markdown(f"""
                <div class="sol-card sol-feasible-border">
                    <div style="margin-bottom: 4px;">
                        <span class="badge-category">{sol.category}</span>
                        <span class="badge-feasible">🟢 FEASIBLE</span>
                    </div>
                    <div style="font-weight: 700; color: #0f172a; font-size: 0.93rem; margin-bottom: 4px;">
                        {sol.title}
                    </div>
                    <div style="font-size: 0.82rem; color: #334155; margin-bottom: 6px;">
                        {sol.description}
                    </div>
                    <div style="font-size: 0.78rem; color: #64748b;">
                        <strong>Target:</strong> {sol.target_bottleneck}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Multi-Scenario Impact Comparison Table
        st.markdown("##### 📋 Scenario Impact Comparison Table")
        st.caption("Side-by-side metric comparison showing absolute values, deltas (Δ), and percentage changes (%) vs Disrupted.")

        comp_dict = {
            "Metric / Indicator": [
                "Occupied Beds",
                "Bed Occupancy Rate (%)",
                "Available Beds",
                "Patients Waiting for Bed (Unmet Demand)",
                "CT Scanner Queue (Patients)",
                "Estimated CT Wait Time (Minutes)",
                "Total Patient Discharges",
                "Operational Feasibility Status"
            ],
            "Baseline (Normal)": [
                f"{pipeline_result.sim_baseline.final_occupied} / {pipeline_result.sim_baseline.total_beds}",
                f"{pipeline_result.sim_baseline.final_occupancy_pct}%",
                f"{pipeline_result.sim_baseline.final_available}",
                f"{pipeline_result.sim_baseline.final_waiting}",
                f"{pipeline_result.sim_baseline.final_ct_queue}",
                f"{pipeline_result.sim_baseline.est_ct_wait_mins} min",
                f"{pipeline_result.sim_baseline.total_discharges}",
                "Standard Operations"
            ],
            "Disrupted (Active)": [
                f"{pipeline_result.sim_disrupted.final_occupied} / {pipeline_result.sim_disrupted.total_beds}",
                f"{pipeline_result.sim_disrupted.final_occupancy_pct}%",
                f"{pipeline_result.sim_disrupted.final_available}",
                f"{pipeline_result.sim_disrupted.final_waiting}",
                f"{pipeline_result.sim_disrupted.final_ct_queue}",
                f"{pipeline_result.sim_disrupted.est_ct_wait_mins} min",
                f"{pipeline_result.sim_disrupted.total_discharges}",
                "Active Disruption"
            ]
        }

        for ev in pipeline_result.ranked_evaluations:
            if ev.is_feasible and ev.sim_result:
                sim = ev.sim_result
                imp = ev.impact
                short_title = ev.solution.title.split(":")[0] if ":" in ev.solution.title else ev.solution.title
                comp_dict[short_title] = [
                    f"{sim.final_occupied} / {sim.total_beds} ({imp.delta_occupied:+d}, {imp.pct_occupied:+.1f}%)",
                    f"{sim.final_occupancy_pct}%",
                    f"{sim.final_available} ({imp.delta_available:+d}, {imp.pct_available:+.1f}%)",
                    f"{sim.final_waiting} ({imp.delta_waiting:+d}, {imp.pct_waiting:+.1f}%)",
                    f"{sim.final_ct_queue} ({imp.delta_ct_queue:+d}, {imp.pct_ct_queue:+.1f}%)",
                    f"{sim.est_ct_wait_mins} min ({imp.delta_ct_wait:+d} min, {imp.pct_ct_wait:+.1f}%)",
                    f"{sim.total_discharges} ({imp.delta_discharges:+d}, {imp.pct_discharges:+.1f}%)",
                    f"FEASIBLE (Rank #{ev.rank})"
                ]

        df_comparison = pd.DataFrame(comp_dict)
        st.dataframe(df_comparison, use_container_width=True, hide_index=True)

        # Plotly Comparison Charts
        st.markdown("<br>", unsafe_allow_html=True)
        col_c_l, col_c_r = st.columns(2)
        with col_c_l:
            fig_q = go.Figure()
            fig_q.add_trace(go.Scatter(
                x=pipeline_result.sim_baseline.hourly_df["Hour"],
                y=pipeline_result.sim_baseline.hourly_df["Patients Waiting"],
                mode="lines+markers",
                name="Baseline (Normal)",
                line=dict(color="#2563eb", width=2)
            ))
            fig_q.add_trace(go.Scatter(
                x=pipeline_result.sim_disrupted.hourly_df["Hour"],
                y=pipeline_result.sim_disrupted.hourly_df["Patients Waiting"],
                mode="lines+markers",
                name="Disrupted",
                line=dict(color="#ef4444", width=2.5)
            ))
            colors = ["#10b981", "#8b5cf6", "#06b6d4"]
            for idx, ev in enumerate(pipeline_result.ranked_evaluations[:3]):
                if ev.is_feasible and ev.sim_result:
                    short_t = ev.solution.title.split(":")[0] if ":" in ev.solution.title else ev.solution.title
                    fig_q.add_trace(go.Scatter(
                        x=ev.sim_result.hourly_df["Hour"],
                        y=ev.sim_result.hourly_df["Patients Waiting"],
                        mode="lines+markers",
                        name=short_t,
                        line=dict(color=colors[idx % len(colors)], width=2, dash="dot")
                    ))
            fig_q.update_layout(
                title="Waiting Queue Trajectory over Simulation Hours",
                xaxis_title="Simulation Hours",
                yaxis_title="Patients Waiting",
                template="plotly_white",
                height=320,
                margin=dict(l=20, r=20, t=35, b=20)
            )
            st.plotly_chart(fig_q, use_container_width=True)

        with col_c_r:
            scens = ["Baseline", "Disrupted"]
            waits = [pipeline_result.sim_baseline.est_ct_wait_mins, pipeline_result.sim_disrupted.est_ct_wait_mins]
            bar_colors = ["#2563eb", "#ef4444"]
            for idx, ev in enumerate(pipeline_result.ranked_evaluations[:3]):
                if ev.is_feasible and ev.sim_result:
                    short_t = ev.solution.title.split(":")[0] if ":" in ev.solution.title else ev.solution.title
                    scens.append(short_t)
                    waits.append(ev.sim_result.est_ct_wait_mins)
                    bar_colors.append(colors[idx % len(colors)])

            fig_ct = go.Figure()
            fig_ct.add_trace(go.Bar(
                x=scens,
                y=waits,
                marker_color=bar_colors,
                text=[f"{w}m" for w in waits],
                textposition="auto"
            ))
            fig_ct.update_layout(
                title="Estimated CT Scanner Wait Time (Minutes)",
                xaxis_title="Scenario",
                yaxis_title="Minutes",
                template="plotly_white",
                height=320,
                margin=dict(l=20, r=20, t=35, b=20)
            )
            st.plotly_chart(fig_ct, use_container_width=True)

    # =========================================================================
    # TAB 6: Trade-Offs & Best Option
    # =========================================================================
    with tab_step6:
        st.subheader("Step 6: Trade-Off Analysis, Constraints & Objective Ranking")
        st.caption("Transparent evaluation of operational trade-offs, resource consumption, and criteria-driven solution ranking.")

        # Best Option Banner
        best = pipeline_result.best_feasible_option
        if best:
            imp = best.impact
            st.markdown(f"""
            <div class="best-option-banner">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; margin-bottom: 6px;">
                    <div style="font-weight: 800; font-size: 1.15rem; color: #065f46;">
                        🏆 Recommended Best Feasible Option: {best.solution.title}
                    </div>
                    <div>
                        <span class="badge-feasible">RANK #1 FEASIBLE</span>
                        <span style="background: #dcfce7; color: #166534; font-size: 0.75rem; font-weight: 700; padding: 2px 8px; border-radius: 9999px; border: 1px solid #86efac;">Score: {best.objective_score}</span>
                    </div>
                </div>
                <div style="font-size: 0.92rem; color: #1e293b; margin-bottom: 8px;">
                    <strong>Selection Rationale:</strong> {best.selection_rationale}
                </div>
                <div style="display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 10px; font-size: 0.88rem; color: #0f172a;">
                    <span>• <strong>Queue Relief:</strong> <span style="color: #16a34a; font-weight: 700;">{imp.delta_waiting} ({imp.pct_waiting:.1f}%)</span></span>
                    <span>• <strong>Beds Restored:</strong> <span style="color: #16a34a; font-weight: 700;">+{imp.delta_available} ({imp.pct_available:.1f}%)</span></span>
                    <span>• <strong>CT Wait:</strong> <span style="color: #16a34a; font-weight: 700;">{imp.delta_ct_wait} min ({imp.pct_ct_wait:.1f}%)</span></span>
                    <span>• <strong>Discharges:</strong> <span style="color: #16a34a; font-weight: 700;">+{imp.delta_discharges}</span></span>
                </div>
                <div style="font-size: 0.82rem; color: #047857; background: #ffffff; padding: 8px 12px; border-radius: 6px; border: 1px solid #bbf7d0;">
                    🔒 <strong>Advisory Protocol Notice:</strong> Ranked via multi-criteria operational objective: 3.0×Queue + 2.0×Available Beds + 0.25×CT Wait + 1.5×Discharges. Requires authorized administrative sign-off before implementation.
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Trade-Off Matrix
        st.markdown("##### ⚖️ Operational Trade-Off & Resource Impact Matrix")
        st.caption("Operational interventions improve flow metrics but allocate finite resources. Review trade-offs before authorizing changes:")

        for to in pipeline_result.trade_offs:
            st.markdown(f"""
            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 14px 18px; margin-bottom: 12px;">
                <div style="font-weight: 700; color: #0f172a; font-size: 0.95rem; margin-bottom: 4px;">
                    {to.solution_title}
                </div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; font-size: 0.85rem; margin-bottom: 6px;">
                    <div>
                        <strong style="color: #16a34a;">🟢 Primary Gain:</strong> {to.primary_gain}<br>
                        <strong style="color: #16a34a;">🟢 Secondary Gain:</strong> {to.secondary_gain}
                    </div>
                    <div>
                        <strong style="color: #dc2626;">🔴 Resources Committed:</strong> {to.resources_consumed}<br>
                        <strong style="color: #b45309;">⚠️ Potential Operational Friction:</strong> {to.potential_trade_off}
                    </div>
                </div>
                <div style="font-size: 0.8rem; color: #475569; background: #f8fafc; padding: 6px 10px; border-radius: 6px;">
                    💡 <strong>Mitigation Protocol:</strong> {to.mitigation_strategy}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # =========================================================================
    # TAB 7: Human Review Governance & Clinical Safeguards
    # =========================================================================
    with tab_step7:
        st.subheader("Step 7: Human Review Governance & Clinical Safeguards")
        st.caption("Advisory decision governance: authorized hospital supervisors review, document considerations, and approve or reject proposed responses.")

        st.markdown("""
        <div style="background: #f0fdf4; border: 1px solid #bbf7d0; border-left: 5px solid #16a34a; padding: 12px 16px; border-radius: 8px; margin-bottom: 16px;">
            <div style="font-weight: 700; color: #166534; font-size: 0.92rem;">
                🛡️ Operational Decision Governance Safeguard Active
            </div>
            <div style="font-size: 0.83rem; color: #14532d; margin-top: 2px;">
                MediFlow AI operates strictly as an advisory decision-support prototype. It does not automatically transfer patients, modify clinical care, discharge inpatients, or alter electronic health records. All recommendations require formal review and authorization by certified healthcare personnel.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("##### Active Solution Review Roster & Administrative Sign-Off")

        # Interactive Human Review Panel for each simulated solution
        for ev in pipeline_result.ranked_evaluations:
            sid = ev.solution.id
            rec = pipeline_result.human_reviews.get(sid)
            if not rec:
                continue

            with st.container(border=True):
                col_r_title, col_r_stat = st.columns([3, 1])
                with col_r_title:
                    st.markdown(f"**{ev.solution.title}**")
                    st.caption(f"Required Approver: `{ev.solution.required_human_approval}` | Acuity Safeguard: *{ev.solution.clinical_priority_safeguard}*")
                with col_r_stat:
                    # Status Badge
                    stat_color = {
                        ReviewStatus.PENDING_REVIEW: "#f59e0b",
                        ReviewStatus.APPROVED_FOR_CONSIDERATION: "#10b981",
                        ReviewStatus.REJECTED: "#ef4444",
                        ReviewStatus.NEEDS_MORE_INFORMATION: "#64748b"
                    }.get(rec.status, "#64748b")
                    st.markdown(f"""
                    <div style="text-align: right;">
                        <span style="background: {stat_color}22; color: {stat_color}; font-weight: 700; font-size: 0.8rem; padding: 4px 10px; border-radius: 9999px; border: 1px solid {stat_color}55;">
                            {rec.status}
                        </span>
                    </div>
                    """, unsafe_allow_html=True)

                col_in1, col_in2, col_in3 = st.columns([1.5, 1.5, 3])
                with col_in1:
                    new_status = st.selectbox(
                        "Decision Status",
                        options=ReviewStatus.ALL_STATUSES,
                        index=ReviewStatus.ALL_STATUSES.index(rec.status) if rec.status in ReviewStatus.ALL_STATUSES else 0,
                        key=f"rev_stat_{sid}"
                    )
                with col_in2:
                    new_role = st.selectbox(
                        "Reviewer Role",
                        options=[
                            "Chief Medical Officer (CMO)",
                            "Chief Nursing Officer (CNO)",
                            "Bed Placement Coordinator",
                            "EVS Operations Supervisor",
                            "Radiology Medical Director",
                            "Emergency Department Chair"
                        ],
                        key=f"rev_role_{sid}"
                    )
                with col_in3:
                    new_notes = st.text_input(
                        "Clinical / Operational Rationale Notes",
                        value=rec.review_notes,
                        key=f"rev_notes_{sid}",
                        placeholder="Enter clinical rationale, staffing constraints, or approval stipulations..."
                    )

                col_btn_save, col_info_ts = st.columns([1, 4])
                with col_btn_save:
                    if st.button("💾 Record Decision", key=f"btn_save_{sid}"):
                        rec.status = new_status
                        rec.reviewer_role = new_role
                        rec.review_notes = new_notes
                        rec.updated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
                        st.session_state["human_reviews"][sid] = rec
                        st.success(f"Updated status for {ev.solution.title} to '{new_status}'")
                        st.rerun()
                with col_info_ts:
                    st.caption(f"Last updated: **{rec.updated_at}** by **{rec.reviewer_role}**")

    # Bottom Notice
    st.markdown("---")
    st.caption("🔒 **MediFlow AI Data Notice:** All values presented in the Integrated Decision Intelligence Engine are synthetic demonstration estimates for hackathon evaluation. MediFlow AI does not store, transmit, or process real protected health information (PHI).")
