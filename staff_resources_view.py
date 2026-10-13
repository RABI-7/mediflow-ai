"""
=============================================================================
 MediFlow AI — Staff & Resource Intelligence
=============================================================================
 Dedicated workspace for monitoring department staffing ratios, cross-unit
 float qualification rules, and diagnostic equipment operational health.
=============================================================================
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, Any

from bottleneck_engine import evaluate_staff_workload_imbalance


def render_staff_resources(
    summary: Dict[str, Any],
    df_departments: pd.DataFrame,
    df_resources: pd.DataFrame
):
    """
    Renders the dedicated Staff & Resource Intelligence workspace.
    """
    # -------------------------------------------------------------------------
    # 1. Page Header
    # -------------------------------------------------------------------------
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <div style="font-size: 1.5rem; font-weight: 800; color: #0f172a; letter-spacing: -0.02em;">
            Staff & Resource Intelligence
        </div>
        <div style="font-size: 0.88rem; color: #475569; margin-top: 2px;">
            Departmental staffing coverage, acuity-adjusted workload balances, qualification-governed floating, and diagnostic equipment status.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Calculate workload analysis
    workloads, rebalance_recs = evaluate_staff_workload_imbalance(df_departments, summary)

    # Key Staffing Metrics Row
    tot_staff = summary.get("staff_on_duty", 64)
    avg_ratio = round(sum(w.workload_to_staff_ratio for w in workloads) / max(1, len(workloads)), 2)
    strained_depts = [w for w in workloads if w.status == "CRITICAL" or w.status == "HIGH_WORKLOAD"]
    balanced_depts = [w for w in workloads if w.status == "NORMAL" or w.status == "BALANCED"]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(
            label="Total Staff on Duty",
            value=f"{tot_staff} Clinicians",
            delta=summary.get("shift_name", "Day Shift (08:00 - 16:00)"),
            delta_color="off"
        )
    with c2:
        st.metric(
            label="Average Workload Ratio",
            value=f"{avg_ratio} : 1",
            delta="Acuity Units / Nurse",
            delta_color="off"
        )
    with c3:
        st.metric(
            label="High-Strain Departments",
            value=f"{len(strained_depts)} Units",
            delta="General Med & Surgery",
            delta_color="inverse"
        )
    with c4:
        st.metric(
            label="Diagnostic Units Operational",
            value="4 of 5 Active",
            delta="CT Scanner 2 Degraded",
            delta_color="inverse"
        )

    # -------------------------------------------------------------------------
    # 2. Departmental Staffing Workload Analysis
    # -------------------------------------------------------------------------
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Departmental Staffing & Acuity Workload")

    workload_rows = []
    for w in workloads:
        workload_rows.append({
            "Department": w.department,
            "Active Staff": w.staff_count,
            "Min Safety Floor": w.min_staff_floor,
            "Inpatient Occupancy": f"{w.occupied_beds} / {w.capacity_beds}",
            "Acuity Multiplier": f"{w.acuity_multiplier:.1f}x",
            "Workload Units": f"{w.workload_units:.1f}",
            "Workload Ratio": f"{w.workload_to_staff_ratio:.2f} : 1",
            "Staffing Health": w.status
        })
    df_wl = pd.DataFrame(workload_rows)
    st.dataframe(df_wl, use_container_width=True, hide_index=True)

    # -------------------------------------------------------------------------
    # 3. Two Columns: Qualification Float Rules vs Diagnostic Equipment Health
    # -------------------------------------------------------------------------
    col_l, col_r = st.columns([1.1, 0.9])

    with col_l:
        with st.container(border=True):
            st.markdown("""
            <div style="font-weight: 700; color: #0f172a; font-size: 0.98rem; margin-bottom: 2px;">
                Authorized Staff Rebalancing & Cross-Coverage Rules
            </div>
            <div style="font-size: 0.82rem; color: #64748b; margin-bottom: 12px;">
                Clinical float eligibility strictly enforced by department competencies and statutory nurse safety floors
            </div>
            """, unsafe_allow_html=True)

            if not rebalance_recs:
                st.info("Department workloads are currently balanced within acceptable operational limits.")
            else:
                for rec in rebalance_recs:
                    st.markdown(f"""
                    <div style="background: #ffffff; border: 1px solid #e2e8f0; border-left: 4px solid #0284c7; border-radius: 8px; padding: 12px 14px; margin-bottom: 10px;">
                        <div style="font-weight: 700; color: #0f172a; font-size: 0.92rem;">
                            Transfer {rec.nurses_to_rebalance} Staff: {rec.from_department} ➔ {rec.to_department}
                        </div>
                        <div style="font-size: 0.84rem; color: #334155; margin-top: 4px;">
                            <strong>Acuity Impact:</strong> {rec.rationale}
                        </div>
                        <div style="font-size: 0.82rem; color: #0284c7; margin-top: 3px;">
                            <strong>Qualification Guardrail:</strong> {rec.qualification_match}
                        </div>
                        <div style="font-size: 0.78rem; color: #059669; margin-top: 4px;">
                            🛡️ <strong>Donor Floor Protection:</strong> {rec.donor_floor_safety}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

    with col_r:
        with st.container(border=True):
            st.markdown("""
            <div style="font-weight: 700; color: #0f172a; font-size: 0.98rem; margin-bottom: 2px;">
                Critical Diagnostic Equipment Health
            </div>
            <div style="font-size: 0.82rem; color: #64748b; margin-bottom: 12px;">
                Real-time queue lengths and estimated procedural wait times for hospital shared diagnostics
            </div>
            """, unsafe_allow_html=True)

            for _, row in df_resources.iterrows():
                is_degraded = "Degraded" in str(row["Availability Status"]) or "High" in str(row["Operational Health"])
                badge_bg = "#fee2e2" if is_degraded else "#ecfdf5"
                badge_fg = "#991b1b" if is_degraded else "#065f46"
                border_color = "#ef4444" if is_degraded else "#10b981"

                st.markdown(f"""
                <div style="background: #ffffff; border: 1px solid #e2e8f0; border-left: 4px solid {border_color}; border-radius: 8px; padding: 10px 14px; margin-bottom: 8px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <strong style="color: #0f172a; font-size: 0.9rem;">{row['Resource / Unit']}</strong>
                        <span style="background: {badge_bg}; color: {badge_fg}; font-size: 0.75rem; font-weight: 700; padding: 2px 8px; border-radius: 9999px;">
                            {row['Operational Health']}
                        </span>
                    </div>
                    <div style="font-size: 0.82rem; color: #475569; margin-top: 3px; display: flex; justify-content: space-between;">
                        <span>Queue: <strong>{row['Queue Length']}</strong></span>
                        <span>Est. Wait: <strong>{row['Est. Wait Time']}</strong></span>
                        <span>Status: <strong>{row['Availability Status']}</strong></span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # Statutory Floor Notice
    st.markdown("""
    <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-left: 4px solid #0284c7; padding: 10px 14px; border-radius: 6px; margin-top: 10px; font-size: 0.82rem; color: #334155;">
        🔒 <strong>Staffing Governance Rule:</strong> Clinical staffing rebalances require charge nurse sign-off. Pediatric and Critical Care nurses with specialized credentials cannot be reassigned without maintaining statutory department safety floors.
    </div>
    """, unsafe_allow_html=True)
