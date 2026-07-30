"""
Streamlit Frontend for Insurance Enrollment Outreach Assistant Agent.

A refined, user-facing UI dashboard built on top of the Outreach Assistant tools:
- predict_employee_enrollment (Prediction Tool)
- rank_employees & lookup_region_profile (Ranking & Lookup Tool)
- explain_prediction (Explanation Tool)
- check_guardrails (Refusal & Safety Guardrails)
- engineer_features (Data Quality Check)

Run command: streamlit run src/app.py
"""

import os
import sys
import random
import pandas as pd
import streamlit as st
from typing import Dict, Any, List, Optional

# Configure project root and src directory in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
for path in [PROJECT_ROOT, SRC_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

# Import agent tools and feature engineering pipeline
from src.tools.predict_tool import predict_employee_enrollment
from src.tools.rank_tool import rank_employees, lookup_region_profile
from src.tools.explain_tool import explain_prediction
from src.tools.refusal_guardrails import check_guardrails
from src.feature_engineering import engineer_features


@st.cache_data
def load_region_list() -> List[str]:
    """Cache and load available region names from region profiles CSV."""
    csv_path = os.path.join(PROJECT_ROOT, "data", "region_benefit_profiles.csv")
    if os.path.exists(csv_path):
        df_reg = pd.read_csv(csv_path)
        return sorted(df_reg["region"].unique().tolist())
    return ["Midwest", "Northeast", "South", "West"]


@st.cache_data
def load_raw_employee_ids() -> List[int]:
    """Cache and load unique employee IDs from raw employee CSV for selection."""
    csv_path = os.path.join(PROJECT_ROOT, "data", "employees_raw.csv")
    if os.path.exists(csv_path):
        df_emp = pd.read_csv(csv_path)
        df_emp = df_emp.drop_duplicates(subset=["employee_id"], keep=False)
        return sorted(df_emp["employee_id"].unique().tolist())
    return []


@st.cache_data
def load_engineered_data() -> pd.DataFrame:
    """Cache and return engineered dataset with quality flags."""
    df_eng, _ = engineer_features()
    return df_eng


def main() -> None:
    """Main Streamlit application entrypoint."""
    st.set_page_config(
        page_title="Insurance Enrollment Assistant",
        page_icon="🛡️",
        layout="wide"
    )

    # Custom CSS for increased typography size & enhanced readability across all tables, headers, and inputs
    st.markdown("""
    <style>
    /* 1. Tab Bar Navigation Labels */
    button[data-baseweb="tab"] p {
        font-size: 1.22rem !important;
        font-weight: 700 !important;
    }
    button[data-baseweb="tab"] {
        padding-top: 0.7rem !important;
        padding-bottom: 0.7rem !important;
    }

    /* 2. Subtitle & Header Captions */
    div[data-testid="stCaptionContainer"] p, .stCaption {
        font-size: 1.1rem !important;
        line-height: 1.6 !important;
        color: #CBD5E1 !important;
    }

    /* 3. Form Input Labels & Widget Labels */
    div[data-widget-label="true"] p, label p {
        font-size: 1.12rem !important;
        font-weight: 700 !important;
        color: #F8FAFC !important;
    }

    /* 4. Table Column Headers & Feature Names */
    div[data-testid="stTable"] th, 
    div[data-testid="stTable"] th p, 
    th, th p, 
    div[role="columnheader"] {
        font-size: 1.18rem !important;
        font-weight: 700 !important;
        color: #60A5FA !important;
    }

    /* 5. Table Cell Data & Feature Value Text */
    div[data-testid="stTable"] td, 
    div[data-testid="stTable"] td p, 
    td, td p, 
    div[role="gridcell"] {
        font-size: 1.1rem !important;
        line-height: 1.6 !important;
        color: #F8FAFC !important;
    }

    /* 6. Metric Card Labels & Values */
    div[data-testid="stMetricLabel"] p, label[data-testid="stMetricLabel"] {
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        color: #94A3B8 !important;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        color: #F8FAFC !important;
    }

    /* 7. Alert & Info Banner Text */
    div[data-testid="stAlert"] p {
        font-size: 1.1rem !important;
        line-height: 1.6 !important;
    }

    /* 8. Button Labels */
    button[kind="primary"] p, button[kind="secondary"] p {
        font-size: 1.12rem !important;
        font-weight: 700 !important;
    }

    /* 9. Section Subheaders */
    .stMarkdown h2, .stMarkdown h3 {
        font-size: 1.45rem !important;
        font-weight: 700 !important;
        color: #F8FAFC !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("🛡️ Insurance Enrollment Outreach Assistant")
    st.caption("A thin UI dashboard wrapping the CLI Insurance Enrollment Agent tools and machine learning inference pipeline.")

    tabs = st.tabs([
        "📊 Rank Outreach Candidates",
        "🔮 Predict & Explain",
        "🌐 Region Profile",
        "🛑 Refusal Demo",
        "🔍 Data Quality Check"
    ])

    regions = load_region_list()

    # -------------------------------------------------------------------------
    # TAB 1: Rank Outreach Candidates
    # -------------------------------------------------------------------------
    with tabs[0]:
        st.header("Rank Outreach Candidates")
        st.caption("Prioritize employee outreach candidates based on predicted enrollment probability and regional capacity constraints.")

        col1, col2 = st.columns(2)
        with col1:
            selected_region = st.selectbox("Select Region", options=regions, index=0)
        with col2:
            k_input = st.number_input(
                "Candidates to Return (K) [0 = Default to HR Capacity]",
                min_value=0,
                max_value=5000,
                value=0,
                step=5
            )

        # Contextual HR capacity metric display
        try:
            profile_res = lookup_region_profile(region=selected_region)
            if profile_res.get("status") == "success" and "data" in profile_res:
                capacity = profile_res["data"].get("hr_outreach_capacity", "N/A")
                st.info(f"📍 **{selected_region} Region Context**: Regional HR Outreach Capacity limit is **{capacity}** candidates.")
        except Exception as e:
            st.warning(f"Could not load region context: {e}")

        if st.button("Rank Candidates", type="primary", key="btn_rank"):
            try:
                top_k_val = int(k_input) if k_input > 0 else 1000
                apply_cap = True if k_input == 0 else False

                rank_res = rank_employees(
                    top_k=top_k_val,
                    region=selected_region,
                    apply_hr_capacity_limit=apply_cap
                )

                if rank_res.get("status") == "success":
                    rankings = rank_res.get("rankings", [])
                    total_el = rank_res.get("total_eligible", 0)
                    ret_cnt = rank_res.get("returned_count", 0)

                    st.success(f"Successfully ranked **{ret_cnt}** candidates out of **{total_el}** eligible employees in {selected_region}.")
                    
                    df_rank = pd.DataFrame(rankings)
                    if not df_rank.empty:
                        df_rank["plan_tier"] = df_rank["plan_tier_requested_clean"] if "plan_tier_requested_clean" in df_rank.columns else df_rank.get("plan_tier_requested", "Unknown")
                        df_rank["contact_channel"] = df_rank["last_contact_channel_clean"] if "last_contact_channel_clean" in df_rank.columns else df_rank.get("last_contact_channel", "Unknown")
                        
                        # Prepare clean display dataframe
                        display_df = pd.DataFrame({
                            "Rank": df_rank["rank"],
                            "Employee ID": df_rank["employee_id"],
                            "Salary": df_rank["salary"].apply(lambda s: f"${s:,.2f}" if pd.notnull(s) else "N/A"),
                            "Employment Type": df_rank["employment_type"],
                            "Requested Plan Tier": df_rank["plan_tier"],
                            "Contact Channel": df_rank["contact_channel"],
                            "Enrollment Probability": df_rank["enrollment_probability"],
                            "Probability Bar": df_rank["enrollment_probability"]
                        })

                        # Display formatted dataframe with 4 decimal places and a progress bar column
                        st.dataframe(
                            display_df,
                            column_config={
                                "Enrollment Probability": st.column_config.NumberColumn(
                                    label="Enrollment Probability",
                                    format="%.4f",
                                    help="Predicted probability of enrollment (0.0000 - 1.0000)"
                                ),
                                "Probability Bar": st.column_config.ProgressColumn(
                                    label="Probability Visual",
                                    min_value=0.0,
                                    max_value=1.0,
                                    format="%.4f"
                                )
                            },
                            use_container_width=True,
                            hide_index=True
                        )

                        with st.expander("Show raw response JSON", expanded=False):
                            st.json(rank_res)
                    else:
                        st.warning("No candidates found matching the selected criteria.")
                else:
                    st.error(f"Ranking tool returned an error: {rank_res.get('message')}")
            except Exception as e:
                st.error(f"An unexpected error occurred during ranking: {str(e)}")

    # -------------------------------------------------------------------------
    # TAB 2: Predict & Explain
    # -------------------------------------------------------------------------
    with tabs[1]:
        st.header("Predict & Explain Employee Enrollment")
        st.caption("Generate model predictions and plain-language driver explanations for individual employees.")

        st.info("💡 **Dataset Reference**: Employee IDs are 5-digit numbers ranging from **10001 to 20000** (e.g., `17825`, `12324`, `10017`, `10527`).")

        emp_ids_sample = load_raw_employee_ids()
        col_id1, col_id2 = st.columns([2, 1])
        with col_id1:
            emp_input = st.text_input("Enter Employee ID", value="17825", help="Enter a 5-digit numeric employee ID (10001 - 20000)")
        with col_id2:
            if emp_ids_sample:
                quick_id = st.selectbox("Or Select Sample ID", options=["Custom Input"] + [str(i) for i in emp_ids_sample[:10]], index=0)
                if quick_id != "Custom Input":
                    emp_input = quick_id

        if st.button("Predict & Explain", type="primary", key="btn_predict"):
            if not emp_input.strip():
                st.error("Please enter a valid Employee ID.")
            else:
                try:
                    emp_id_val = int(emp_input.strip())
                except ValueError:
                    st.error("Invalid Employee ID format. Must be a 5-digit numeric integer.")
                    emp_id_val = None

                if emp_id_val is not None:
                    # 1. Prediction Tool Call
                    try:
                        pred_res = predict_employee_enrollment(employee_id=emp_id_val)
                        if pred_res.get("status") == "success" and pred_res.get("predictions"):
                            pred_info = pred_res["predictions"][0]
                            prob = pred_info.get("enrollment_probability", 0.0)
                            cls = pred_info.get("predicted_enrolled", 0)

                            st.subheader("Prediction Overview")
                            c1, c2, c3 = st.columns(3)
                            c1.metric("Employee ID", emp_id_val)
                            c2.metric("Enrollment Probability", f"{prob:.4f}")
                            c3.metric("Predicted Outcome", "Likely Enrolled" if cls == 1 else "Unlikely Enrolled")

                            # 2. Explanation Tool Call
                            expl_res = explain_prediction(employee_id=emp_id_val)
                            if expl_res.get("status") == "success":
                                summary = expl_res.get("narrative_summary", "")
                                drivers = expl_res.get("top_drivers", [])

                                st.subheader("Model Explanation")
                                if cls == 1:
                                    st.success(summary)
                                else:
                                    st.info(summary)

                                if drivers:
                                    st.markdown("**Key Model Drivers:**")
                                    drivers_df = pd.DataFrame(drivers)
                                    drivers_df = drivers_df.rename(columns={
                                        "feature": "Feature",
                                        "effect": "Effect Direction",
                                        "reasoning": "Driver Reasoning"
                                    })
                                    st.dataframe(drivers_df[["Feature", "Effect Direction", "Driver Reasoning"]], use_container_width=True, hide_index=True)

                                with st.expander("Show raw response JSON", expanded=False):
                                    st.json({"prediction": pred_res, "explanation": expl_res})
                            else:
                                st.error(f"Explanation error: {expl_res.get('message')}")
                        else:
                            st.error(f"Employee ID **{emp_id_val}** not found in database.")
                    except Exception as e:
                        st.error(f"Prediction failed: {str(e)}")

    # -------------------------------------------------------------------------
    # TAB 3: Region Profile
    # -------------------------------------------------------------------------
    with tabs[2]:
        st.header("Region Profile Lookup")
        st.caption("Inspect macro regional benefit statistics, premium costs, and outreach capacity metrics.")

        selected_reg_lookup = st.selectbox("Select Region", options=regions, key="lookup_reg_select")

        if st.button("Lookup Profile", type="primary", key="btn_lookup"):
            try:
                profile_res = lookup_region_profile(region=selected_reg_lookup)
                if profile_res.get("status") == "success":
                    data = profile_res.get("data", {})
                    st.subheader(f"Region Details: {selected_reg_lookup}")
                    
                    # Formatted metrics tiles
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Total Employees", f"{data.get('n_employees_region', 0):,}")
                    m2.metric("Hist. Enrollment Rate", f"{data.get('hist_enrollment_rate_region', 0):.1%}")
                    m3.metric("Avg Premium Cost", f"${data.get('avg_premium_cost_usd', 0):,}/mo")
                    m4.metric("HR Capacity Limit", data.get("hr_outreach_capacity", "N/A"))

                    # Structured clean table output
                    profile_df = pd.DataFrame([{
                        "Attribute": k.replace("_", " ").title(),
                        "Value": str(v)
                    } for k, v in data.items()])
                    
                    st.table(profile_df)

                    with st.expander("Show raw response JSON", expanded=False):
                        st.json(data)
                else:
                    st.error(f"Profile lookup failed: {profile_res.get('message')}")
            except Exception as e:
                st.error(f"Error looking up region profile: {str(e)}")

    # -------------------------------------------------------------------------
    # TAB 4: Refusal Demo
    # -------------------------------------------------------------------------
    with tabs[3]:
        st.header("Safety Guardrails & Refusal Audit")
        st.caption("Demonstrate target leakage prevention, Fair AI non-discrimination enforcement, and medical safety guardrails.")

        st.subheader("1. Target Leakage Prevention Test")
        st.write("Simulates an explicit request for the excluded `legacy_propensity_score` feature:")

        if st.button("Try requesting legacy_propensity_score directly", key="btn_refusal"):
            try:
                refusal_res = explain_prediction(employee_id=10017, feature_requested="legacy_propensity_score")
                if refusal_res.get("status") == "refusal":
                    st.warning(refusal_res.get("message"))
                    with st.expander("Show raw response JSON", expanded=False):
                        st.json(refusal_res)
                else:
                    st.error(f"Expected refusal message, received: {refusal_res}")
            except Exception as e:
                st.error(f"Error during refusal test: {str(e)}")

        st.divider()

        st.subheader("2. Fair AI Non-Discrimination Audit")
        st.markdown(
            "> **Compliance Policy**: Model explanations strictly exclude protected demographic attributes "
            "(`gender`, `marital_status`, `age`) from narrative reasoning."
        )

        st.write("Example: Explanation for Employee **17825** demonstrating protected attributes are excluded from key drivers:")
        try:
            demo_expl = explain_prediction(employee_id=17825)
            if demo_expl.get("status") == "success":
                st.info(demo_expl.get("narrative_summary", ""))
                drivers = demo_expl.get("top_drivers", [])
                if drivers:
                    drivers_df = pd.DataFrame(drivers).rename(columns={
                        "feature": "Feature",
                        "effect": "Effect Direction",
                        "reasoning": "Driver Reasoning"
                    })
                    st.table(drivers_df[["Feature", "Effect Direction", "Driver Reasoning"]])
                
                with st.expander("Show raw response JSON", expanded=False):
                    st.json(demo_expl)
        except Exception as e:
            st.error(f"Could not load example explanation: {str(e)}")

    # -------------------------------------------------------------------------
    # TAB 5: Data Quality Check
    # -------------------------------------------------------------------------
    with tabs[4]:
        st.header("Data Quality Check")
        st.caption("Inspect raw data entry anomalies, missing values, and post-processing quality flags side by side.")

        df_eng = load_engineered_data()
        
        # Identify rows with at least one quality flag = 1 or Unknown category
        messy_df = df_eng[
            (df_eng["has_application_date"] == 0) |
            (df_eng["contact_after_app"] == 1) |
            (df_eng["tenure_inconsistent"] == 1) |
            (df_eng["no_prior_record"] == 1) |
            (df_eng["last_contact_channel_clean"] == "Unknown") |
            (df_eng["plan_tier_requested_clean"] == "Unknown")
        ]
        messy_ids = messy_df["employee_id"].tolist()

        if "dq_emp_id" not in st.session_state:
            st.session_state["dq_emp_id"] = 17825

        col_q1, col_q2 = st.columns([2, 1])

        with col_q2:
            st.write(" ")
            st.write(" ")
            if st.button("🎲 Random Messy Row", key="btn_random_messy"):
                if messy_ids:
                    st.session_state["dq_emp_id"] = random.choice(messy_ids)

        with col_q1:
            dq_input = st.text_input(
                "Enter Employee ID for Quality Inspection",
                value=str(st.session_state["dq_emp_id"]),
                help="Select an employee ID to inspect raw vs cleaned data quality attributes"
            )

        try:
            target_id = int(dq_input.strip())
        except ValueError:
            st.error("Invalid Employee ID format. Please enter a 5-digit numeric integer.")
            target_id = None

        if target_id is not None:
            match_rows = df_eng[df_eng["employee_id"] == target_id]
            if match_rows.empty:
                st.error(f"Employee ID **{target_id}** not found in database.")
            else:
                row = match_rows.iloc[0]
                
                # Fetch raw record directly for unparsed raw string values
                csv_path = os.path.join(PROJECT_ROOT, "data", "employees_raw.csv")
                raw_df = pd.read_csv(csv_path)
                raw_row_match = raw_df[raw_df["employee_id"] == target_id]
                raw_row = raw_row_match.iloc[0] if not raw_row_match.empty else row

                st.subheader(f"Side-by-Side Quality Comparison for Employee {target_id}")

                # Build clean two-column comparison table
                comparison_data = [
                    {
                        "Attribute / Field": "Application Date",
                        "Raw Input Value": str(raw_row.get("application_date", "NaN")),
                        "Cleaned / Processed Value": "Missing (NaN)" if row["has_application_date"] == 0 else str(row.get("app_date", "Parsed")),
                        "Quality Flag Status": "❌ Missing (`has_application_date=0`)" if row["has_application_date"] == 0 else "✅ Valid Date Recorded"
                    },
                    {
                        "Attribute / Field": "Last Contact Date",
                        "Raw Input Value": str(raw_row.get("last_contact_date", "NaN")),
                        "Cleaned / Processed Value": f"Days to App: {row['days_contact_to_app']:.1f}" if pd.notnull(row['days_contact_to_app']) else "NaN",
                        "Quality Flag Status": "⚠️ Contact Post-Application (`contact_after_app=1`)" if row["contact_after_app"] == 1 else "✅ Contact Pre-Application"
                    },
                    {
                        "Attribute / Field": "Last Contact Channel",
                        "Raw Input Value": str(raw_row.get("last_contact_channel", "NaN")),
                        "Cleaned / Processed Value": str(row["last_contact_channel_clean"]),
                        "Quality Flag Status": "ℹ️ Imputed to 'Unknown'" if row["last_contact_channel_clean"] == "Unknown" else "✅ Standardized Category"
                    },
                    {
                        "Attribute / Field": "Plan Tier Requested",
                        "Raw Input Value": str(raw_row.get("plan_tier_requested", "NaN")),
                        "Cleaned / Processed Value": str(row["plan_tier_requested_clean"]),
                        "Quality Flag Status": "ℹ️ Imputed to 'Unknown'" if row["plan_tier_requested_clean"] == "Unknown" else "✅ Standardized Tier"
                    },
                    {
                        "Attribute / Field": "Prior Year Enrolled Status",
                        "Raw Input Value": str(raw_row.get("prior_year_enrolled", "NaN")),
                        "Cleaned / Processed Value": f"no_prior={row['no_prior_record']}, clean={row['prior_year_enrolled_clean']}",
                        "Quality Flag Status": "ℹ️ New Hire Sentinel (-1)" if row["no_prior_record"] == 1 else "✅ Historical Status (0/1)"
                    },
                    {
                        "Attribute / Field": "Tenure vs Age Check",
                        "Raw Input Value": f"Tenure: {raw_row.get('tenure_years')}, Age: {raw_row.get('age')}",
                        "Cleaned / Processed Value": f"Tenure Years: {row['tenure_years']}, Age: {row['age']}",
                        "Quality Flag Status": "⚠️ Inconsistent (`tenure_inconsistent=1`)" if row["tenure_inconsistent"] == 1 else "✅ Tenure Consistent with Age"
                    }
                ]

                comp_df = pd.DataFrame(comparison_data)
                st.table(comp_df)

                # Plain-Language Detected Issues List
                st.subheader(f"Detected Issues for Employee {target_id}")

                detected_issues = []

                if row["has_application_date"] == 0:
                    detected_issues.append("Application date missing")

                if row["contact_after_app"] == 1:
                    detected_issues.append("Contact recorded after application date")

                if row["tenure_inconsistent"] == 1:
                    detected_issues.append("Tenure inconsistent with age")

                if row["no_prior_record"] == 1:
                    detected_issues.append("prior_year_enrolled sentinel: new hire, no prior record")

                raw_chan_val = str(raw_row.get("last_contact_channel", "")).strip().lower()
                if pd.isnull(raw_row.get("last_contact_channel")) or raw_chan_val in ["nan", "none", ""]:
                    detected_issues.append("Missing last_contact_channel (imputed to 'Unknown')")

                raw_tier_val = str(raw_row.get("plan_tier_requested", "")).strip().lower()
                if pd.isnull(raw_row.get("plan_tier_requested")) or raw_tier_val in ["nan", "none", ""]:
                    detected_issues.append("Missing plan_tier_requested (imputed to 'Unknown')")

                if detected_issues:
                    for issue in detected_issues:
                        st.markdown(f"- ⚠️ {issue}")
                else:
                    st.success("✅ No data quality anomalies detected for this selected employee record.")


if __name__ == "__main__":
    main()
