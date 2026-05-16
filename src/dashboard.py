from pathlib import Path

import pandas as pd
import streamlit as st

from agents import generate_readiness_brief
from ml_model import train_late_delivery_model
from risk_scoring import apply_risk_scoring


st.set_page_config(
    page_title="Defense Production Readiness Agent",
    page_icon="🛰️",
    layout="wide",
)


@st.cache_data
def load_data() -> pd.DataFrame:
    scored_path = Path("outputs/reports/scored_work_orders.csv")
    raw_path = Path("data/defense_production_work_orders.csv")

    if scored_path.exists():
        return pd.read_csv(scored_path)

    if raw_path.exists():
        raw_df = pd.read_csv(raw_path)
        scored_df = apply_risk_scoring(raw_df)
        modeled_df, _ = train_late_delivery_model(scored_df)
        return modeled_df

    st.error("No data file found. Run `python src/main.py` first.")
    return pd.DataFrame()


df = load_data()

st.title("Defense Production Readiness Agent")
st.caption(
    "Synthetic defense manufacturing analytics dashboard for delivery risk, "
    "ML late-delivery prediction, bottleneck identification, and agent-style "
    "operational recommendations."
)

if df.empty:
    st.stop()


with st.sidebar:
    st.header("Filters")

    selected_programs = st.multiselect(
        "Program",
        options=sorted(df["program"].unique()),
        default=sorted(df["program"].unique()),
    )

    selected_risks = st.multiselect(
        "Risk Category",
        options=["Low", "Medium", "High", "Critical"],
        default=["Low", "Medium", "High", "Critical"],
    )

    selected_cells = st.multiselect(
        "Production Cell",
        options=sorted(df["production_cell"].unique()),
        default=sorted(df["production_cell"].unique()),
    )


filtered_df = df[
    (df["program"].isin(selected_programs))
    & (df["risk_category"].isin(selected_risks))
    & (df["production_cell"].isin(selected_cells))
]


tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "Executive Summary",
        "Risk Overview",
        "ML Late-Delivery Prediction",
        "Bottlenecks",
        "Agent Recommendations",
        "Data Explorer",
    ]
)


with tab1:
    st.subheader("Executive Summary")

    total_orders = len(filtered_df)
    avg_risk = filtered_df["delivery_risk_score"].mean()
    high_or_critical = len(
        filtered_df[filtered_df["risk_category"].isin(["High", "Critical"])]
    )
    material_holds = len(filtered_df[filtered_df["material_available"] == False])
    quality_holds = len(filtered_df[filtered_df["quality_hold"] == True])
    total_cost_impact = filtered_df["estimated_cost_impact"].sum()

    predicted_late = (
        len(filtered_df[filtered_df["late_delivery_prediction"] == 1])
        if "late_delivery_prediction" in filtered_df.columns
        else 0
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Work Orders", f"{total_orders:,}")
    col2.metric("Average Risk Score", f"{avg_risk:.1f}")
    col3.metric("High/Critical Orders", f"{high_or_critical:,}")

    col4, col5, col6 = st.columns(3)
    col4.metric("Material Holds", f"{material_holds:,}")
    col5.metric("Predicted Late Orders", f"{predicted_late:,}")
    col6.metric("Estimated Cost Impact", f"${total_cost_impact:,.0f}")

    st.markdown("### Program Risk Summary")

    if "late_delivery_prediction" in filtered_df.columns:
        program_summary = (
            filtered_df.groupby("program")
            .agg(
                work_orders=("work_order_id", "count"),
                avg_risk_score=("delivery_risk_score", "mean"),
                predicted_late_orders=("late_delivery_prediction", "sum"),
                high_critical_orders=(
                    "risk_category",
                    lambda x: x.isin(["High", "Critical"]).sum(),
                ),
                total_cost_impact=("estimated_cost_impact", "sum"),
            )
            .reset_index()
            .sort_values("avg_risk_score", ascending=False)
        )
    else:
        program_summary = (
            filtered_df.groupby("program")
            .agg(
                work_orders=("work_order_id", "count"),
                avg_risk_score=("delivery_risk_score", "mean"),
                high_critical_orders=(
                    "risk_category",
                    lambda x: x.isin(["High", "Critical"]).sum(),
                ),
                total_cost_impact=("estimated_cost_impact", "sum"),
            )
            .reset_index()
            .sort_values("avg_risk_score", ascending=False)
        )

    st.dataframe(program_summary, width="stretch")


with tab2:
    st.subheader("Risk Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Risk Category Distribution")
        risk_counts = filtered_df["risk_category"].value_counts().reindex(
            ["Low", "Medium", "High", "Critical"]
        )
        st.bar_chart(risk_counts)

    with col2:
        st.markdown("### Average Risk by Program")
        program_risk = (
            filtered_df.groupby("program")["delivery_risk_score"]
            .mean()
            .sort_values(ascending=False)
        )
        st.bar_chart(program_risk)

    st.markdown("### Highest-Risk Work Orders")

    top_risk = filtered_df.sort_values(
        "delivery_risk_score",
        ascending=False,
    ).head(15)

    st.dataframe(
        top_risk[
            [
                "work_order_id",
                "program",
                "part_category",
                "production_cell",
                "priority_level",
                "status",
                "delivery_risk_score",
                "risk_category",
                "primary_risk_driver",
                "days_behind_schedule",
                "supplier_delay_days",
                "rework_hours",
            ]
        ],
        width="stretch",
    )


with tab3:
    st.subheader("ML Late-Delivery Prediction")

    if "late_delivery_probability" not in filtered_df.columns:
        st.warning(
            "Late-delivery prediction columns were not found. "
            "Run `python src/main.py` to regenerate the project outputs."
        )
    else:
        avg_probability = filtered_df["late_delivery_probability"].mean()
        predicted_late_count = len(
            filtered_df[filtered_df["late_delivery_prediction"] == 1]
        )
        actual_late_count = len(
            filtered_df[filtered_df["late_delivery_flag"] == 1]
        )

        col1, col2, col3 = st.columns(3)
        col1.metric("Average Late Probability", f"{avg_probability:.1%}")
        col2.metric("Predicted Late Orders", f"{predicted_late_count:,}")
        col3.metric("Actual Synthetic Late Orders", f"{actual_late_count:,}")

        st.markdown("### Highest ML-Predicted Late Delivery Risk")

        ml_top = filtered_df.sort_values(
            "late_delivery_probability",
            ascending=False,
        ).head(20)

        st.dataframe(
            ml_top[
                [
                    "work_order_id",
                    "program",
                    "part_category",
                    "production_cell",
                    "priority_level",
                    "status",
                    "late_delivery_probability",
                    "late_delivery_prediction",
                    "delivery_risk_score",
                    "risk_category",
                    "primary_risk_driver",
                    "supplier_delay_days",
                    "rework_hours",
                    "machine_availability_pct",
                    "labor_capacity_pct",
                ]
            ],
            width="stretch",
        )

        st.markdown("### Average Late-Delivery Probability by Program")

        late_by_program = (
            filtered_df.groupby("program")["late_delivery_probability"]
            .mean()
            .sort_values(ascending=False)
        )

        st.bar_chart(late_by_program)

        report_path = Path("outputs/reports/late_delivery_model_report.md")

        if report_path.exists():
            st.markdown("### Model Report")
            st.markdown(report_path.read_text(encoding="utf-8"))
        else:
            st.info(
                "Model report not found yet. Run `python src/main.py` to create "
                "`outputs/reports/late_delivery_model_report.md`."
            )


with tab4:
    st.subheader("Bottleneck Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Primary Risk Drivers")
        driver_counts = filtered_df["primary_risk_driver"].value_counts()
        st.bar_chart(driver_counts)

    with col2:
        st.markdown("### Average Risk by Production Cell")
        cell_risk = (
            filtered_df.groupby("production_cell")["delivery_risk_score"]
            .mean()
            .sort_values(ascending=False)
        )
        st.bar_chart(cell_risk)

    st.markdown("### Bottleneck Detail")

    if "late_delivery_probability" in filtered_df.columns:
        bottleneck_detail = (
            filtered_df.groupby(["primary_risk_driver", "production_cell"])
            .agg(
                work_orders=("work_order_id", "count"),
                avg_risk_score=("delivery_risk_score", "mean"),
                avg_late_probability=("late_delivery_probability", "mean"),
                avg_days_behind=("days_behind_schedule", "mean"),
                avg_supplier_delay=("supplier_delay_days", "mean"),
                avg_rework_hours=("rework_hours", "mean"),
            )
            .reset_index()
            .sort_values("avg_risk_score", ascending=False)
        )
    else:
        bottleneck_detail = (
            filtered_df.groupby(["primary_risk_driver", "production_cell"])
            .agg(
                work_orders=("work_order_id", "count"),
                avg_risk_score=("delivery_risk_score", "mean"),
                avg_days_behind=("days_behind_schedule", "mean"),
                avg_supplier_delay=("supplier_delay_days", "mean"),
                avg_rework_hours=("rework_hours", "mean"),
            )
            .reset_index()
            .sort_values("avg_risk_score", ascending=False)
        )

    st.dataframe(bottleneck_detail, width="stretch")


with tab5:
    st.subheader("Agent Recommendations")

    brief = generate_readiness_brief(filtered_df)
    st.markdown(brief)


with tab6:
    st.subheader("Data Explorer")

    st.dataframe(filtered_df, width="stretch")

    csv = filtered_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Filtered Data",
        data=csv,
        file_name="filtered_defense_production_readiness.csv",
        mime="text/csv",
    )