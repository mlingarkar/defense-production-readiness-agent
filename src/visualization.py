from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def create_figures(df: pd.DataFrame, output_dir: str = "outputs/figures") -> None:
    """
    Create saved PNG visualizations for the production readiness project.

    These figures are generated from the scored and modeled work order dataset.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Risk category distribution
    risk_order = ["Low", "Medium", "High", "Critical"]
    risk_counts = df["risk_category"].value_counts().reindex(risk_order).fillna(0)

    plt.figure(figsize=(8, 5))
    risk_counts.plot(kind="bar")
    plt.title("Delivery Risk Category Distribution")
    plt.xlabel("Risk Category")
    plt.ylabel("Number of Work Orders")
    plt.tight_layout()
    plt.savefig(output_path / "risk_distribution.png")
    plt.close()

    # Primary bottleneck / risk drivers
    driver_counts = df["primary_risk_driver"].value_counts()

    plt.figure(figsize=(10, 5))
    driver_counts.plot(kind="bar")
    plt.title("Primary Bottleneck / Risk Drivers")
    plt.xlabel("Primary Risk Driver")
    plt.ylabel("Number of Work Orders")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(output_path / "bottleneck_drivers.png")
    plt.close()

    # Average delivery risk score by program
    program_summary = (
        df.groupby("program")["delivery_risk_score"]
        .mean()
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(9, 5))
    program_summary.plot(kind="bar")
    plt.title("Average Delivery Risk Score by Program")
    plt.xlabel("Program")
    plt.ylabel("Average Delivery Risk Score")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(output_path / "program_risk_summary.png")
    plt.close()

    # Average ML-predicted late-delivery probability by program
    if "late_delivery_probability" in df.columns:
        late_program_summary = (
            df.groupby("program")["late_delivery_probability"]
            .mean()
            .sort_values(ascending=False)
        )

        plt.figure(figsize=(9, 5))
        late_program_summary.plot(kind="bar")
        plt.title("Average ML-Predicted Late Delivery Probability by Program")
        plt.xlabel("Program")
        plt.ylabel("Average Late Delivery Probability")
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        plt.savefig(output_path / "late_delivery_probability_by_program.png")
        plt.close()