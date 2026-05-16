import pandas as pd


def production_readiness_agent(df: pd.DataFrame) -> list[str]:
    high_risk = df[df["risk_category"].isin(["High", "Critical"])]
    recommendations = []

    if high_risk.empty:
        return [
            "No high-risk work orders were detected. Continue monitoring schedule, material, quality, labor, and machine availability indicators."
        ]

    top_orders = high_risk.sort_values("delivery_risk_score", ascending=False).head(5)

    for _, row in top_orders.iterrows():
        action = (
            f"{row['work_order_id']} for {row['program']} is categorized as "
            f"{row['risk_category']} risk with a delivery risk score of "
            f"{row['delivery_risk_score']}. Primary driver: {row['primary_risk_driver']}. "
        )

        if row["primary_risk_driver"] == "Supplier / Material Delay":
            action += (
                "Recommended action: review supplier status, escalate material recovery planning, "
                "and prioritize expediting options for schedule-critical orders."
            )
        elif row["primary_risk_driver"] == "Quality / Rework":
            action += (
                "Recommended action: prioritize quality review, allocate rework capacity, "
                "and confirm inspection requirements before downstream release."
            )
        elif row["primary_risk_driver"] == "Schedule Delay":
            action += (
                "Recommended action: review production sequencing, rebalance workload, "
                "and evaluate schedule recovery options."
            )
        elif row["primary_risk_driver"] == "Inspection Hold":
            action += (
                "Recommended action: coordinate inspection resources, resolve hold documentation, "
                "and confirm release criteria before production handoff."
            )
        elif row["primary_risk_driver"] == "Machine Capacity":
            action += (
                "Recommended action: review machine utilization, maintenance status, "
                "and potential alternate routing through available production cells."
            )
        elif row["primary_risk_driver"] == "Labor Capacity":
            action += (
                "Recommended action: review staffing coverage, cross-trained operator availability, "
                "and short-term labor allocation."
            )
        else:
            action += (
                "Recommended action: escalate for program-level review due to elevated priority pressure."
            )

        recommendations.append(action)

    return recommendations


def material_risk_agent(df: pd.DataFrame) -> str:
    material_risk = df[
        (df["material_available"] == False) | (df["supplier_delay_days"] > 0)
    ]

    if material_risk.empty:
        return (
            "Material risk is currently low. No supplier or material delays are present "
            "in the simulated work order set."
        )

    affected_programs = material_risk["program"].nunique()
    avg_delay = material_risk["supplier_delay_days"].mean()

    return (
        f"Material risk is affecting {len(material_risk)} work orders across "
        f"{affected_programs} programs. The average supplier delay among affected orders "
        f"is {avg_delay:.1f} days. Recommended action: review supplier recovery plans, "
        "prioritize urgent or high-risk work orders, and evaluate expediting options where appropriate."
    )


def quality_hold_agent(df: pd.DataFrame) -> str:
    quality_risk = df[
        (df["quality_hold"] == True)
        | (df["inspection_hold"] == True)
        | (df["rework_hours"] > 8)
    ]

    if quality_risk.empty:
        return (
            "Quality and inspection risk is currently low. No major hold or rework pattern "
            "is present in the filtered work order set."
        )

    avg_rework = quality_risk["rework_hours"].mean()
    top_cell = quality_risk["production_cell"].value_counts().idxmax()

    return (
        f"Quality or inspection risk is present in {len(quality_risk)} work orders. "
        f"Average rework hours among affected orders are {avg_rework:.1f}. "
        f"The most frequently affected production cell is {top_cell}. "
        "Recommended action: prioritize inspection release, confirm rework staffing, "
        "and review recurring quality patterns by production cell."
    )


def program_briefing_agent(df: pd.DataFrame) -> str:
    total_orders = len(df)
    critical_orders = len(df[df["risk_category"] == "Critical"])
    high_orders = len(df[df["risk_category"] == "High"])
    avg_score = df["delivery_risk_score"].mean()

    top_program = (
        df.groupby("program")["delivery_risk_score"]
        .mean()
        .sort_values(ascending=False)
        .index[0]
    )

    top_driver = df["primary_risk_driver"].value_counts().idxmax()

    return (
        f"Production readiness review completed for {total_orders} active work orders. "
        f"The current average delivery risk score is {avg_score:.1f}. "
        f"There are {critical_orders} critical-risk and {high_orders} high-risk work orders. "
        f"The program with the highest average risk is {top_program}. "
        f"The most common primary risk driver is {top_driver}. "
        "Operational attention should focus on high-risk delivery items, supplier recovery, "
        "inspection release, quality holds, and rework capacity."
    )


def generate_readiness_brief(df: pd.DataFrame) -> str:
    recommendations = production_readiness_agent(df)

    brief = "# Production Readiness Brief\n\n"

    brief += "## Executive Summary\n\n"
    brief += program_briefing_agent(df) + "\n\n"

    brief += "## Material Risk Agent\n\n"
    brief += material_risk_agent(df) + "\n\n"

    brief += "## Quality Hold Agent\n\n"
    brief += quality_hold_agent(df) + "\n\n"

    brief += "## Production Readiness Agent Recommendations\n\n"
    for item in recommendations:
        brief += f"- {item}\n"

    brief += "\n## Project Disclaimer\n\n"
    brief += (
        "This readiness brief is generated from synthetic portfolio data and is intended "
        "for educational and portfolio demonstration purposes only. The analysis and "
        "recommendations are simulated and should not be used for real production, quality, "
        "engineering, safety, or defense decision-making."
    )

    return brief