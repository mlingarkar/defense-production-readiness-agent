import pandas as pd


PRIORITY_WEIGHTS = {
    "Low": 0,
    "Medium": 5,
    "High": 11,
    "Urgent": 18,
}


def score_work_order(row: pd.Series) -> float:
    """
    Calculate delivery readiness risk using operational risk drivers.

    Higher values indicate greater risk of schedule, quality, or delivery impact.
    """
    score = 0

    score += min(row["days_behind_schedule"] * 3.2, 32)
    score += min(row["supplier_delay_days"] * 2.4, 28)

    if not row["material_available"]:
        score += 12
    if row["inspection_hold"]:
        score += 8
    if row["quality_hold"]:
        score += 12

    score += min(row["rework_hours"] * 1.4, 16)
    score += min(row["defect_count"] * 2.8, 14)

    if row["machine_availability_pct"] < 75:
        score += 10
    elif row["machine_availability_pct"] < 85:
        score += 5

    if row["labor_capacity_pct"] < 75:
        score += 9
    elif row["labor_capacity_pct"] < 85:
        score += 4

    score += PRIORITY_WEIGHTS.get(row["priority_level"], 0)

    return round(min(score, 100), 2)


def classify_risk(score: float) -> str:
    if score >= 75:
        return "Critical"
    if score >= 55:
        return "High"
    if score >= 30:
        return "Medium"
    return "Low"


def identify_primary_driver(row: pd.Series) -> str:
    drivers = {
        "Schedule Delay": row["days_behind_schedule"] * 3.2,
        "Supplier / Material Delay": row["supplier_delay_days"] * 2.4 + (12 if not row["material_available"] else 0),
        "Quality / Rework": row["rework_hours"] * 1.4 + row["defect_count"] * 2.8 + (12 if row["quality_hold"] else 0),
        "Inspection Hold": 8 if row["inspection_hold"] else 0,
        "Machine Capacity": 10 if row["machine_availability_pct"] < 75 else 5 if row["machine_availability_pct"] < 85 else 0,
        "Labor Capacity": 9 if row["labor_capacity_pct"] < 75 else 4 if row["labor_capacity_pct"] < 85 else 0,
        "Priority Pressure": PRIORITY_WEIGHTS.get(row["priority_level"], 0),
    }

    return max(drivers, key=drivers.get)


def apply_risk_scoring(df: pd.DataFrame) -> pd.DataFrame:
    scored_df = df.copy()
    scored_df["delivery_risk_score"] = scored_df.apply(score_work_order, axis=1)
    scored_df["risk_category"] = scored_df["delivery_risk_score"].apply(classify_risk)
    scored_df["primary_risk_driver"] = scored_df.apply(identify_primary_driver, axis=1)
    return scored_df
