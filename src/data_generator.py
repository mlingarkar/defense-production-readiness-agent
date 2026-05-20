import numpy as np
import pandas as pd


def generate_work_orders(num_orders: int = 350, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic defense production work order data.
    """
    np.random.seed(seed)

    programs = ["Falcon Shield", "Orion Sensor", "Atlas Comms", "Viper Ground", "Horizon ISR"]
    part_categories = [
        "Guidance Assembly",
        "Sensor Housing",
        "Power Module",
        "Comms Enclosure",
        "Ground Support Bracket",
        "Flight Control Subassembly",
    ]
    production_cells = ["Cell A", "Cell B", "Cell C", "Cell D", "Cell E"]
    statuses = ["In Progress", "Inspection", "Material Hold", "Rework", "Ready for Delivery"]
    priority_levels = ["Low", "Medium", "High", "Urgent"]

    today = pd.Timestamp.today().normalize()

    records = []

    for i in range(1, num_orders + 1):
        planned_start_offset = np.random.randint(-45, 10)
        planned_duration = np.random.randint(5, 35)
        planned_start = today + pd.Timedelta(days=planned_start_offset)
        planned_completion = planned_start + pd.Timedelta(days=planned_duration)

        schedule_variance = int(np.random.normal(loc=1, scale=7))
        actual_or_forecast_completion = planned_completion + pd.Timedelta(days=schedule_variance)

        material_available = np.random.choice([True, False], p=[0.76, 0.24])
        supplier_delay_days = 0 if material_available else np.random.randint(3, 25)

        inspection_hold = np.random.choice([True, False], p=[0.18, 0.82])
        quality_hold = np.random.choice([True, False], p=[0.14, 0.86])
        rework_hours = max(0, int(np.random.normal(loc=6 if quality_hold else 2, scale=5)))

        machine_availability_pct = np.clip(np.random.normal(loc=88, scale=12), 45, 100)
        labor_capacity_pct = np.clip(np.random.normal(loc=86, scale=14), 40, 100)

        defect_count = max(0, int(np.random.poisson(lam=2.1 if quality_hold else 0.8)))
        estimated_cost_impact = round(
            (supplier_delay_days * np.random.uniform(900, 1800))
            + (rework_hours * np.random.uniform(120, 260))
            + (defect_count * np.random.uniform(300, 1200)),
            2,
        )

        records.append(
            {
                "work_order_id": f"WO-{i:04d}",
                "program": np.random.choice(programs),
                "part_category": np.random.choice(part_categories),
                "production_cell": np.random.choice(production_cells),
                "priority_level": np.random.choice(priority_levels, p=[0.18, 0.36, 0.32, 0.14]),
                "status": np.random.choice(statuses, p=[0.42, 0.18, 0.14, 0.12, 0.14]),
                "planned_start_date": planned_start.date(),
                "planned_completion_date": planned_completion.date(),
                "forecast_completion_date": actual_or_forecast_completion.date(),
                "days_behind_schedule": max(0, schedule_variance),
                "material_available": material_available,
                "supplier_delay_days": supplier_delay_days,
                "inspection_hold": inspection_hold,
                "quality_hold": quality_hold,
                "rework_hours": rework_hours,
                "defect_count": defect_count,
                "machine_availability_pct": round(machine_availability_pct, 1),
                "labor_capacity_pct": round(labor_capacity_pct, 1),
                "estimated_cost_impact": estimated_cost_impact,
            }
        )

    return pd.DataFrame(records)


if __name__ == "__main__":
    df = generate_work_orders()
    df.to_csv("data/defense_production_work_orders.csv", index=False)
    print("Synthetic dataset created: data/defense_production_work_orders.csv")
