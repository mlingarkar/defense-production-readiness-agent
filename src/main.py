from pathlib import Path

from agents import generate_readiness_brief
from data_generator import generate_work_orders
from ml_model import train_late_delivery_model, write_model_report
from risk_scoring import apply_risk_scoring
from visualization import create_figures


def main() -> None:
    data_dir = Path("data")
    reports_dir = Path("outputs/reports")
    models_dir = Path("outputs/models")

    data_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    raw_df = generate_work_orders(num_orders=350, seed=42)
    raw_df.to_csv(data_dir / "defense_production_work_orders.csv", index=False)

    scored_df = apply_risk_scoring(raw_df)

    modeled_df, model_metrics = train_late_delivery_model(scored_df)
    modeled_df.to_csv(reports_dir / "scored_work_orders.csv", index=False)

    write_model_report(model_metrics)

    create_figures(modeled_df)

    brief = generate_readiness_brief(modeled_df)
    with open(
        reports_dir / "production_readiness_brief.md",
        "w",
        encoding="utf-8",
    ) as file:
        file.write(brief)

    print("Project outputs created successfully.")
    print("Dataset: data/defense_production_work_orders.csv")
    print("Scored and modeled data: outputs/reports/scored_work_orders.csv")
    print("Readiness brief: outputs/reports/production_readiness_brief.md")
    print("ML model report: outputs/reports/late_delivery_model_report.md")
    print("Saved model: outputs/models/late_delivery_model.joblib")
    print("Figures: outputs/figures/")


if __name__ == "__main__":
    main()