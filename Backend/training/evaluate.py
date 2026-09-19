import joblib
import numpy as np

from app.config import MODEL_PATH
from training.dataset_loader import load_dataset
from training.train import train_model

EXPERIMENT_NAMES = {"raw": "Sin limpieza", "clean": "Con limpieza"}


def run_experiment(clean: bool) -> tuple[dict, object]:
    name = "clean" if clean else "raw"
    print(f"\n=== Experiment: {EXPERIMENT_NAMES[name]} ===")
    X, y = load_dataset(clean=clean)
    print(f"Loaded: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes")

    model, metrics = train_model(X, y)
    print(f"Accuracy: {metrics['accuracy_mean']:.4f} | "
          f"Precision: {metrics['precision_mean']:.4f} | "
          f"Recall: {metrics['recall_mean']:.4f} | "
          f"F1: {metrics['f1_mean']:.4f}")

    return metrics, model


def compare_experiments(results_raw: dict, results_clean: dict) -> str:
    print("\n=== Comparison: Raw vs Clean ===")
    print(f"{'Metric':<14}{'Sin limpieza':>14}{'Con limpieza':>14}{'Diff':>12}")
    print("-" * 54)
    for key in ["accuracy_mean", "precision_mean", "recall_mean", "f1_mean"]:
        raw = results_raw[key]
        clean = results_clean[key]
        diff = clean - raw
        print(f"{key:<14}{raw:>13.4f}{clean:>13.4f}{diff:+11.4f}")

    winner = "clean" if results_clean["accuracy_mean"] >= results_raw["accuracy_mean"] else "raw"
    print(f"\nWinner: {EXPERIMENT_NAMES[winner]}")
    return winner


def main() -> None:
    metrics_raw, model_raw = run_experiment(clean=False)
    metrics_clean, model_clean = run_experiment(clean=True)

    winner = compare_experiments(metrics_raw, metrics_clean)
    model = model_clean if winner == "clean" else model_raw

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Best model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()