"""Shared evaluation: metrics json, per-class report, confusion matrix, error analysis."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, f1_score)


ROOT = Path(__file__).resolve().parents[1]


def evaluate(texts, y_true, y_pred, confidence, labels, name):
    out = ROOT / "results"
    out.mkdir(exist_ok=True)

    metrics = {
        "model": name,
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro"), 4),
        "weighted_f1": round(f1_score(y_true, y_pred, average="weighted"), 4),
        "n_test": len(y_true),
    }
    (out / f"{name}_metrics.json").write_text(json.dumps(metrics, indent=2))

    report = classification_report(y_true, y_pred, labels=labels, zero_division=0)
    (out / f"{name}_report.txt").write_text(report)

    fig, ax = plt.subplots(figsize=(9, 8))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, labels=labels, normalize="true", xticks_rotation=45,
        ax=ax, colorbar=False, values_format=".2f",
    )
    ax.set_title(f"{name}: normalized confusion matrix")
    fig.tight_layout()
    fig.savefig(out / f"{name}_confusion_matrix.png", dpi=150)
    plt.close(fig)

    # every wrong prediction, most confident mistakes first, for error analysis
    errors = pd.DataFrame(
        {"text": list(texts), "true": list(y_true), "pred": list(y_pred),
         "confidence": list(confidence)}
    )
    errors = errors[errors["true"] != errors["pred"]]
    errors.sort_values("confidence", ascending=False).to_csv(
        out / f"{name}_errors.csv", index=False
    )

    print(f"\n== {name} ==")
    print(json.dumps(metrics, indent=2))
    print(report)
    return metrics
