"""
Evaluate model predictions: Accuracy, AUC, Sensitivity, Specificity, F1 Score.
"""

import os
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    confusion_matrix, precision_score, recall_score
)
from utils.config import eval_config, data_config

def evaluate_model(pred_csv, label_csv, model_name="DLRN", threshold=0.5):
    # === Load true labels ===
    label_df = pd.read_csv(label_csv)
    true_labels = label_df.set_index("patient_id")["label"]

    # === Load prediction ===
    pred_df = pd.read_csv(pred_csv).set_index("patient_id")
    merged = pred_df.join(true_labels, how="inner")

    if merged.isnull().values.any():
        merged = merged.dropna()
        print(f"[WARN] Missing labels or predictions for some patient_ids. Cleaned.")

    y_true = merged["label"].values
    y_prob = merged["pred_prob"].values
    y_pred = (y_prob >= threshold).astype(int)

    # === Compute metrics ===
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    f1 = f1_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    spec = tn / (tn + fp)

    metrics = {
        "Model": model_name,
        "Threshold": threshold,
        "Accuracy": acc,
        "AUC": auc,
        "Sensitivity": recall,
        "Specificity": spec,
        "Precision": prec,
        "F1": f1
    }

    return metrics

def evaluate_all_models():
    os.makedirs("evaluation", exist_ok=True)
    threshold = eval_config.get("threshold_range", [0.5])[0]  # default use first

    summary = []

    # Evaluate DLRN model
    if os.path.exists("fusion_model/dlrn_predictions.csv"):
        metrics_dlrn = evaluate_model(
            pred_csv="fusion_model/dlrn_predictions.csv",
            label_csv=data_config["label_file"],
            model_name="DLRN",
            threshold=threshold
        )
        summary.append(metrics_dlrn)

    # === Extend here to support Radiomics or DL-only prediction files ===
    # if os.path.exists("radiomics/radiomics_predictions.csv"):
    #     ...

    df = pd.DataFrame(summary)
    df.to_csv("evaluation/evaluation_summary.csv", index=False)
    print("[INFO] Evaluation results saved to evaluation/evaluation_summary.csv")
    print(df.to_string(index=False))

if __name__ == "__main__":
    evaluate_all_models()
