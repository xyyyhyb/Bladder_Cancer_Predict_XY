import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler

def train_dlrn_model(data_config, model_config, device=None):
    # === Load XGBoost radiomic model and selected features ===
    xgb_model_data = joblib.load("radiomics/xgb_radiomics_model.pkl")
    selected_radiomic_features = xgb_model_data["selected_features"]
    scaler_rad = xgb_model_data["scaler"]
    selector_rad = xgb_model_data["selector"]

    # === Load radiomic feature CSV ===
    radiomics_df = pd.read_csv("radiomics/radiomic_features.csv")
    label_df = pd.read_csv(data_config["label_file"])
    merged = radiomics_df.merge(label_df, on="patient_id")

    X_rad = merged[selected_radiomic_features]
    y = merged["label"]
    X_rad_scaled = scaler_rad.transform(X_rad)

    # === Load deep learning features ===
    dl_df = pd.read_csv("deep_learning/deep_features.csv")
    X_dl = dl_df.drop(columns=["patient_id"])
    scaler_dl = StandardScaler()
    X_dl_scaled = scaler_dl.fit_transform(X_dl)

    # === Merge by patient_id ===
    df_final = merged[["patient_id", "label"]].merge(dl_df, on="patient_id")
    X_combined = np.concatenate([X_rad_scaled, X_dl_scaled], axis=1)
    y_combined = df_final["label"]

    # === Train logistic regression classifier (DLRN) ===
    X_train, X_val, y_train, y_val = train_test_split(X_combined, y_combined, test_size=0.3, random_state=42)
    clf = LogisticRegressionCV(cv=5, max_iter=1000, penalty='l2', solver='liblinear')
    clf.fit(X_train, y_train)

    # === Evaluate ===
    y_pred = clf.predict(X_val)
    y_prob = clf.predict_proba(X_val)[:, 1]

    acc = accuracy_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_prob)
    f1 = f1_score(y_val, y_pred)

    print(f"[DLRN Evaluation] Accuracy: {acc:.4f} | AUC: {auc:.4f} | F1 Score: {f1:.4f}")

    # === Save model and components ===
    os.makedirs("fusion_model", exist_ok=True)
    joblib.dump({
        "model": clf,
        "scaler_dl": scaler_dl,
        "scaler_rad": scaler_rad,
        "features_rad": selected_radiomic_features,
        "features_dl": X_dl.columns.tolist()
    }, "fusion_model/dlrn_model.pkl")

    print("[INFO] DLRN model saved to fusion_model/dlrn_model.pkl")
