import pandas as pd
import numpy as np
import joblib
import os

from sklearn.preprocessing import StandardScaler

def infer_dlrn():
    # === Load trained hybrid model ===
    model_data = joblib.load("fusion_model/dlrn_model.pkl")
    clf = model_data["model"]
    features_rad = model_data["features_rad"]
    scaler_rad = model_data["scaler_rad"]
    scaler_dl = model_data["scaler_dl"]

    # === Load features ===
    radiomics_df = pd.read_csv("radiomics/radiomic_features.csv")
    dl_df = pd.read_csv("deep_learning/deep_features.csv")

    # === Match patients across both sets ===
    common_ids = set(radiomics_df["patient_id"]) & set(dl_df["patient_id"])
    if len(common_ids) == 0:
        raise ValueError("No matching patient_id found between radiomics and deep learning features!")

    rad_df = radiomics_df[radiomics_df["patient_id"].isin(common_ids)].copy()
    dl_df = dl_df[dl_df["patient_id"].isin(common_ids)].copy()

    rad_df = rad_df.set_index("patient_id").loc[common_ids]
    dl_df = dl_df.set_index("patient_id").loc[common_ids]

    # === Feature extraction and normalization ===
    rad_scaled = scaler_rad.transform(rad_df[features_rad])
    dl_scaled = scaler_dl.transform(dl_df)

    X_combined = np.concatenate([rad_scaled, dl_scaled], axis=1)

    # === Predict probability ===
    prob = clf.predict_proba(X_combined)[:, 1]

    result_df = pd.DataFrame({
        "patient_id": list(common_ids),
        "pred_prob": prob
    })

    os.makedirs("fusion_model", exist_ok=True)
    result_df.to_csv("fusion_model/dlrn_predictions.csv", index=False)
    print("[INFO] Predictions saved to fusion_model/dlrn_predictions.csv")
