import os
import pandas as pd
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    confusion_matrix, precision_score, recall_score
)

from utils.config import data_config

def train_xgboost_model(
    feature_csv="radiomics/radiomic_features.csv",
    label_csv=None,
    model_output="radiomics/xgb_radiomics_model.pkl"
):
    os.makedirs(os.path.dirname(model_output), exist_ok=True)
    label_csv = label_csv or data_config["label_file"]

    # === 加载数据 ===
    df = pd.read_csv(feature_csv)
    label_df = pd.read_csv(label_csv)
    df = df.merge(label_df, on="patient_id")

    X = df.drop(columns=["patient_id", "label"])
    y = df["label"]

    # === 标准化 ===
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # === 特征选择：LASSO + SelectFromModel ===
    print("[INFO] Performing LASSO-based feature selection...")
    lasso = LassoCV(cv=5, random_state=42).fit(X_scaled, y)
    selector = SelectFromModel(lasso, prefit=True)
    X_selected = selector.transform(X_scaled)
    selected_features = X.columns[selector.get_support()].tolist()

    print(f"[INFO] Selected {len(selected_features)} radiomic features")

    # === 训练/验证集划分 ===
    X_train, X_val, y_train, y_val = train_test_split(X_selected, y, test_size=0.3, random_state=42)

    # === 训练XGBoost模型 ===
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )

    model.fit(X_train, y_train)

    # === 模型评估 ===
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]

    acc = accuracy_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_prob)
    f1 = f1_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
    spec = tn / (tn + fp)

    print(f"[RESULT] Accuracy: {acc:.4f}")
    print(f"[RESULT] AUC: {auc:.4f}")
    print(f"[RESULT] F1 Score: {f1:.4f}")
    print(f"[RESULT] Sensitivity: {recall:.4f} | Specificity: {spec:.4f} | Precision: {prec:.4f}")

    # === 保存模型及附属信息 ===
    joblib.dump({
        "model": model,
        "scaler": scaler,
        "selector": selector,
        "selected_features": selected_features
    }, model_output)

    print(f"[INFO] Radiomics XGBoost model saved to: {model_output}")
