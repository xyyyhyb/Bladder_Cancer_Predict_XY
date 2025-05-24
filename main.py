import os
from utils.config import data_config

# === 各模块导入 ===
from radiomics.extract_features import extract_radiomic_features
from radiomics.train_xgboost import train_xgboost_model
from deep_learning.extract_features import extract_deep_features
from deep_learning.train_dl import train_resnet101_model
from fusion_model.train_combined_model import train_dlrn_model
from fusion_model.dlrn_infer import infer_dlrn
from evaluation.evaluate_metrics import evaluate_all_models
from grad_cam import generate_grad_cam

import torch

def main():
    print("\n=== Step 1: Extract Radiomic Features ===")
    extract_radiomic_features(data_config)

    print("\n=== Step 2: Train Radiomics XGBoost Model ===")
    train_xgboost_model()

    print("\n=== Step 3: Train ResNet101 DL Model ===")
    train_resnet101_model(data_config=data_config, train_config=train_config)

    print("\n=== Step 4: Extract Deep Learning Features ===")
    extract_deep_features(data_config)

    print("\n=== Step 5: Train Combined DLRN Model ===")
    train_dlrn_model(data_config, model_config=None)

    print("\n=== Step 6: Run DLRN Inference ===")
    infer_dlrn()

    print("\n=== Step 7: Evaluate Model Performance ===")
    evaluate_all_models()

    print("\n=== Step 8: Generate Grad-CAM (optional) ===")
    generate_grad_cam()

if __name__ == "__main__":
    from utils.config import train_config, model_config  
    main()
