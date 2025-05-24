"""
Configuration file for bladder cancer AI project.
"""

data_config = {
    "nii_image_dir": "data/processed/",            
    "mask_dir": "data/masks/",                      
    "label_file": "data/labels.csv",               
    "center_split_file": "data/cohorts_split.json", 
    "window_level": [-75, 175],                     
    "max_slice_selection": "z_axis_max",            
}

train_config = {
    "batch_size": 16,
    "epochs": 50,
    "learning_rate": 1e-4,
    "optimizer": "adam",
    "weight_decay": 1e-5,
    "early_stopping_patience": 10,
    "lr_scheduler": "cosine",       
    "num_workers": 4,
    "image_size": 224,              
    "roi_size": [64, 64, 48],       
    "train_val_split_ratio": 0.7,   
    "cross_validation_folds": 5     
}

model_config = {
    "radiomics_model": "xgboost",              
    "dl_model": "resnet101",                   
    "feature_combination": "concat+lasso",    
    "final_model": "DLRN",                    
    "shap_top_features": 20                    
}

survival_config = {
    "cox_features": ["Combined", "age", "BMI", "hydronephrosis", "gender"],
    "risk_cutoff": 0.283,                      
    "time_points": [1, 3, 5]                   
}

eval_config = {
    "metrics": ["AUC", "Accuracy", "Sensitivity", "Specificity", "F1"],          
    "doctor_vs_ai": True,                      
    "radiologist_included": True              
}

esrgan_config = {
    "scale_factor": 4,
    "lr": 1e-4,
    "batch_size": 8,
    "epochs": 100,
    "gan_arch": "ESRGAN",      
    "hr_patch_size": 128,
    "lr_patch_size": 32
}

vit_settings = {
    "image_size": 64,
    "frames": 48,
    "image_patch_size": 16,
    "frame_patch_size": 2,
    "dim": 1024,
    "depth": 6,
    "heads": 8,
    "mlp_dim": 2048,
    "dropout": 0.1
}
