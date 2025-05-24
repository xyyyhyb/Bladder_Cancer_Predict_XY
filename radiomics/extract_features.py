import os
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor
from tqdm import tqdm
from utils.config import data_config

def extract_radiomic_features(data_config, output_csv="radiomics/radiomic_features.csv"):
    image_dir = data_config['nii_image_dir']
    mask_dir = data_config['mask_dir']
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    # === Define PyRadiomics parameters ===
    params = {
        'binWidth': 25,
        'resampledPixelSpacing': None,
        'interpolator': 'sitkBSpline',
        'verbose': False
    }

    extractor = featureextractor.RadiomicsFeatureExtractor(**params)

    results = []
    failed_patients = []

    for filename in tqdm(os.listdir(image_dir), desc="Extracting radiomic features"):
        if not filename.endswith(".nii.gz"):
            continue

        pid = filename.replace(".nii.gz", "")
        image_path = os.path.join(image_dir, filename)
        mask_path = os.path.join(mask_dir, filename)

        if not os.path.exists(mask_path):
            print(f"[WARN] Missing mask for {pid}, skipping.")
            failed_patients.append(pid)
            continue

        try:
            feature_vector = extractor.execute(image_path, mask_path)
            # Exclude diagnostics
            filtered = {k: v for k, v in feature_vector.items() if "diagnostics" not in k}
            filtered["patient_id"] = pid
            results.append(filtered)
        except Exception as e:
            print(f"[ERROR] Feature extraction failed for {pid}: {str(e)}")
            failed_patients.append(pid)

    # === Save results ===
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"[INFO] Radiomic features saved to: {output_csv}")
    print(f"[SUMMARY] Total patients processed: {len(results)}, Failed: {len(failed_patients)}")
