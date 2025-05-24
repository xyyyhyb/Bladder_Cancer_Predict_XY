import os
import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import nibabel as nib
import numpy as np
from PIL import Image
from tqdm import tqdm
from utils.config import data_config

class DLFeatureDataset(Dataset):
    def __init__(self, image_dir, label_file, transform=None):
        self.image_dir = image_dir
        self.labels = self._load_labels(label_file)
        self.transform = transform
        self.image_list = list(self.labels.keys())

    def _load_labels(self, label_file):
        df = pd.read_csv(label_file)
        return dict(zip(df['patient_id'], df['label']))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        pid = self.image_list[idx]
        img_path = os.path.join(self.image_dir, f"{pid}.nii.gz")
        try:
            img_nib = nib.load(img_path)
            img_data = img_nib.get_fdata()
            max_slice = np.argmax([np.sum(slice) for slice in img_data.transpose(2, 0, 1)])
            slice_img = img_data[:, :, max_slice]
            slice_img = np.clip(slice_img, -75, 175)
            slice_img = ((slice_img + 75) / 250 * 255).astype(np.uint8)
            pil_img = Image.fromarray(slice_img).convert("L")
            if self.transform:
                pil_img = self.transform(pil_img)
            return pil_img, pid
        except Exception as e:
            print(f"[ERROR] Failed to process {pid}: {str(e)}")
            return None

def extract_deep_features(
    data_config,
    model_path="best_resnet101.pth",
    output_csv="deep_learning/deep_features.csv",
    batch_size=4,
    num_workers=2
):
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = DLFeatureDataset(data_config["nii_image_dir"], data_config["label_file"], transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # === Load pretrained ResNet101 ===
    model = models.resnet101(pretrained=False)
    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = torch.nn.Identity()  # Output avgpool 2048-d
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    all_features = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting DL features"):
            valid_batch = [item for item in batch if item is not None]
            if len(valid_batch) == 0:
                continue
            imgs, pids = zip(*valid_batch)
            imgs = torch.stack(imgs).to(device)

            try:
                feats = model(imgs).cpu().numpy()
                for i in range(len(pids)):
                    row = {"patient_id": pids[i]}
                    row.update({f"DL_{j}": feats[i][j] for j in range(feats.shape[1])})
                    all_features.append(row)
            except Exception as e:
                print(f"[ERROR] Model forward failed: {str(e)}")

    df = pd.DataFrame(all_features)
    df.to_csv(output_csv, index=False)
    print(f"[INFO] Deep features saved to {output_csv}")
