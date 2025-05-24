"""
Generate Grad-CAM heatmaps for visualizing model attention.
"""

import os
import torch
import numpy as np
import cv2
import nibabel as nib
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import torch.nn.functional as F

from utils.config import data_config

class CT2DSliceDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_list = [f.replace(".nii.gz", "") for f in os.listdir(image_dir) if f.endswith(".nii.gz")]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        pid = self.image_list[idx]
        path = os.path.join(self.image_dir, pid + ".nii.gz")
        try:
            data = nib.load(path).get_fdata()
            max_slice = np.argmax([np.sum(slice) for slice in data.transpose(2, 0, 1)])
            img = data[:, :, max_slice]
            img = np.clip(img, -75, 175)
            img = ((img + 75) / 250 * 255).astype(np.uint8)
            pil_img = Image.fromarray(img).convert("L")
            if self.transform:
                pil_img_tensor = self.transform(pil_img)
            else:
                pil_img_tensor = transforms.ToTensor()(pil_img)
            return pil_img_tensor.unsqueeze(0), pid, img
        except Exception as e:
            print(f"[ERROR] Failed to load {pid}: {str(e)}")
            return None

def show_cam_on_image(img, mask, output_path):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    if len(img.shape) == 2:
        img_rgb = np.stack([img] * 3, axis=-1)
    else:
        img_rgb = img
    overlay = cv2.addWeighted(heatmap, 0.4, img_rgb.astype(np.uint8), 0.6, 0)
    cv2.imwrite(output_path, overlay)

def generate_grad_cam(
    model_path="deep_learning/best_resnet101.pth",
    image_dir=data_config["nii_image_dir"],
    output_dir="deep_learning/grad_cam/"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Load Model ===
    model = models.resnet101(pretrained=False)
    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # === Hook Target Layer ===
    target_layer = model.layer4[-1]
    feature_map = {}
    gradients = {}

    def forward_hook(module, input, output):
        feature_map["value"] = output

    def backward_hook(module, grad_in, grad_out):
        gradients["value"] = grad_out[0]

    target_layer.register_forward_hook(forward_hook)
    target_layer.register_backward_hook(backward_hook)

    # === Prepare Dataset ===
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    dataset = CT2DSliceDataset(image_dir, transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    os.makedirs(output_dir, exist_ok=True)

    print("[INFO] Starting Grad-CAM generation...")
    for batch in tqdm(dataloader):
        if batch is None or batch[0] is None:
            continue
        img, pid, raw = batch
        img = img.squeeze(0).to(device)  # [1, 1, 224, 224]

        try:
            output = model(img)
            pred_class = output.argmax(dim=1)
            score = output[:, pred_class]
            model.zero_grad()
            score.backward()

            grads_val = gradients["value"].cpu().data.numpy()[0]
            fmap = feature_map["value"].cpu().data.numpy()[0]
            weights = np.mean(grads_val, axis=(1, 2))

            cam = np.zeros(fmap.shape[1:], dtype=np.float32)
            for i, w in enumerate(weights):
                cam += w * fmap[i, :, :]

            cam = np.maximum(cam, 0)
            cam = cv2.resize(cam, (raw.shape[2], raw.shape[1]))
            cam = cam - np.min(cam)
            cam = cam / np.max(cam + 1e-8)

            output_path = os.path.join(output_dir, f"{pid[0]}_cam.jpg")
            show_cam_on_image(raw.squeeze().numpy(), cam, output_path)
        except Exception as e:
            print(f"[ERROR] Grad-CAM failed for {pid}: {str(e)}")

    print(f"[INFO] Grad-CAM heatmaps saved to {output_dir}")
