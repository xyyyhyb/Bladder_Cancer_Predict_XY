import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset, random_split
import nibabel as nib
import numpy as np
from PIL import Image
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

from utils.config import data_config, train_config

class Bladder2DDataset(Dataset):
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
            img_data = nib.load(img_path).get_fdata()
            max_slice = np.argmax([np.sum(slice) for slice in img_data.transpose(2, 0, 1)])
            slice_img = img_data[:, :, max_slice]
            slice_img = np.clip(slice_img, -75, 175)
            slice_img = ((slice_img + 75) / 250 * 255).astype(np.uint8)
            pil_img = Image.fromarray(slice_img).convert("L")
            if self.transform:
                pil_img = self.transform(pil_img)
            label = self.labels[pid]
            return pil_img, torch.tensor(label, dtype=torch.long)
        except Exception as e:
            print(f"[ERROR] Failed to load {pid}: {str(e)}")
            return None

def train_resnet101_model(data_config, train_config, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize((train_config['image_size'], train_config['image_size'])),
        transforms.ToTensor()
    ])

    dataset = Bladder2DDataset(data_config['nii_image_dir'], data_config['label_file'], transform=transform)
    dataset = [d for d in dataset if d is not None]  # Filter None
    total_size = len(dataset)
    train_size = int(train_config['train_val_split_ratio'] * total_size)
    val_size = total_size - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=train_config['batch_size'], shuffle=True, num_workers=train_config['num_workers'])
    val_loader = DataLoader(val_ds, batch_size=train_config['batch_size'], shuffle=False, num_workers=train_config['num_workers'])

    model = models.resnet101(pretrained=True)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=train_config['learning_rate'], weight_decay=train_config['weight_decay'])

    best_val_loss = float("inf")
    patience = train_config['early_stopping_patience']
    early_stop_counter = 0

    os.makedirs("deep_learning", exist_ok=True)
    log_file = open("deep_learning/train_log.txt", "w")

    for epoch in range(train_config['epochs']):
        model.train()
        train_loss = 0.0
        y_true, y_pred = [], []

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(torch.argmax(outputs, dim=1).cpu().numpy())

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        print(f"[Train Epoch {epoch+1}] Loss: {train_loss/len(train_loader.dataset):.4f} | Acc: {acc:.4f} | F1: {f1:.4f}")

        model.eval()
        val_loss = 0.0
        y_val_true, y_val_pred, y_val_prob = [], [], []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                y_val_true.extend(labels.cpu().numpy())
                y_val_pred.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                y_val_prob.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())

        val_acc = accuracy_score(y_val_true, y_val_pred)
        val_f1 = f1_score(y_val_true, y_val_pred)
        val_auc = roc_auc_score(y_val_true, y_val_prob)
        print(f"[Validation Epoch {epoch+1}] Loss: {val_loss/len(val_loader.dataset):.4f} | Acc: {val_acc:.4f} | AUC: {val_auc:.4f} | F1: {val_f1:.4f}")
        log_file.write(f"{epoch+1},{val_loss:.4f},{val_acc:.4f},{val_auc:.4f},{val_f1:.4f}\n")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), "deep_learning/best_resnet101.pth")
            print("[INFO] Saved best model.")
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print("[INFO] Early stopping triggered.")
                break

    log_file.close()
