import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from PIL import Image
import torchvision.transforms.functional as TF

import synchrotron_nn as snn

# Configuration for training
DATA_DIR = "../SOFT/SOFT_runs/parameter_scan/data_combined"
IMG_SIZE = 600  # Change if needed
BATCH_SIZE = 64
NUM_EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Custom dataset for the grayscale images in the training database
class RunawayDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []

        for fname in os.listdir(os.path.join(root_dir, "no_runaway")):
            if not fname.endswith(".jpeg"): continue
            self.samples.append({
                "path": os.path.join(root_dir, "no_runaway", fname),
                "runaway": 0,
                "q_profile": -1
            })

        for q_type in ["linear", "quadratic"]:
            folder = os.path.join(root_dir, "runaway", q_type)
            for fname in os.listdir(folder):
                if not fname.endswith(".jpeg"): continue
                self.samples.append({
                    "path": os.path.join(folder, fname),
                    "runaway": 1,
                    "q_profile": 0 if q_type == "linear" else 1
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample["path"]).convert("L")  # Grayscale
        image = TF.resize(image, (IMG_SIZE, IMG_SIZE))
        image = TF.to_tensor(image)  # Result: shape (1, H, W)

        return image, sample["runaway"], sample["q_profile"]

# Load the data to the dataset, and create a dataoader to handle it
dataset = RunawayDataset(DATA_DIR)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=32, pin_memory=True)

# Generate the model
model = snn.Synchrotron_nn().to(DEVICE)

# Loss and optimizer
criterion_runaway = nn.BCELoss()
criterion_qprofile = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    correct_runaway = 0
    correct_q = 0
    total = 0

    for images, runaway_labels, q_labels in dataloader:
        images = images.to(DEVICE)
        runaway_labels = runaway_labels.float().to(DEVICE)
        q_labels = q_labels.to(DEVICE)

        pred_runaway, pred_q = model(images)

        loss_runaway = criterion_runaway(pred_runaway, runaway_labels)

        mask = runaway_labels == 1
        if mask.sum() > 0:
            loss_q = criterion_qprofile(pred_q[mask], q_labels[mask])
            loss = loss_runaway + loss_q
        else:
            loss = loss_runaway

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accuracy metrics
        total += images.size(0)
        pred_runaway_binary = (pred_runaway > 0.5).long()
        correct_runaway += (pred_runaway_binary == runaway_labels.long()).sum().item()
        correct_q += ((pred_q.argmax(dim=1) == q_labels) & mask).sum().item()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | "
          f"Loss: {total_loss:.3f} | "
          f"Runaway Acc: {correct_runaway / total:.2f} | "
          f"Q Acc: {correct_q / max(mask.sum().item(), 1):.2f}")
          
torch.save(model.state_dict(), "model_weights.pth")

print("Training complete.")
