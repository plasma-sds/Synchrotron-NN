import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models
from PIL import Image
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

import synchrotron_nn as snn

# Configuration
DATA_DIR = "../SOFT/SOFT_runs/parameter_scan/data_combined"
IMG_SIZE = 600
BATCH_SIZE = 32
NUM_EPOCHS = 10
DEVICE = torch.device("cpu")

#generator = torch.Generator().manual_seed(42)

# Custom dataset
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
        image = Image.open(sample["path"]).convert("L")
        image = TF.resize(image, (IMG_SIZE, IMG_SIZE))
        image = TF.to_tensor(image)
        return image, sample["runaway"], sample["q_profile"]

# Load full dataset
dataset = RunawayDataset(DATA_DIR)

# Split into train (60%), val (20%), test (20%)
total_len = len(dataset)
train_len = int(0.6 * total_len)
val_len = int(0.2 * total_len)
test_len = total_len - train_len - val_len
train_set, val_set, test_set = random_split(dataset, [train_len, val_len, test_len])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)

# Model, loss, optimizer
model = snn.Synchrotron_nn().to(DEVICE)
criterion_runaway = nn.BCELoss()
criterion_qprofile = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Save accuracy during training, so it can be plotted later
train_losses = []
val_losses = []
train_runaway_accs = []
val_runaway_accs = []
train_q_accs = []
val_q_accs = []

# Training loop
for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    correct_runaway = 0
    correct_q = 0
    total = 0
    total_runaway = 0

    for images, runaway_labels, q_labels in train_loader:
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

        total += images.size(0)
        total_runaway += mask.sum().item()
        pred_runaway_binary = (pred_runaway > 0.5).long()
        correct_runaway += (pred_runaway_binary == runaway_labels.long()).sum().item()
        correct_q += ((pred_q.argmax(dim=1) == q_labels) & mask).sum().item()
        total_loss += loss.item()

    # Append average metrics once per epoch
    train_losses.append(total_loss / len(train_loader))
    train_runaway_accs.append(correct_runaway / total)
    train_q_accs.append(correct_q / max(total_runaway, 1))

    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | "
          f"Train Loss: {train_losses[-1]:.3f} | "
          f"Runaway Acc: {train_runaway_accs[-1]:.2f} | "
          f"Q Acc: {train_q_accs[-1]:.2f}")

    # Validation loop
    model.eval()
    val_loss = 0
    val_correct_runaway = 0
    val_correct_q = 0
    val_total = 0
    val_runaway_total = 0

    with torch.no_grad():
        for images, runaway_labels, q_labels in val_loader:
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

            val_loss += loss.item()
            val_total += images.size(0)
            val_runaway_total += mask.sum().item()
            pred_runaway_binary = (pred_runaway > 0.5).long()
            val_correct_runaway += (pred_runaway_binary == runaway_labels.long()).sum().item()
            val_correct_q += ((pred_q.argmax(dim=1) == q_labels) & mask).sum().item()

    # Append validation metrics once per epoch
    val_losses.append(val_loss / len(val_loader))
    val_runaway_accs.append(val_correct_runaway / val_total)
    val_q_accs.append(val_correct_q / max(val_runaway_total, 1))

    print(f"   >> Validation Loss: {val_losses[-1]:.3f} | "
          f"Runaway Acc: {val_runaway_accs[-1]:.2f} | "
          f"Q Acc: {val_q_accs[-1]:.2f}")



# Save model after training
torch.save(model.state_dict(), "model_weights.pth")
print("Training complete.")

epochs = range(1, NUM_EPOCHS + 1)

plt.figure(figsize=(12, 6))

# Plot loss
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, label="Train Loss")
plt.plot(epochs, val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Over Epochs")
plt.legend()

# Plot accuracies
plt.subplot(1, 2, 2)
plt.plot(epochs, train_runaway_accs, label="Train Runaway Acc")
plt.plot(epochs, val_runaway_accs, label="Val Runaway Acc")
plt.plot(epochs, train_q_accs, label="Train Q Acc")
plt.plot(epochs, val_q_accs, label="Val Q Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy Over Epochs")
plt.legend()

plt.tight_layout()
plt.savefig("training_plot.png")

# Check on test data if training was successful

# Create a test DataLoader
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)

# Evaluate on test data
model.eval()
test_loss = 0
correct_runaway = 0
correct_q = 0
total = 0
runaway_total = 0

with torch.no_grad():
    for images, runaway_labels, q_labels in test_loader:
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

        test_loss += loss.item()
        total += images.size(0)
        runaway_total += mask.sum().item()

        pred_runaway_binary = (pred_runaway > 0.5).long()
        correct_runaway += (pred_runaway_binary == runaway_labels.long()).sum().item()
        correct_q += ((pred_q.argmax(dim=1) == q_labels) & mask).sum().item()

print(f"\n🧪 Final Test Results:")
print(f"Test Loss: {test_loss:.3f}")
print(f"Runaway Accuracy: {correct_runaway / total:.2f}")
print(f"Q Profile Accuracy: {correct_q / max(runaway_total, 1):.2f}")



