import torch
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

import synchrotron_nn as snn
import Utility.decision_highlighter as dh

# Define DEVICE and IMAGE size
DEVICE = torch.device("cpu")
IMG_SIZE = 600

# Load the model weights
model = snn.Synchrotron_nn().to(DEVICE)
model.load_state_dict(torch.load("training_data/Second_training/model_weights.pth"))
model.eval()  # Set the model to evaluation mode


# Load and preprocess a new image
image_path = "training_data/Second_training/test_image_video4.jpeg"
image = Image.open(image_path).convert("L")  # Convert to grayscale
image = TF.resize(image, (IMG_SIZE, IMG_SIZE))
image = TF.to_tensor(image).unsqueeze(0)  # Add batch dimension: shape (1, 1, H, W)
image = image.to(DEVICE)  # Move to the appropriate device (GPU or CPU)

with torch.no_grad():
    pred_runaway, pred_qprofile_logits = model(image)

# Runaway confidence
runaway_confidence = pred_runaway.item()
runaway_label = "Yes" if runaway_confidence > 0.5 else "No"

# Q-profile confidence
qprofile_probs = torch.softmax(pred_qprofile_logits, dim=1)
qprofile_confidence, pred_qprofile_class = torch.max(qprofile_probs, dim=1)
qprofile_confidence = qprofile_confidence.item()
qprofile_label = "Linear" if pred_qprofile_class.item() == 0 else "Quadratic"

# Output
print(f"Runaway Prediction: {runaway_label} (Confidence: {runaway_confidence:.2f})")
print(f"Quadratic Profile Prediction: {qprofile_label} (Confidence: {qprofile_confidence:.2f})")

# Load grayscale image
image = Image.open(image_path).convert('L')
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])
input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

# Hook into last conv layer
target_layer = model.base.layer4[-1].conv2
cam_extractor = dh.GradCAM(model, target_layer)

# Generate CAM
cam_mask = cam_extractor.generate_cam(input_tensor)

# Visualize
img_np = np.array(image.resize((IMG_SIZE, IMG_SIZE))) / 255.0
img_np = np.expand_dims(img_np, axis=2)
visualization = dh.show_cam_on_image(img_np, cam_mask)

plt.imshow(visualization)
plt.title("Grad-CAM: Runaway Detected")
plt.axis('off')
plt.show()
