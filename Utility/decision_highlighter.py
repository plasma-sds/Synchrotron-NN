#######################################################################
##
## Script to highlight parts of the image which is used by a neural
## network to make a prediction.
##
## Created by Soma Olasz, 05.2025
##
#######################################################################

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook()

    def hook(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate_cam(self, input_tensor):
        runaway_out, _ = self.model(input_tensor)
        self.model.zero_grad()
        runaway_out.backward(torch.ones_like(runaway_out))

        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations[0]

        for i in range(activations.shape[0]):
            activations[i] *= pooled_gradients[i]

        cam = torch.sum(activations, dim=0)
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / cam.max()

        # 🔧 Resize CAM to match input image size
        cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0), size=(input_tensor.shape[2], input_tensor.shape[3]), mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        return cam


def show_cam_on_image(grayscale_img, mask):
    grayscale_img = np.repeat(grayscale_img, 3, axis=2)  # Convert to RGB
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + grayscale_img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

