## The neural network class

from torch import nn
from torchvision import models

# Modified ResNet18 to accept grayscale images (1 channel)
class Synchrotron_nn(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = models.resnet18(pretrained=True)
        self.base.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 1 input channel
        self.base.fc = nn.Identity()
        self.feature_dim = 512

        self.runaway_head = nn.Linear(self.feature_dim, 1)
        self.qprofile_head = nn.Linear(self.feature_dim, 2)

    def forward(self, x):
        feats = self.base(x)
        runaway_out = torch.sigmoid(self.runaway_head(feats))
        qprofile_out = self.qprofile_head(feats)
        return runaway_out.squeeze(1), qprofile_out