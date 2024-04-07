import torch.nn as nn

# He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition pp. 5934-5938
# arXiv:1810.04805.

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(channels, channels)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(channels, channels)

    def forward(self, x):
        residual = x
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x += residual
        x = self.relu(x)
        return x