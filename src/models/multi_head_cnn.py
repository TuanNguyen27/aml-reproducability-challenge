import torch
import torch.nn.functional as F

class MultiHeadCNN(torch.nn.Module):
    def __init__(self, n_heads=1, split_mnist=True):
        super().__init__()
        # check for bad parameters
        if n_heads < 1:
            raise ValueError('Network requires at least one head.')

        # shared network
        self.conv1 = torch.nn.Conv2d(1, 32, 3, 1)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = torch.nn.Dropout2d(0.25)
        self.dropout2 = torch.nn.Dropout2d(0.5)
        self.fc1 = torch.nn.Linear(9216, 128)
        self.y_dim = 10 if split_mnist else 2
        # multi-heads
        self.heads = torch.nn.ModuleList([
            torch.nn.Linear(128, self.y_dim) for _ in range(n_heads)
        ])

    def forward(self, x, head_idx=0):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.heads[head_idx](x)
        output = F.log_softmax(x, dim=1)
        return output
