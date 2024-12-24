import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=10, kernel_size=3, stride=2)
        self.max_pool1 = nn.MaxPool1d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv1d(10, 20, 3, 2)
        self.max_pool2 = nn.MaxPool1d(3, 2)
        self.conv3 = nn.Conv1d(20, 40, 3, 2)
        self.liner1 = nn.Linear(5720, 120)
        self.liner2 = nn.Linear(120, 84)
        self.liner3 = nn.Linear(84, 5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.max_pool1(x)
        x = F.relu(self.conv2(x))
        x = self.max_pool2(x)
        x = F.relu(self.conv3(x))
        x = x.view(-1, 5720)
        x = F.relu(self.liner1(x))
        x = F.relu(self.liner2(x))
        x = self.liner3(x)
        return x
