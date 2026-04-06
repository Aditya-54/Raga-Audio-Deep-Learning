import torch
import torch.nn as nn

class CNN1D(nn.Module):
    def __init__(self, num_classes, input_size):
        super(CNN1D, self).__init__()

        self.conv1 = nn.Conv1d(1, 64, kernel_size=3)
        self.pool = nn.MaxPool1d(2)
        self.bn1 = nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(128)

        self.flatten = nn.Flatten()

        self._to_linear = None
        self._get_conv_output(input_size)

        self.fc1 = nn.Linear(self._to_linear, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def _get_conv_output(self, input_size):
        x = torch.zeros(1, 1, input_size)
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        self._to_linear = x.numel()

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.bn1(x)

        x = self.pool(torch.relu(self.conv2(x)))
        x = self.bn2(x)

        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)

        return self.fc2(x)