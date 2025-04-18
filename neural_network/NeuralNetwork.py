import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, layer_1_size=128, layer_2_size=64):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        # First hidden layer
        self.fc1 = nn.Linear(input_dim, layer_1_size)
        self.relu1 = nn.LeakyReLU()
        # Second hidden layer
        self.fc2 = nn.Linear(layer_1_size, layer_2_size)
        self.relu2 = nn.LeakyReLU()
        # Output layer
        self.fc3 = nn.Linear(layer_2_size, 1)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x
