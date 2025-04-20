import torch.nn as nn


class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, layer_1_size=128, layer_2_size=64):
        super(NeuralNetwork, self).__init__()
        # Input layer
        self.layer1 = nn.Linear(input_dim, layer_1_size)
        self.activation1 = nn.LeakyReLU()
        # Hidden layer
        self.layer2 = nn.Linear(layer_1_size, layer_2_size)
        self.activation2 = nn.LeakyReLU()
        # Output layer
        self.output = nn.Linear(layer_2_size, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation1(x)
        x = self.layer2(x)
        x = self.activation2(x)
        x = self.output(x)
        return x
