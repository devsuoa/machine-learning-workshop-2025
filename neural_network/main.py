import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from common import TARGET
from process_data import process_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('neural_network/data.csv')
# df = process_data()
X = df.drop(columns=[TARGET]).values
y = df[TARGET].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# convert to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# create train loader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)


class SimpleNN(nn.Module):
    def __init__(self, input_dim, layer_1_size=128, layer_2_size=64):
        super(SimpleNN, self).__init__()
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


model = SimpleNN(input_dim=X.shape[1])

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    for batch_X, batch_y in train_loader:
        preds = model(batch_X)
        loss = loss_fn(preds, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

model.eval()
with torch.no_grad():
    test_preds = model(X_test_tensor)
    test_preds = test_preds.squeeze().numpy()
    y_test_np = y_test_tensor.squeeze().numpy()

mse = mean_squared_error(y_test_np, test_preds)
r2 = r2_score(y_test_np, test_preds)

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R-squared (R²): {r2:.4f}")
