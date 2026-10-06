"""
TP1 - Exercise 3: XOR with PyTorch
Beginner-friendly training example.
"""

import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)

# 1) Dataset
X = torch.tensor([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
], dtype=torch.float32)

y = torch.tensor([
    [0.0],
    [1.0],
    [1.0],
    [0.0],
], dtype=torch.float32)

# 2) Neural network
class XORNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        # 2 inputs -> 4 hidden neurons
        self.hidden = nn.Linear(2, 4)

        # 4 hidden neurons -> 1 output
        self.output = nn.Linear(4, 1)

    def forward(self, x):
        x = torch.tanh(self.hidden(x))
        x = torch.sigmoid(self.output(x))
        return x

model = XORNetwork()

# 3) Loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.05)

# 4) Training
for epoch in range(2000):
    # Forward propagation
    predictions = model(X)

    # Calculate error / loss
    loss = criterion(predictions, y)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()

    # Update weights and biases
    optimizer.step()

    if (epoch + 1) % 500 == 0:
        print(f"Epoch {epoch + 1}: loss = {loss.item():.6f}")

# 5) Final predictions
with torch.no_grad():
    predictions = model(X)

print("\nPyTorch predictions:")
for sample, prediction in zip(X, predictions):
    print(f"{sample.tolist()} -> {prediction.item():.4f}")

print("\nRounded predictions:")
print((predictions > 0.5).int().view(-1).tolist())
