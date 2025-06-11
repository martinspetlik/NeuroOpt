import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# ----- 1. Generate synthetic data -----
torch.manual_seed(0)
n_samples = 10000
X = torch.randn(n_samples, 10)
y = torch.cat([
    X.mean(dim=1, keepdim=True),
    X.std(dim=1, unbiased=False, keepdim=True)
], dim=1)

# ----- 2. Prepare DataLoader -----
batch_size = 64
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# ----- 3. Define the model -----
class FullyConnectedNN(nn.Module):
    def __init__(self, activation_fn=nn.ReLU()):
        super(FullyConnectedNN, self).__init__()
        self.activation = activation_fn
        self.fc1 = nn.Linear(10, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.output = nn.Linear(64, 2)

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.output(x)
        return x


activation = nn.ReLU()
#activation = nn.LeakyReLU()
#activation = nn.Tanh()
model = FullyConnectedNN(activation_fn=activation)

# ----- 4. Define loss and optimizer -----
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ----- 5. Training loop -----
n_epochs = 50
for epoch in range(n_epochs):
    total_loss = 0.0
    for xb, yb in dataloader:
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)

    avg_loss = total_loss / n_samples
    print(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.6f}")


# ----- 6. Testing -----
# Set seed for reproducibility (optional)
torch.manual_seed(42)
n_test_samples = 1000
X_test = torch.randn(n_test_samples, 10)
y_test = torch.cat([
    X_test.mean(dim=1, keepdim=True),
    X_test.std(dim=1, unbiased=False, keepdim=True)
], dim=1)


model.eval()  # Set to evaluation mode (disables dropout etc.)
with torch.no_grad():
    y_pred = model(X_test)
    test_loss = nn.MSELoss()(y_pred, y_test)

    print("y_test[0]: {}, y_pred[0]: {}".format(y_test[0], y_pred[0]))


# ----- 7. Derivatives with respect to inputs -----

# 1 sample with 10 features
x = torch.randn(1, 10, requires_grad=True)
# Forward pass
y = model(x)  # shape: (1, 2)
grad_output = torch.autograd.grad(outputs=y[0, 0], inputs=x, retain_graph=True)[0]
print("Gradient of output[0] w.r.t input:", grad_output)
grad_output = torch.autograd.grad(outputs=y[0, 1], inputs=x, retain_graph=True)[0]
print("Gradient of output[1] w.r.t input:", grad_output)

jacobian = torch.autograd.functional.jacobian(lambda inp: model(inp), x)
jacobian = jacobian.squeeze(0).squeeze(1)  # shape: (2, 10)
print("Jacobian:", jacobian)