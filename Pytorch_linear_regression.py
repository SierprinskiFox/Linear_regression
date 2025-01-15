import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score

# Load and preprocess the data
df = pd.read_csv('../mongodb_data.csv')
df = df[df['Price Increase Ratio'] != 'Placeholder']
df['Price Increase Ratio'] = pd.to_numeric(df['Price Increase Ratio'], errors='coerce')
df['Price Increase Ratio'].fillna(df['Price Increase Ratio'].mean(), inplace=True)

data = df[['Sentiment', 'Views', 'Likes', 'Price Increase Ratio']]
X = data[['Sentiment', 'Views', 'Likes']].values
y = data['Price Increase Ratio'].values.astype(np.float32).reshape(-1, 1)

# Split and scale the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Check for NaN or Inf values
X_train = np.nan_to_num(X_train)
y_train = np.nan_to_num(y_train)
X_test = np.nan_to_num(X_test)
y_test = np.nan_to_num(y_test)

# Convert data to PyTorch tensors
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.FloatTensor(y_train)
y_test = torch.FloatTensor(y_test)

# Define PyTorch Dataset
class RegressionDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

# Create Dataloaders
train_dataset = RegressionDataset(X_train, y_train)
test_dataset = RegressionDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the Linear Regression Model
class LinearRegressionModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)

# Initialize the model, loss function, and optimizer
model = LinearRegressionModel(input_size=3, output_size=1)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0

    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()

        # Forward pass
        predictions = model(batch_X)

        # Compute loss
        loss = criterion(predictions, batch_y)
        epoch_loss += loss.item()

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

    scheduler.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(train_loader):.4f}")

# Evaluate the model
model.eval()
y_true = []
y_pred = []

with torch.no_grad():
    for batch_X, batch_y in test_loader:
        predictions = model(batch_X)
        y_true.append(batch_y.numpy())
        y_pred.append(predictions.numpy())

# Flatten lists
y_true = np.vstack(y_true)
y_pred = np.vstack(y_pred)

# Calculate and print evaluation metrics
test_loss = criterion(torch.FloatTensor(y_pred), torch.FloatTensor(y_true)).item()
r2 = r2_score(y_true, y_pred)

print(f"Test Loss: {test_loss:.4f}")
print(f"R² Score: {r2:.4f}")
