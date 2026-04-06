import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model import CNN1D
from utils import load_data

# Load data
(X_train, X_test, y_train, y_test), encoder = load_data('../data/Dataset.csv')

train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                              torch.tensor(y_train, dtype=torch.long))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Model
model = CNN1D(num_classes=len(encoder.classes_), input_size=X_train.shape[2])

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(20):
    model.train()
    total_loss = 0

    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()

        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# Save model
torch.save(model.state_dict(), '../models/model.pth')
print("Model saved!")