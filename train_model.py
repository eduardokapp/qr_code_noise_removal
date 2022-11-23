"""
Training the network:
    1. Use a dataloader to generate batches from the processed dataset
    2. Define the neural network
    3. Train it

@author: eduardokapp
"""

import torch
import torch.nn as nn
import numpy as np
from neural_network import network
import joblib

# input params
folder = 'qr_dataset'
input_size = 5
output_size = 1
batch_size = 408*408 # each image has 408*408 windows
learning_rate=0.001
num_epochs = 50

## Section 1: Defining the dataloader
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = np.load(f'{folder}/processed_dataset/train_dataset.npy')
dataset = dataset.astype(np.float32)

# define tensor datasets
train_dataset = torch.utils.data.TensorDataset(
    torch.from_numpy(dataset[:,0:5]),
    torch.from_numpy(dataset[:, 5])
)

# define dataloaders
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=False
)

## Section 2: Define the network
net = network(input_size, output_size).to(device)

# Loss and optimizer
criterion = nn.functional.mse_loss
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)

n_total_steps = len(train_loader)
# training the network
for epoch in range(num_epochs):
    for i, (features, labels) in enumerate(train_loader):
        X = features.to(device)
        y = labels.to(device).reshape(batch_size, 1)

        # Forward pass
        outputs = net(X)
        loss = criterion(outputs, y.float())
        acc = (torch.round(outputs) ==  y.float()).sum()/batch_size

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 10 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.16f}, Acc: {acc.item():.4f}')


joblib.dump(net, 'model.pkl')
