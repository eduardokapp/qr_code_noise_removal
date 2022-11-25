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
batch_size = 100
learning_rate=0.001
num_epochs = 25

## Section 1: Defining the dataloader
# Device configuration
device = torch.device('cpu')

dataset = np.load(f'{folder}/processed_dataset/train_dataset.npy')
dataset = dataset.astype(np.float32)

# define tensor datasets
train_dataset = torch.utils.data.TensorDataset(
    torch.from_numpy(dataset[:,0:5]),
    torch.from_numpy(dataset[:, 5]).bool()
)

# define dataloaders
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True
)

## Section 2: Define the network
net = network(input_size, output_size).to(device)

# Loss and optimizer
criterion = nn.functional.binary_cross_entropy
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)

n_total_steps = len(train_loader)
# training the network
for epoch in range(num_epochs):
    for i, (features, labels) in enumerate(train_loader):
        X = features.to(device)
        y = labels.to(device).reshape(-1, output_size)
        # Forward pass
        outputs = net(X)
        loss = criterion(outputs, y.float())
        #acc = ((outputs[:,1] > 0.5) ==  y.bool()).sum()/batch_size

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.16f}')


net = net.to(torch.device('cpu'))
joblib.dump(net, 'model.pkl')
