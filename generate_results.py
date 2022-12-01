"""
Testing script. It takes the test images, generates output images
and compare them with the "truth" images.

@author: eduardokapp
"""
import os
import numpy as np
import torch
import joblib
import skimage
from skimage.util import view_as_windows


# load the model in mem
model = joblib.load('model.pkl')
folder = 'qr_dataset'
batch_size = 100

# first, we'll simply evaluate performance on the tabular test dataset
# then, we'll generate some output images to do some visual inspection.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = np.load(f'{folder}/processed_dataset/test_dataset.npy')
dataset = dataset.astype(np.float32)

# define tensor datasets
test_dataset = torch.utils.data.TensorDataset(
    torch.from_numpy(dataset[:,0:5]),
    torch.from_numpy(dataset[:, 5])
)

# define dataloaders
test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False
)

# test on tabular test set
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for features, labels in test_loader:
        features = features.to(device)
        labels = labels.to(device)
        outputs = model(features)
        # max returns (value ,index)
        predicted = outputs.data
        actuals = labels

        n_samples += labels.size(0)
        n_correct += ((predicted > 0.5) == actuals.float()).sum().item()
    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network on the {n_correct/n_samples} test samples: {acc} %')


# now to generate output images, we'll read each noisy img
# pad it with zeros and for each window we'll generate an output with
# the model

noisy_imgs = os.listdir(f'{folder}/test/noisy')
os.makedirs(f'{folder}/test/preds', exist_ok=True)

for img_path in noisy_imgs:
    if 'DS_Store' in img_path:
        continue
    print(f'Predicting {img_path}...')
    img = skimage.io.imread(f'{folder}/test/noisy/{img_path}').astype(np.float32)
    img[img > 0] = 1

    # initialize a new img with same size
    predicted_img = np.zeros(img.shape).astype(np.float32)

    # pad img with zeros
    img = np.pad(img, (1, 1))
    windows = view_as_windows(img, (3,3)).reshape((410**2, 3**2))
    windows = windows[:, [1, 3, 4, 5, 7]]

    x = 0
    y = 0
    for window in windows:
        features = torch.from_numpy(window).to(device)
        with torch.no_grad():
            out = np.float32(model(features).item() > 0.5)
        predicted_img[x, y] = out
        if y < (predicted_img.shape[0] - 1):
            y += 1
        else:
            y = 0
            x += 1
    predicted_img[predicted_img > 0] = 255
    skimage.io.imsave(f'{folder}/test/preds/{img_path}', predicted_img.astype(np.uint8))
