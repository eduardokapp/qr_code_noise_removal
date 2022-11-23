"""
Transform the train images into a tabular dataset that takes every
3x3 crux window and stretch it to a (5,1) array and use it as inputs.

The labels or outputs will be the center pixel in the target image (pure)
respective window.

@author: eduardokapp
"""

import os
import numpy as np
import skimage.io as img_io
from skimage.util import view_as_windows

folder = 'qr_dataset'

# create new folder for the processed dataset
os.makedirs(f'{folder}/processed_dataset', exist_ok=True)

n_images = 50

for dataset_type in ['train', 'test']:
    # we'll do things in a loop, so as not to overwhelm memory usage
    pure_imgs = os.listdir(f'{folder}/{dataset_type}/pure')
    noisy_imgs = os.listdir(f'{folder}/{dataset_type}/noisy')

    # the dataset number of observations is going to be:
    # n_images (50) * n_windows (408*408), with 6 columns (5 for the window stretched and 1 for the label)
    dataset = np.zeros((166464*n_images, 6))
    idx = 0
    for pure, noisy in zip(pure_imgs, noisy_imgs):
        if 'DS_Store' in pure:
            continue
        # read imgs
        pure_img = img_io.imread(f'{folder}/{dataset_type}/pure/{pure}').astype(np.float32)
        pure_img[pure_img > 0] = 1
        noisy_img = img_io.imread(f'{folder}/{dataset_type}/noisy/{noisy}').astype(np.float32)
        noisy_img[noisy_img > 0] = 1

        # for each pair, we'll slide a 3x3 window on the noisy img
        # and find the center pixel in the pure img
        features = view_as_windows(noisy_img, (3,3)).reshape((408**2, 3**2))
        # we want a 3x3 crux, so we only select some of the image
        features = features[:, [1, 3, 4, 5, 7]]

        # the labels are only the center pixels in each 3x3 window
        labels = view_as_windows(pure_img, (3,3)).reshape((408**2, 3**2))[:, 4]

        dataset[idx:(idx+166464),:] = np.column_stack((features, labels))
        idx += 1

    # write dataset to file
    np.save(file=f'{folder}/processed_dataset/{dataset_type}_dataset', arr=dataset)
