"""
Generate QR code images and their noisy counterpart by adding salt and pepper
noise.

@author: eduardokapp
"""

import os
import skimage.io as img_io
from skimage import img_as_uint
from skimage.util import random_noise
import numpy as np
import qrcode

folder = 'qr_dataset'

os.makedirs(folder, exist_ok=True)

for img_type in ['train', 'test']:
    os.makedirs(f'{folder}/{img_type}/pure', exist_ok=True)
    os.makedirs(f'{folder}/{img_type}/noisy', exist_ok=True)
    for data in np.random.choice(np.arange(1000, 10000), size=50, replace=False):
        # generate QR code object
        # using this version means its a (410, 410) img.
        qr = qrcode.QRCode(
            version=4,
            error_correction=qrcode.constants.ERROR_CORRECT_M,
            box_size=10,
            border=4
        )
        qr.add_data(data)
        # generate QR code final data
        qr.make(fit=True)

        # make it to a binary image
        img = qr.make_image(fill_color='black', back_color='white')
        img = np.asarray(img).astype(np.float32)

        # generate noisy version with salt and pepper noise
        noisy_img = random_noise(img, mode='s&p', seed=data)

        # convert both images to the range [0, 255]
        img[img>0] = 255
        noisy_img[noisy_img>0] = 255

        # write pure image to path
        img_io.imsave(f"{folder}/{img_type}/pure/img_{data}.png", img.astype(np.uint8))

        # write noisy_img
        img_io.imsave(f"{folder}/{img_type}/noisy/img_{data}.png", noisy_img.astype(np.uint8))

print("Done.")
