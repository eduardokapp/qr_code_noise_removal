# Overview
This is a simple project that uses a one layer neural network to estimate
the boolean function that removes Salt and Pepper (S&P) noise from QR code
images, which are binary. It was done for the Machine Learning for
Computer Vision course @ the São Paulo University (MAC-6914).


# Running the project

1. The QR codes are generated randomly in `create_dataset.py`.
2. The training image pairs are then transformed into a tabular dataset in
`preprocess_dataset.py`.
3. The neural network architecture is defined in `neural_network.py`.
4. Training is done in `train_model.py`
5. Finally, results and image predictions are done in `generate_results.py`


# Requirements
* numpy
* skimage
* pytorch
