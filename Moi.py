import pandas as pd
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from typing import Callable, List
import numpy as np
import os
from glob import glob
import random
import matplotlib as plt
import cv2
import matplotlib.gridspec as gridspec
import seaborn as sns
import zlib
import itertools
import sklearn
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix

PATH = os.path.abspath('data')
SOURCE_IMAGES = os.path.join(PATH, "images")
images = glob(os.path.join(SOURCE_IMAGES, "*.png"))

labels = pd.read_csv('data/sample_labels.csv')


def proc_images():
    """
    Returns two arrays:
        x is an array of resized images
        y is an array of labels
    """
    NoFinding = "No Finding"  # 0
    Consolidation = "Consolidation"  # 1
    Infiltration = "Infiltration"  # 2
    Pneumothorax = "Pneumothorax"  # 3
    Edema = "Edema"  # 7
    Emphysema = "Emphysema"  # 7
    Fibrosis = "Fibrosis"  # 7
    Effusion = "Effusion"  # 4
    Pneumonia = "Pneumonia"  # 7
    Pleural_Thickening = "Pleural_Thickening"  # 7
    Cardiomegaly = "Cardiomegaly"  # 7
    NoduleMass = "Nodule"  # 5
    Hernia = "Hernia"  # 7
    Atelectasis = "Atelectasis"  # 6
    RareClass = ["Edema", "Emphysema", "Fibrosis", "Pneumonia", "Pleural_Thickening", "Cardiomegaly", "Hernia"]
    x = []  # images as arrays
    y = []  # labels
    WIDTH = 128
    HEIGHT = 128
    for i in range(10):
        img = images[i]
        base = os.path.basename(img)
        # Read and resize image
        full_size_image = cv2.imread(img)
        findingString = labels["Finding Labels"][labels["Image Index"] == base].values[0]
        symbol = "|"

        if symbol in findingString:
            continue
        else:
            resized = cv2.resize(full_size_image, (WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC).flatten()
            x.append(torch.tensor(resized, dtype=torch.float))

            if NoFinding in findingString:
                finding = torch.tensor([1, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float)
            elif Consolidation in findingString:
                finding = torch.tensor([0, 1, 0, 0, 0, 0, 0, 0], dtype=torch.float)
            elif Infiltration in findingString:
                finding = torch.tensor([0, 0, 1, 0, 0, 0, 0, 0], dtype=torch.float)
            elif Pneumothorax in findingString:
                finding = torch.tensor([0, 0, 0, 1, 0, 0, 0, 0], dtype=torch.float)
            elif Edema in findingString:
                finding = torch.tensor([0, 0, 0, 0, 0, 0, 0, 1], dtype=torch.float)
            elif Emphysema in findingString:
                finding = torch.tensor([0, 0, 0, 0, 0, 0, 0, 1], dtype=torch.float)
            elif Fibrosis in findingString:
                finding = torch.tensor([0, 0, 0, 0, 0, 0, 0, 1], dtype=torch.float)
            elif Effusion in findingString:
                finding = torch.tensor([0, 0, 0, 0, 1, 0, 0, 0], dtype=torch.float)
            elif Pneumonia in findingString:
                finding = torch.tensor([0, 0, 0, 0, 0, 0, 0, 1], dtype=torch.float)
            elif Pleural_Thickening in findingString:
                finding = torch.tensor([0, 0, 0, 0, 0, 0, 0, 1], dtype=torch.float)
            elif Cardiomegaly in findingString:
                finding = torch.tensor([0, 0, 0, 0, 0, 0, 0, 1], dtype=torch.float)
            elif NoduleMass in findingString:
                finding = torch.tensor([0, 0, 0, 0, 0, 1, 0, 0], dtype=torch.float)
            elif Hernia in findingString:
                finding = torch.tensor([0, 0, 0, 0, 0, 0, 0, 1], dtype=torch.float)
            elif Atelectasis in findingString:
                finding = torch.tensor([0, 0, 0, 0, 0, 0, 1, 0], dtype=torch.float)
            else:
                del x[-1]
                continue

            y.append(finding)
    return x, y


def train_batch(network: torch.nn.Module,  # the network
                X_batch: torch.FloatTensor,  # the X batch
                Y_batch: torch.FloatTensor,  # the Y batch
                # a function from a FloatTensor (prediction) and a FloatTensor (Y) to a FloatTensor (the loss)
                loss_fn: Callable[[torch.FloatTensor, torch.FloatTensor], torch.FloatTensor],
                # the optimizer
                optimizer: Optimizer) -> float:
    network.train()

    prediction_batch = network(X_batch)  # forward pass
    batch_loss = loss_fn(prediction_batch, Y_batch)  # loss calculation
    batch_loss.backward()  # gradient computation
    optimizer.step()  # back-propagation
    optimizer.zero_grad()  # gradient reset

    return batch_loss.item()


def train_epoch(network: torch.nn.Module,
                # a list of data points x
                dataloader: DataLoader,
                loss_fn: Callable[[torch.FloatTensor, torch.FloatTensor], torch.FloatTensor],
                optimizer: Optimizer,
                device: str) -> float:
    loss = 0.

    for i, (x_batch, y_batch) in enumerate(dataloader):
        x_batch = x_batch.to(device)  # convert back to your chosen device
        y_batch = y_batch.to(device)
        loss += train_batch(network=network, X_batch=x_batch, Y_batch=y_batch, loss_fn=loss_fn, optimizer=optimizer)

    loss /= (i + 1)  # divide loss by number of batches for consistency

    return loss


def eval_batch(network: torch.nn.Module,  # the network
               X_batch: torch.FloatTensor,  # the X batch
               Y_batch: torch.FloatTensor,  # the Y batch
               loss_fn: Callable[[torch.FloatTensor, torch.FloatTensor], torch.FloatTensor]) -> float:
    network.eval()

    with torch.no_grad():
        prediction_batch = network(X_batch)  # forward pass
        batch_loss = loss_fn(prediction_batch, Y_batch)  # loss calculation

    return batch_loss.item()


def eval_epoch(network: torch.nn.Module,
               # a list of data points x
               dataloader: DataLoader,
               loss_fn: Callable[[torch.FloatTensor, torch.FloatTensor], torch.FloatTensor],
               device: str) -> float:
    loss = 0.

    for i, (x_batch, y_batch) in enumerate(dataloader):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        loss += eval_batch(network=network, X_batch=x_batch, Y_batch=y_batch, loss_fn=loss_fn)

    loss /= (i + 1)

    return loss


# ====================================================================================================
DEVICE = 'cpu'
NUM_EPOCHS = 1

# Process images and divide in train and test set.
x, y = proc_images()
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2)

assert len(x_train) == len(y_train)
assert len(x_val) == len(y_val)

# Put list of tensors into one tensor and normalise RGB values between 0 and 1.
x_train = torch.stack(x_train) / 255.0
x_val = torch.stack(x_val) / 255.0
y_train = torch.stack(y_train)
y_val = torch.stack(y_val)

# Dataset and dataloader for pytorch.
train_dataset = TensorDataset(x_train, y_train)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataset = TensorDataset(x_val, y_val)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Our neural network with 1 hidden layer.
f = torch.nn.Sequential(
    torch.nn.Linear(in_features=49152, out_features=24576),
    torch.nn.Sigmoid(),
    torch.nn.Linear(in_features=24576, out_features=8),
).to(DEVICE)

# Optimizer and loss function
opt = torch.optim.Adam(f.parameters(), lr=1e-05)
loss_function = torch.nn.BCEWithLogitsLoss(reduction='mean')

train_losses = []
val_losses = []

for t in range(NUM_EPOCHS):
    train_loss = train_epoch(f, train_dataloader, optimizer=opt, loss_fn=loss_function, device=DEVICE)
    val_loss = eval_epoch(f, val_dataloader, loss_function, device=DEVICE)

    print('Epoch {}'.format(t))
    print(' Training Loss: {}'.format(train_loss))
    print(' Validation Loss: {}'.format(val_loss))

    train_losses.append(train_loss)
    val_losses.append(val_loss)

plt.plot(train_losses)
plt.plot(val_losses)
plt.legend(['Training', 'Validation'])
plt.show()
