import pandas as pd
import torch
import numpy as np
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torchvision import transforms
from typing import Callable, List
import os
from glob import glob
import matplotlib.pyplot as plt
import cv2

SOURCE_IMAGES = os.path.abspath('images')
images = glob(os.path.join(SOURCE_IMAGES, "*.png"))

labels = pd.read_csv('sample_labels.csv')


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

    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    for img in images:
        base = os.path.basename(img)
        # Read and resize image
        full_size_image = cv2.imread(img)
        resized = cv2.resize(full_size_image, (256, 256), interpolation=cv2.INTER_CUBIC)
        findingString = labels["Finding Labels"][labels["Image Index"] == base].values[0]
        symbol = "|"

        if symbol in findingString:
            continue
        else:
            x.append(preprocess(resized))

            if NoFinding in findingString:
                finding = torch.tensor(0, dtype=torch.long)
            elif Consolidation in findingString:
                finding = torch.tensor(1, dtype=torch.long)
            elif Infiltration in findingString:
                finding = torch.tensor(2, dtype=torch.long)
            elif Pneumothorax in findingString:
                finding = torch.tensor(3, dtype=torch.long)
            elif Edema in findingString:
                finding = torch.tensor(7, dtype=torch.long)
            elif Emphysema in findingString:
                finding = torch.tensor(7, dtype=torch.long)
            elif Fibrosis in findingString:
                finding = torch.tensor(7, dtype=torch.long)
            elif Effusion in findingString:
                finding = torch.tensor(4, dtype=torch.long)
            elif Pneumonia in findingString:
                finding = torch.tensor(7, dtype=torch.long)
            elif Pleural_Thickening in findingString:
                finding = torch.tensor(7, dtype=torch.long)
            elif Cardiomegaly in findingString:
                finding = torch.tensor(7, dtype=torch.long)
            elif NoduleMass in findingString:
                finding = torch.tensor(5, dtype=torch.long)
            elif Hernia in findingString:
                finding = torch.tensor(7, dtype=torch.long)
            elif Atelectasis in findingString:
                finding = torch.tensor(6, dtype=torch.long)
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
                optimizer: Optimizer):
    network.train()

    prediction_batch = network.forward(X_batch)  # forward pass
    _, preds = torch.max(prediction_batch, 1)
    batch_loss = loss_fn(prediction_batch, Y_batch)  # loss calculation
    batch_correct = (Y_batch.eq(preds.long())).double().sum().item()
    batch_loss.backward()  # gradient computation
    optimizer.step()  # back-propagation
    optimizer.zero_grad()  # gradient reset

    return batch_loss.item(), batch_correct, np.float(preds.size(0))


def train_epoch(network: torch.nn.Module,
                # a list of data points x
                dataloader: DataLoader,
                loss_fn: Callable[[torch.FloatTensor, torch.FloatTensor], torch.FloatTensor],
                optimizer: Optimizer,
                device: str):
    loss = 0.
    correct = 0
    total = 0

    for i, (x_batch, y_batch) in enumerate(dataloader):
        x_batch = x_batch.to(device)  # convert back to your chosen device
        y_batch = y_batch.to(device)
        loss_batch, correct_batch, total_batch = train_batch(network=network, X_batch=x_batch, Y_batch=y_batch,
                                                             loss_fn=loss_fn, optimizer=optimizer)
        loss += loss_batch
        correct += correct_batch
        total += total_batch

    loss /= (i + 1)  # divide loss by number of batches for consistency

    return loss, correct / total


def eval_batch(network: torch.nn.Module,  # the network
               X_batch: torch.FloatTensor,  # the X batch
               Y_batch: torch.FloatTensor,  # the Y batch
               loss_fn: Callable[[torch.FloatTensor, torch.FloatTensor], torch.FloatTensor]):
    network.eval()

    with torch.no_grad():
        prediction_batch = network.forward(X_batch)  # forward pass
        _, preds = torch.max(prediction_batch, 1)
        batch_loss = loss_fn(prediction_batch, Y_batch)  # loss calculation
        batch_correct = (Y_batch.eq(preds.long())).double().sum().item()

    return batch_loss.item(), batch_correct, np.float(preds.size(0))


def eval_epoch(network: torch.nn.Module,
               # a list of data points x
               dataloader: DataLoader,
               loss_fn: Callable[[torch.FloatTensor, torch.FloatTensor], torch.FloatTensor],
               device: str):
    loss = 0.
    correct = 0
    total = 0

    for i, (x_batch, y_batch) in enumerate(dataloader):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        loss_batch, correct_batch, total_batch = eval_batch(network=network, X_batch=x_batch, Y_batch=y_batch,
                                                            loss_fn=loss_fn)
        loss += loss_batch
        correct += correct_batch
        total += total_batch

    loss /= (i + 1)

    return loss, correct / total


def run(modelId, epochs, preTrained, featureExtract, optim):
    DEVICE = 'cuda'

    # Process images and divide in train and test set.
    x, y = proc_images()
    cutoff = int(len(x) * 0.8)
    x_train = x[1:cutoff]
    x_val = x[cutoff + 1:]
    y_train = y[1:cutoff]
    y_val = y[cutoff + 1:]

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

    # Neural network to use.
    if modelId == 'alexnet':
        model = torch.hub.load('pytorch/vision:v0.5.0', 'alexnet', pretrained=preTrained)
        paramsToLearn = feature_extract(model, featureExtract, 'class', 4096)
    elif modelId == 'resnet18':
        model = torch.hub.load('pytorch/vision:v0.5.0', 'resnet18', pretrained=preTrained)
        paramsToLearn = feature_extract(model, featureExtract, 'fc', 512)
    elif modelId == 'resnet152':
        model = torch.hub.load('pytorch/vision:v0.5.0', 'resnet152', pretrained=preTrained)
        paramsToLearn = feature_extract(model, featureExtract, 'fc', 2048)
    elif modelId == 'vgg':
        model = torch.hub.load('pytorch/vision:v0.5.0', 'vgg19', pretrained=preTrained)
        paramsToLearn = feature_extract(model, featureExtract, 'class', 4096)
    elif modelId == 'vgg_bn':
        model = torch.hub.load('pytorch/vision:v0.5.0', 'vgg19_bn', pretrained=preTrained)
        paramsToLearn = feature_extract(model, featureExtract, 'class', 4096)

    model.to(DEVICE)

    if optim == 'adam':
        opt = torch.optim.Adam(paramsToLearn)
    elif optim == 'adamw':
        opt = torch.optim.Adam(paramsToLearn, weight_decay=0.01)
    elif optim == 'sgd':
        opt = torch.optim.SGD(paramsToLearn)
    elif optim == 'sgdm':
        opt = torch.optim.SGD(paramsToLearn, lr=0.01, momentum=0.9)

    loss_function = torch.nn.CrossEntropyLoss()

    train_losses = []
    val_losses = []

    train_accs = []
    val_accs = []

    for t in range(epochs):
        train_loss, train_acc = train_epoch(model, train_dataloader, optimizer=opt, loss_fn=loss_function,
                                            device=DEVICE)
        val_loss, val_acc = eval_epoch(model, val_dataloader, loss_function, device=DEVICE)

        print('Epoch {}'.format(t))
        print('Training Loss: {}'.format(train_loss))
        print('Training Accuracy: {}'.format(train_acc))
        print('Validation Loss: {}'.format(val_loss))
        print('Validation Accuracy: {}'.format(val_acc))

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

    print('Training Loss: {}'.format(train_losses[len(train_losses) - 1]))
    print('Validation Loss: {}'.format(val_losses[len(val_losses) - 1]))
    # plt.plot(train_losses)
    # plt.plot(val_losses)
    # plt.legend(['Training', 'Validation'])
    # plt.show()


def feature_extract(model, featureExtract, classOrFc, inputAmount):
    paramsToLearn = []

    # Set all features to not require training.
    if featureExtract:
        for param in model.parameters():
            param.requires_grad = False

    # Re-add the last layer (features of a new layer are set to requires_grad = True by default).
    if classOrFc == 'class':
        model.classifier[6] = torch.nn.Linear(inputAmount, 8)
    elif classOrFc == 'fc':
        model.fc = torch.nn.Linear(inputAmount, 8)

    # Extract features that need training.
    for param in model.parameters():
        if param.requires_grad:
            paramsToLearn.append(param)

    return paramsToLearn


# ====================================================================================================
print("vgg pre-trained")
# run('vgg', 100, True, False, 'adam')

print("vgg")
# run('vgg', 100, False, False, 'adam')

print("vgg_bn pre-trained")
# run('vgg_bn', 100, True, False, 'adam')

print("vgg_bn")
# run('vgg_bn', 100, False, False, 'adam')

print("alexnet pre-trained")
run('alexnet', 100, True, False, 'adam')

print("alexnet")
run('alexnet', 100, False, False, 'adam')

print("resnet18 pre-trained")
run('resnet18', 100, True, False, 'adam')

print("resnet18")
run('resnet18', 100, False, False, 'adam')

print("resnet152 pre-trianed adam")
run('resnet152', 100, True, False, 'adam')

print("resnet152 adam")
run('resnet152', 100, False, False, 'adam')

print("resnet152 adamW")
run('resnet152', 100, True, False, 'adamw')

print("resnet152 SGD")
run('resnet152', 100, True, False, 'sgd')

print("resnet152 SGDm")
run('resnet152', 100, True, False, 'sgdm')