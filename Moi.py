import pandas as pd
import torch
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


def run(modelId, epochs, preTrained, featureExtract):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        paramsToLearn = featureExtract(model, featureExtract, 'class', 4096)
    elif modelId == 'resnet18':
        model = torch.hub.load('pytorch/vision:v0.5.0', 'resnet18', pretrained=preTrained)
        paramsToLearn = featureExtract(model, featureExtract, 'fc', 512)
    elif modelId == 'resnet152':
        model = torch.hub.load('pytorch/vision:v0.5.0', 'resnet152', pretrained=preTrained)
        paramsToLearn = featureExtract(model, featureExtract, 'fc', 512)
    elif modelId == 'vgg':
        model = torch.hub.load('pytorch/vision:v0.5.0', 'vgg19', pretrained=preTrained)
        paramsToLearn = featureExtract(model, featureExtract, 'class', 4096)
    elif modelId == 'vgg_bn':
        model = torch.hub.load('pytorch/vision:v0.5.0', 'vgg19_bn', pretrained=preTrained)
        paramsToLearn = featureExtract(model, featureExtract, 'class', 4096)

    # Optimizer and loss function
    opt = torch.optim.Adam(paramsToLearn, lr=1e-05)
    loss_function = torch.nn.BCEWithLogitsLoss(reduction='mean')

    train_losses = []
    val_losses = []

    for t in range(epochs):
        train_loss = train_epoch(model, train_dataloader, optimizer=opt, loss_fn=loss_function, device=DEVICE)
        val_loss = eval_epoch(model, val_dataloader, loss_function, device=DEVICE)

        print('Epoch {}'.format(t))
        print(' Training Loss: {}'.format(train_loss))
        print(' Validation Loss: {}'.format(val_loss))

        train_losses.append(train_loss)
        val_losses.append(val_loss)

    plt.plot(train_losses)
    plt.plot(val_losses)
    plt.legend(['Training', 'Validation'])
    plt.show()


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
run('resnet18', 1, True, False)
