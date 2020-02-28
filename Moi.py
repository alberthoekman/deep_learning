import pandas as pd
import torch
import time
import numpy as np
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torchvision import transforms
from torchvision import datasets
from typing import Callable, List
import os
from glob import glob
import matplotlib.pyplot as plt
import cv2
import torch.nn as nn
import torch.nn.functional as F
import csv
import torch.backends.cudnn as cudnn


# SOURCE_IMAGES = os.path.abspath('images')
# images = glob(os.path.join(SOURCE_IMAGES, "*.png"))

# labels = pd.read_csv('sample_labels.csv')

# ResNet
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


# ResNext
class Block(nn.Module):
    '''Grouped convolution block.'''
    expansion = 2

    def __init__(self, in_planes, cardinality=32, bottleneck_width=4, stride=1):
        super(Block, self).__init__()
        group_width = cardinality * bottleneck_width
        self.conv1 = nn.Conv2d(in_planes, group_width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(group_width)
        self.conv2 = nn.Conv2d(group_width, group_width, kernel_size=3, stride=stride, padding=1, groups=cardinality,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(group_width)
        self.conv3 = nn.Conv2d(group_width, self.expansion * group_width, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * group_width)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * group_width:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * group_width, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * group_width)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNeXt(nn.Module):
    def __init__(self, num_blocks, cardinality, bottleneck_width, num_classes=10):
        super(ResNeXt, self).__init__()
        self.cardinality = cardinality
        self.bottleneck_width = bottleneck_width
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(num_blocks[0], 1)
        self.layer2 = self._make_layer(num_blocks[1], 2)
        self.layer3 = self._make_layer(num_blocks[2], 2)
        # self.layer4 = self._make_layer(num_blocks[3], 2)
        self.linear = nn.Linear(cardinality * bottleneck_width * 8, num_classes)

    def _make_layer(self, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(Block(self.in_planes, self.cardinality, self.bottleneck_width, stride))
            self.in_planes = Block.expansion * self.cardinality * self.bottleneck_width
        # Increase bottleneck_width by 2 after each stage.
        self.bottleneck_width *= 2
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        # out = self.layer4(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNeXt29_2x64d():
    return ResNeXt(num_blocks=[3, 3, 3], cardinality=2, bottleneck_width=64)


def ResNeXt29_4x64d():
    return ResNeXt(num_blocks=[3, 3, 3], cardinality=4, bottleneck_width=64)


def ResNeXt29_8x64d():
    return ResNeXt(num_blocks=[3, 3, 3], cardinality=8, bottleneck_width=64)


def ResNeXt29_32x4d():
    return ResNeXt(num_blocks=[3, 3, 3], cardinality=32, bottleneck_width=4)


# VGG
cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


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


def run(modelId, epochs, optim):
    DEVICE = 'cuda'

    # Process images and divide in train and test set.
    #     x, y = proc_images()
    #     cutoff = int(len(x) * 0.8)
    #     x_train = x[1:cutoff]
    #     x_val = x[cutoff + 1:]
    #     y_train = y[1:cutoff]
    #     y_val = y[cutoff + 1:]

    #     assert len(x_train) == len(y_train)
    #     assert len(x_val) == len(y_val)

    # Put list of tensors into one tensor and normalise RGB values between 0 and 1.
    #     x_train = torch.stack(x_train) / 255.0
    #     x_val = torch.stack(x_val) / 255.0
    #     y_train = torch.stack(y_train)
    #     y_val = torch.stack(y_val)

    # Dataset and dataloader for pytorch.
    #     train_dataset = TensorDataset(x_train, y_train)
    #     train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    #     val_dataset = TensorDataset(x_val, y_val)
    #     val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = datasets.CIFAR10(root='./cifar', train=True,
                                download=True, transform=transform_train)
    train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                                   shuffle=True)

    testset = datasets.CIFAR10(root='./cifar', train=False,
                               download=True, transform=transform_test)
    val_dataloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                                 shuffle=False)

    # Neural network to use.
    if modelId == 'resnext':
        model = ResNeXt29_2x64d()
    elif modelId == 'resnet':
        model = ResNet18()
    elif modelId == 'vgg':
        model = VGG("VGG16")

    model.to(DEVICE)
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True

    if optim == 'adam':
        opt = torch.optim.Adam(model.parameters(), lr=0.1)
    elif optim == 'adamw':
        opt = torch.optim.Adam(model.parameters(), lr=0.1, weight_decay=0.01)
    elif optim == 'sgd':
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
    elif optim == 'sgdm':
        opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    loss_function = torch.nn.CrossEntropyLoss()

    train_losses = []
    val_losses = []

    train_accs = []
    val_accs = []

    with open(str(modelId) + '_' + str(optim) + '.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Training Loss", "Validation Loss", "Training Accuracy", "Validation Accuracy"])

    for t in range(epochs):
        print('Epoch {}'.format(t))
        print('Time before: {}'.format(time.perf_counter()))
        train_loss, train_acc = train_epoch(model, train_dataloader, optimizer=opt, loss_fn=loss_function,
                                            device=DEVICE)
        val_loss, val_acc = eval_epoch(model, val_dataloader, loss_function, device=DEVICE)

        print('Training Loss: {}'.format(train_loss))
        print('Training Accuracy: {}'.format(train_acc))
        print()
        print('Validation Loss: {}'.format(val_loss))
        print('Validation Accuracy: {}'.format(val_acc))
        print()
        print('Time after: {}'.format(time.perf_counter()))
        print()

        with open(str(modelId) + '_' + str(optim) + '.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([t, train_loss, val_loss, train_acc, val_acc])
    print('============================================================================')
    # print('Training Loss: {}'.format(train_losses[len(train_losses)-1]))
    # print('Validation Loss: {}'.format(val_losses[len(val_losses)-1]))
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
        model.classifier[6] = torch.nn.Linear(inputAmount, 10)
    elif classOrFc == 'fc':
        model.fc = torch.nn.Linear(inputAmount, 10)

    # Extract features that need training.
    for param in model.parameters():
        if param.requires_grad:
            paramsToLearn.append(param)

    return paramsToLearn


# ====================================================================================================
print("vgg adam")
run('vgg', 15, 'adam')

print("vgg adamw")
run('vgg', 15, 'adamw')

print("vgg sgd")
run('vgg', 15, 'sgd')

print("vgg sgdm")
run('vgg', 15, 'sgdm')

print("resnet adam")
run('resnet', 15, 'adam')

print("resnet adamw")
run('resnet', 15, 'adamw')

print("resnet sgd")
run('resnet', 15, 'sgd')

print("resnet sgdm")
run('resnet', 15, 'sgdm')

print("resnext adam")
run('resnext', 15, 'adam')

print("resnext adamw")
run('resnext', 15, 'adamw')

print("resnext sgd")
run('resnext', 15, 'sgd')

print("resnext sgdm")
run('resnext', 15, 'sgdm')