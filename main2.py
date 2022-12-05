'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import ipdb
import wandb
import argparse
import numpy as np

from models import *
from utils import progress_bar, process_csv2

# CUDA_LAUNCH_BLOCKING="1"

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--run_name', default="cifar10-resnet18-ce-1", type=str, help='name of the run')
args = parser.parse_args()

wandb.init(project="CS839-Project")
#name the wandb run
wandb.run.name = args.run_name



device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
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

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=100, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)
classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

class_vectors = process_csv2("./vectors.csv")
dim = len(class_vectors["plane"])

# Model
print('==> Building model..')
# net = VGG('VGG19')
# net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
net = ResNet50(num_classes=dim)
net.linear = nn.Sequential(nn.Linear(512,1024), nn.ReLU(inplace=True), nn.Linear(1024,2059))
# ipdb.set_trace()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
# net = SimpleDLA()
net = net.to(device)
wandb.watch(net)
print(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.BCELoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    references = torch.tensor(np.array(list(class_vectors.values())).astype(float)).to(device)
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets_base = inputs.to(device), targets.to(device)
        # set target as a vector of vector of dim elements corresponding to the class_vectors of each class
        targets = torch.tensor([class_vectors[classes[c]] for c in targets_base]).to(device)
        optimizer.zero_grad()
        outputs = net(inputs).sigmoid()
        loss = criterion(outputs, targets.to(torch.float32))
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        # predict output as the class of the closest vector (using cosine_similarity) in class_vectors
        # ipdb.set_trace()
        # predicted = torch.argmax(F.cosine_similarity(outputs, references, dim=1), 1)
        predicted = torch.Tensor([torch.argmax(F.cosine_similarity(outputs[j], torch.Tensor(np.array([references[i].cpu().numpy() for i in range(10)])).to(device), dim = 0)) for j in range(len(outputs))]).to(device) 
        # _, predicted = outputs.max(1)
        total += targets_base.size(0)
        correct += predicted.eq(targets_base).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        # Add Logging using WANDB
        wandb.log({"train_loss": train_loss/(batch_idx+1), "train_acc": 100.*correct/total})


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    references = torch.tensor(np.array(list(class_vectors.values())).astype(float)).to(device)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets_base = inputs.to(device), targets.to(device)
            targets = torch.tensor([class_vectors[classes[c]] for c in targets_base]).to(device)
            outputs = net(inputs).sigmoid()
            loss = criterion(outputs, targets.to(torch.float32))

            test_loss += loss.item()
            predicted = torch.Tensor([torch.argmax(F.cosine_similarity(outputs[j], torch.Tensor(np.array([references[i].cpu().numpy() for i in range(10)])).to(device), dim = 0)) for j in range(len(outputs))]).to(device) 
            total += targets_base.size(0)
            correct += predicted.eq(targets_base).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        # Add Logging using WANDB
        wandb.log({"test_loss": test_loss/(batch_idx+1), "test_acc": 100.*correct/total})
    

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc

for epoch in range(start_epoch, start_epoch+20):
    # Add Logging using WANDB
    wandb.log({"epoch": epoch})
    train(epoch)
    test(epoch)
    scheduler.step()
