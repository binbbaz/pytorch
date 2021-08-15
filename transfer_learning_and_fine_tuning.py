#imports
import torch
import torch.nn
from torch.nn.modules import batchnorm
import torch.optim as optim
import torchvision.datasets as dataset
import torchvision.transforms as transform
from torch.utils.data import DataLoader
import torchvision.models as mod
import torch.nn as nn
import sys


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


#Set device
device = torch.device('gpu' if torch.cuda.is_available() else 'cpu')

#hyperparameters
in_channel = 3
num_classes = 10
learning_rate = 0.001
batch_size = 1024
num_epochs = 5

model = mod.vgg16(pretrained=True)
#print(model)
model.avgpool = Identity()
model.classifier = nn.Linear(512, 10)
model.to(device)

# we can also do the following if we want to change some specific layers of the fully connected
'''model.classifier = nn.Sequential(nn.Linear(512, 100),
                                    nn.Dropout(p=0.5), can also add whatever techniques you want inbetween
                                    nn.Linear(100, 10))'''
#print(model)

transforms = transform.ToTensor()
train_dataset = dataset.CIFAR10(root="dataset/", train=True, transform=transforms, download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = dataset.CIFAR10(root="dataset/", train=False, transform=transforms, download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

criterion = nn.CrossEntropyLoss()
optim = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for batch_idx, (data, labels) in enumerate(train_loader):
        data = data.to(device)
        labels = labels.to(device)

        score = model(data)

        loss = criterion(score, labels)

        optim.zero_grad()

        loss.backward()
        optim.step()

def evaluate_model(loader, model):
    num_correct = 0
    num_samples = 0

    with torch.no_grad():
        for data, target in loader:
            data = data.to(device)
            targets = targets.to(device)

            score = model(data)
            _, prediction = score.max(1)
            num_correct += (prediction == targets).sum()

            print(f'Got {num_correct} / {num_samples} with accuracy \
                {float(num_correct)/float(num_samples)*100:.2f}')

    model.train()

if __name__ == '__main__':
    evaluate_model(train_loader)
    evaluate_model(test_loader)

 