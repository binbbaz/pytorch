import torch
import torch.nn as nn
import torchvision.datasets as dataset 
from torch.utils.data import DataLoader
import torch.nn.functional as F 
import torch.optim as optim
import torchvision.transforms as transform


class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(ConvolutionalNeuralNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels=8, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.conv2 = nn.Conv2d(in_channels = 8, out_channels=16, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.fc1 = nn.Linear(16 * 7 * 7, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x

#save model
def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> saving checkpoint")
    torch.save(state, filename)

#load model
def load_checkpoint(checkpoint):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optim.load_state_dict(checkpoint['optimizer'])

#Setting our device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Hyperparameter
in_channels = 1
num_classes = 10
epochs =10
learning_rate = 0.001
batch_size = 64
load_model = True

#Load data
transform = transform.ToTensor()

train_data = dataset.MNIST(root='dataset/', train=True, transform=transform, download= True)
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True,)

test_data = dataset.MNIST(root='dataset/', train=False, transform=transform, download= True)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

#Initializing Network
model = ConvolutionalNeuralNetwork(in_channels = in_channels, num_classes=num_classes).to(device)
#x = torch.randn(64, 1, 28, 28)
#print(model(x) )

#loss and Optimizer
criterion = nn.CrossEntropyLoss()
optim = optim.Adam(model.parameters(), lr=learning_rate)

if load_model:
    load_checkpoint(torch.load("my_checkpoint.pth.tar"))

#Training our model
for epoch in range(epochs):
    losses = []

    if epoch % 3 == 0:
        checkpoint = {'state_dict': model.state_dict(), 'optimizer': optim.state_dict()}
        save_checkpoint(checkpoint)

    for batch_idx , (data, targets) in enumerate(train_loader):
        data = data.to(device)
        targets = targets.to(device)

        scores = model(data)
        loss = criterion(scores, targets)
        losses.append(loss.item())

        optim.zero_grad()
        loss.backward()

        optim.step()

        model.eval()
    print ("Epoch-------> {}|Loss:{}".format(epoch, loss))


#check accuracy
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()
    counter = 1
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum() #* 
            num_samples += predictions.size(0)
 
        #print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples) * 100:.2f)}')
        print("Got {}/ {} with accuracy: {}".format(num_correct, num_samples,float(num_correct)/float(num_samples) * 100))
    model.train()

if __name__ == "__main__":
    check_accuracy(train_loader, model)
    check_accuracy(test_loader, model)
#I had an accuracy of 89% when I used SGD and 99... when I used Adam
