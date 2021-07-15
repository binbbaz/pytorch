import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as dataset
from torchvision.datasets.mnist import MNIST
from torchvision.transforms import transforms
import torch.optim as optim
#Creating a fully Connected Network
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

'''model = NeuralNetwork(784, 10)
input_x = torch.randn(64,784)
print(model(input_x).shape)'''
#set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#Hyper parameters
input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs =  10

#Load Data
transform = transforms.ToTensor()

train_data = dataset.MNIST(root='dataset/', train= True, transform =transform, download=True)
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

test_data = dataset.MNIST(root='dataset/', train= False, transform =transform, download=True)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

# Initialize Network
model = NeuralNetwork(input_size=input_size, num_classes=num_classes).to(device)
 
 
#Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train Network
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):

        #getting data to cuda if available
        data = data.to(device=device)
        targets = targets.to(device=device)

        #flatten the data (converting 2D to a vector)
        #print(data.shape) - torch.Size([64, 1, 28, 28])
        data = data.reshape(data.shape[0], -1)


        score = model(data)
        loss = criterion(score, targets)

        #backward to the network
        optimizer.zero_grad()
        loss.backward() #updating the weight

        #gradient descent or adam step
        optimizer.step()
    print ("Epoch-------> {}|Loss:{}".format(epoch, loss))

#Check accuracy on training and test to see how good our model is
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()
    counter = 1
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            x = x.reshape(x.shape[0], -1)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum() #* 
            num_samples += predictions.size(0)
 
        #print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples) * 100:.2f)}')
        print("Got {}/ {} with accuracy: {}".format(num_correct, num_samples,float(num_correct)/float(num_samples) * 100))
    model.train()

check_accuracy(train_loader, model)
check_accuracy(test_loader, model)