# https://www.analyticsvidhya.com/blog/2019/10/building-image-classification-models-cnn-pytorch/
# https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html

import sys
import torch
import torch.nn as nn
import torch.optim as optim

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        
        self.cnn_layers = nn.Sequential(
                nn.Conv2d(3,32,kernel_size=3,stride=1,padding=1), # input and output channel
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2,stride=2),
                nn.Conv2d(32,16,kernel_size=3,stride=1,padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2,stride=2),
                )

        self.linear_layers = nn.Sequential(
                nn.Linear(32*32,10)
                )

    def forward(self,x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0),-1)
        linear_input_shape = x.view(-1,x.shape[-1]).shape[0]
        print("Linear input shape",linear_input_shape)
        x = self.linear_layers(x)
        return x


def train(model):
    
    tr_loss = 0

    x_train, y_train = torch.randn(1000,3,32,32,requires_grad=True), torch.randn(1000,dtype=torch.long) 
    
    if torch.cuda.is_available():
        x_train = x_train.cuda()
        y_train = y_train.cuda()
    
    optimizer = optim.Adam(model.parameters(),lr=0.07)
    criterion = nn.CrossEntropyLoss()
    optimizer.zero_grad()
    
    for i in range(1):
        output = model(x_train)
        loss = criterion(output,y_train)
        train_losses.append(loss)
        loss.backward()
        optimizer.step()


def main():
    X = torch.randn(1000,3,32,32)
    model = CNN()
    
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()

    # print(model)
    train(model)

main()
