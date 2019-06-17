# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 12:07:08 2019

@author: radus
"""

import torch
import torchvision
import torchvision.transforms as transforms
import torch.cuda
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data 
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import torch.backends.cudnn as cudnn

def train_test(epochs):
    for epoch in range(epochs):
        for step, (x,y) in enumerate(train_loader):
            x=x.to(device)
            y=y.to(device)
            bx=Variable(x).to(device)
            by=Variable(y).to(device)
            output=model(bx)[0].to(device);
            loss_value=loss(output,by)
            loss_value=loss_value.to(device)
            optimizer.zero_grad();
            loss_value.backward();
            optimizer.step();
            if step%100==0:
                test_output, test_layer = model(test_x)[0].to(device), model(test_x)[1].to(device);
                pred_y=torch.max(test_output, 1)[1].data.squeeze()
                accuracy=   (pred_y==test_y).sum().item()/float(test_y.size(0));
                print("Epoch:",epoch, "Loss: ", loss_value.item(), "Accuracy", accuracy);
        print("Training complete");

    

    
def import_image():
    import cv2
    img=cv2.imread('digit.jpg',0);
    img=cv2.resize(img,(28,28))
    cv2.imshow("s",img);
    cv2.waitKey(0)
    cv2.destroyAllWindows();
    img=torch.tensor(img[None][None]).float();
    output, _ = model(img);
    pred_y=torch.max(output,1)[1].data.numpy().squeeze();
    print("Predicted digit: ",pred_y);

    
    
class Skynet(nn.Module):
    def __init__(self):
        super(Skynet, self).__init__();
        self.layer1=nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=16,kernel_size=5, stride=1, padding=2),nn.ReLU(),
                nn.MaxPool2d(kernel_size=2));
        self.layer2=nn.Sequential(
                nn.Conv2d(16, 32,5,1,2),nn.ReLU(),
                nn.MaxPool2d(2));
        self.dense=nn.Sequential(
                nn.Linear(32*7*7,10),nn.ReLU());
        
    def forward(self, x):
        x=x.to(device)
        x=self.layer1(x);
        x=self.layer2(x);
        x=x.view(x.size(0),-1);
        output=self.dense(x);
        return output, x;


epochs=16
batchSize=10000;
lr=0.01
trans=transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                         transforms.RandomVerticalFlip(p=0.5),
                         transforms.ToTensor()])
train_data = torchvision.datasets.MNIST(root='dataset/MNIST',train=True,transform=trans, download=True);

print("Data downloaded!");

train_loader=Data.DataLoader(dataset=train_data, batch_size=batchSize, shuffle=True);
test_data = torchvision.datasets.MNIST(root='dataset/MNIST', train=False);
test_x = Variable(torch.unsqueeze(test_data.test_data, dim=1)).type(torch.FloatTensor)[:2000]/255.0;
test_y=test_data.test_labels[:2000];
device= 'cuda' if torch.cuda.is_available() else 'cpu';

model=Skynet();
model= model.to(device);
cudnn.benchmark=True;   
print(model)
print("Model created sucessfully");

optimizer=optim.Adam(model.parameters(), lr=lr);
loss=nn.CrossEntropyLoss();
train_test(epochs)
import_image();
torch.save(model.state_dict(),'MNIST_MODEL.txt');


