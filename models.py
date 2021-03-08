import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class PretrainedResNet(nn.Module):
    def __init__(self):
        super().__init__()
        ResNet = models.resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(ResNet.children())[:-1])
        for param in self.backbone:
            param.requires_grad = False
        self.classifier = nn.Linear(512,20,bias=True)
    
    def forward(self, x):
        features = self.backbone(x)
        features = features.view(-1, 512)
        out = self.classifier(features)
        return out
    
class CaffeNet(nn.Module):
    def __init__(self, num_classes=20, size=227, c_dim=3):
        super().__init__()
        self.num_classes = num_classes
        # size
        self.conv1 = nn.Conv2d(c_dim, 96, 11, 4)
        # size is 55
        self.conv2 = nn.Conv2d(96 , 256, 5, 1, padding=2)
        self.conv3 = nn.Conv2d(256, 384, 3, 1, padding=1)
        self.conv4 = nn.Conv2d(384, 384, 3, 1, padding=1)
        self.conv5 = nn.Conv2d(384, 256, 3, 1, padding=1)
        
        map_size = (size - 11) / 4 + 1
        map_size = (((map_size - 1)/ 2 - 1) / 2 - 1) / 2
        self.flat_dim = int(map_size**2 * 256)
        
        self.nonlinear = nn.ReLU()
        self.pool1 = nn.MaxPool2d(3,2)
        self.pool2 = nn.MaxPool2d(3,2)
        self.pool3 = nn.MaxPool2d(3,2)
        
        self.dropout = nn.Dropout(p=0.5)
        
        self.fc1 = nn.Linear(self.flat_dim, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)
        
        self.backbone = nn.Sequential()
        
    def backbone(self, x):
        x = self.conv1(x)
        x = self.nonlinear(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.nonlinear(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.nonlinear(x)
        x = self.conv4(x)
        x = self.nonlinear(x)
        x = self.conv5(x)
        x = self.nonlinear(x)
        x = self.pool3(x)
        x = x.view(-1, self.flat_dim)
        return x
    
    def classifier(self, x):
        x = self.fc1(x)
        x = self.nonlinear(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.nonlinear(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x
    
    def tsne_features(self, x):
        x = self.backbone(x)
        x = self.fc1(x)
        x = self.nonlinear(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
    def forward(self, x):
        features = self.backbone(x)
        out = self.classifier(features)
        return out 
