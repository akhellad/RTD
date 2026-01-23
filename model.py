import torch.nn as nn 
import torch.optim as optim

class BackBone(nn.Module):
    def __init__(self):
        super().__init__()
        self.bloc1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64)
        )

        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn7 = nn.BatchNorm2d(512)
        self.conv8 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn8 = nn.BatchNorm2d(512)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv1(x)  
        x = self.bn1(x)   
        x = self.relu(x)  
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)     
        x = self.maxpool(x)    
        x = self.conv3(x)  
        x = self.bn3(x)    
        x = self.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)      
        x = self.maxpool(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv7(x)     
        x = self.bn7(x)
        x = self.relu(x)
        x = self.conv8(x)
        x = self.bn8(x)
        x = self.relu(x)      
        x = self.maxpool(x)
        return x 
         