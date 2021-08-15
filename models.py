import torch.nn as nn

from efficientnet_pytorch import EfficientNet

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b4')
        self.output = nn.Linear(1000, 24)
        
    def forward(self, inputs):
        features = self.model(inputs)
        output = self.output(features)
        
        return output

class TestNet(nn.Module):
    def __init__(self, model):
        super(Net, self).__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b4')
        self.output = nn.Linear(1000, 1)
        
    def forward(self, inputs):
        features = self.model(inputs)
        output = self.output(features)
        
        return output