import torch.nn as nn


class Res3D(nn.Module):
    def __init__(self, res3d):
        '''
        Args:
            res3d: the pretrained model
        '''
        super(Res3D, self).__init__()
        self.convs = nn.Sequential(
            res3d.stem,
            res3d.layer1,
            res3d.layer2,
            res3d.layer3,
            res3d.layer4,
            res3d.avgpool)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.convs(x)
        x = self.flatten(x)
        return x
    
   