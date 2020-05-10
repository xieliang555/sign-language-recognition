import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

    
class CNN_RNN(nn.Module):
    def __init__(self, vocab_size, nhid, nlayer, dropout=0.5):
        super(CNN_RNN, self).__init__()
        self.resnet18 = torchvision.models.resnet18(pretrained = True)
        cnn_out_dim = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Identity()
        self.gru = nn.GRU(
            input_size = cnn_out_dim, hidden_size = nhid,
            num_layers = nlayer, dropout = dropout, bidirectional = True)
        self.out = nn.Linear(nhid*2, vocab_size+1)
        
    def forward(self, x):
        n,t,c,h,w = x.size()
        x = x.view(-1,c,h,w)
        x = self.resnet18(x)
        x = x.view(n, t, x.size(-1))
        x = x.permute(1,0,2)
        # pad pack?
        x, _ = self.gru(x)
        out = self.out(x)
        out = F.log_softmax(out, dim = -1)
        return out
    
    
    
    
class CNN_RNN_Concat(nn.Module):
    def __init__(self, vocab_size, nhid, nlayer, dropout=0.5):
        super(CNN_RNN_Concat, self).__init__()
        self.resnet18 = torchvision.models.resnet18(pretrained=True)
        cnn_out_dim = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Identity()
        self.gru = nn.GRU(
            input_size = cnn_out_dim*2, hidden_size = nhid, 
            num_layers = nlayer, dropout=dropout, bidirectional=True)
        self.out = nn.Linear(nhid*2, vocab_size+1)
    
    def forward(self, fullVideos, trackedHandVideos):
        n,t,c,h1,w1 = fullVideos.size()
        _,_,_,h2,w2 = trackedHandVideos.size()
        fullVideos = fullVideos.view(-1,c,h1,w1)
        trackedHandVideos = trackedHandVideos.view(-1,c,h2,w2)
        feature1 = self.resnet18(fullVideos).view(n,t,-1).permute(1,0,2)
        feature2 = self.resnet18(trackedHandVideos).view(n,t,-1).permute(1,0,2)
        # concat? add ?
        feature = torch.cat((feature1, feature2), dim=-1)
        feature, _ = self.gru(feature)
        out = self.out(feature)
        out = F.log_softmax(out, dim=-1)
        return out
    
       
    
    
    
    
    
    
    