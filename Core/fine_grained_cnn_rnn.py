import torch
import torch.nn as nn
from .fine_grained_cnn import attention_net


class Fine_grained_CNN_RNN(nn.Module):
    def __init__(self, vocabSize, nHid=1024, nLayer=1,dropout=0.5):
        super(Fine_grained_CNN_RNN, self).__init__()
        self.cnn = attention_net(vocabSize, 6)
        feature_dim = self.cnn.concat_net.in_features
        self.rnn = nn.GRU(
            input_size = feature_dim, hidden_size = nHid, 
            num_layers = nLayer, dropout=dropout, bidirectional=True)
        self.out = nn.Linear(nHid*2, vocabSize+1)
    
    def forward(self, x):
        n,t,c,h,w = x.size()
        x = x.view(-1,c,h,w)
        raw_logits, concat_logits, part_logits, _, top_n_prob, concat_feature = self.cnn(x)
        concat_feature = concat_feature.view(n,t,concat_feature.size(-1))
        concat_feature = concat_feature.permute(1,0,2)
        y,_ = self.rnn(concat_feature)
        y = self.out(y)
        y = nn.functional.log_softmax(y, dim = -1)
        # raw_logits: [N, C]
        # concat_logits: [N, C]
        # part_logits: [N, topN, C]
        # y: [T, N, C]
        return [raw_logits, concat_logits, part_logits, top_n_prob, y]
    
    