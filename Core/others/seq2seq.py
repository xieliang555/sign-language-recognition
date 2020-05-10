import torch
import torchvision
import torch.nn as nn



class Res3d_transformer(nn.Module):
    '''
    Args:
        nclip: the number of clips in a input video
    '''

    def __init__(self, clip_size, cnn, transformer):
        super(Seq2Seq, self).__init__()
        self.clip_size = clip_size

        self.cnn = cnn

        # ?
        for params in self.cnn.parameters():
            params.requires_grad = False

        self.transformer = transformer

    def forward(self, src, tgt, src_padding_mask, device):
        # src: [N, C, T, H, W]
        # x: [nclip, N, C, T, H, W]
        # cnn_out: [nclip, N, E]
        # x, src_padding_mask = utils.uniform_temporal_segment(src, src_padding_mask, self.clip_size)
        x, src_padding_mask = utils.overlap_temporal_segment(src, src_padding_mask, self.clip_size)
        x = x.to(device)
        
        cnn_out = torch.zeros(len(x), src.size(
            0), self.transformer.d_model).to(device)
        
        for idx, clip in enumerate(x):
            cnn_out[idx, :, :] = self.cnn(clip)
        
        y = self.transformer(cnn_out, tgt, src_padding_mask, device)
        return y
    
    
    
class CNN_RNN(nn.Module):
    def __init__(self, vocab_size, nhid, nlayer, dropout=0.5):
        super(CNN_GRU, self).__init__()
        
        # resnet18
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
        out = nn.functional.log_softmax(out, dim = -1)
        return out
       
    
    
    
    
    
    
    