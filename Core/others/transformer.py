import torch
import torch.nn as nn
import math


def get_padding_mask(data):
    if len(data.size()) == 2:
        # tgt: [T, N]
        # the index of tgt padding is 1
        mask = data.eq(1).transpose(0, 1)
    elif len(data.size()) == 5:
        # src: [N, C, T, H, W]
        # the index of src padding is 0
        data = data.permute(0, 2, 1, 3, 4).sum(dim=(-3, -2, -1))
        mask = data.eq(0)

    mask = mask.masked_fill(mask == True, float(
        '-inf')).masked_fill(mask == False, float(0.0))
    return mask


def get_subsequent_mask(tgt):
    # torch.triu 上三角
    seq_len = tgt.size(0)
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    mask = mask.masked_fill(mask == 1.0, float(
        '-inf')).masked_fill(mask == 0.0, float(0.0))
    return mask


class Transformer(nn.Module):
    '''
    Args:
        d_model: the embdedding feature dimension
    '''

    def __init__(self, tgt_vocab_size, d_model=512, dropout=0.1,
                 nhead=8, nlayer=6, nhid=2048, activation='relu'):
        super(Transformer, self).__init__()
        self.d_model = d_model

        self.embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pe = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead,
            num_encoder_layers=nlayer,
            num_decoder_layers=nlayer,
            dim_feedforward=nhid,
            activation=activation)
        self.out = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, x, tgt, src_padding_mask, device):
        src_padding_mask = src_padding_mask.to(device)
        memory_padding_mask = src_padding_mask.to(device)
        tgt_padding_mask = get_padding_mask(tgt).to(device)
        tgt_subsequent_mask = get_subsequent_mask(tgt).to(device)

        x = x * math.sqrt(self.d_model)
        x = self.dropout(self.pe(x))
        tgt = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.dropout(self.pe(tgt))
        out = self.transformer(
            x, tgt, tgt_mask=tgt_subsequent_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=memory_padding_mask)

        return self.out(out)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        # 只需确定一维
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(- torch.arange(0, d_model,
                                            2).float() * math.log(10000) / d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x
    
    
#  beam k >1 ?    
def greedy_decoder(model, inputs, targets, src_padding_mask, device):
    '''
        targets: ['sos', 'w1', 'w2'...'wn']
        beam: 1
        仅用于测试集，运行太慢
    '''
    dec_inputs = targets.clone()
    for i in range(1, len(dec_inputs)):
        out = model(inputs, dec_inputs, src_padding_mask, device)
        out = out.max(-1)[1]
        dec_inputs[i] = out[i-1]
    return dec_inputs
    

# # beam >1 ?
# def greedy_decoder(model, inputs, targets, src_padding_mask, device):
#     """
#         targets: ['sos','w1','w2'...'wn']
#         Beam search: K=1
#     """
#     src_padding_mask = src_padding_mask.to(device)
#     memory_padding_mask = src_padding_mask.to(device)
#     tgt_padding_mask = get_padding_mask(targets).to(device)
#     tgt_subsequent_mask = get_subsequent_mask(targets).to(device)

#     enc_inputs = model.cnn(inputs)
#     enc_inputs = enc_inputs * math.sqrt(model.transformer.d_model)
#     enc_inputs = model.transformer.dropout(model.transformer.pe(enc_inputs))
#     enc_outputs = model.transformer.transformer.encoder(
#         enc_inputs, src_key_padding_mask=src_padding_mask)

#     dec_inputs = targets.clone()
#     for i in range(1, len(dec_inputs)):
#         dec_inputs = model.transformer.embedding(
#             dec_inputs) * math.sqrt(model.transformer.d_model)
#         dec_inputs = model.transformer.dropout(
#             model.transformer.pe(dec_inputs))

#         dec_outputs = model.transformer.transformer.decoder(
#             dec_inputs, enc_outputs, 
#             tgt_mask=tgt_subsequent_mask, 
#             tgt_key_padding_mask=tgt_padding_mask, 
#             memory_key_padding_mask=memory_padding_mask)
        
#         out = model.transformer.out(dec_outputs)
#         idx = out.max(dim=-1, keepdim=False)[1]
#         dec_inputs[i] = idx.data[i-1]
#     return dec_inputs
