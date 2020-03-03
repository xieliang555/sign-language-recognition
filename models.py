import torch
import torchvision
import torch.nn as nn
from torchsummaryX import summary

import utils

import math


class Res3D(nn.Module):
    def __init__(self, res3d):
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


class Transformer(nn.Module):
    '''
    Encoder + Decoder structure
    Args:
        d_model: the embdedding feature dimension
    '''

    def __init__(self, device, tgt_vocab_size, d_model=512, dropout=0.1,
                 nhead=8, nlayer=6, nhid=2048, activation='relu'):
        super(Transformer, self).__init__()
        self.device = device
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

    def get_paddding_mask(self, tgt):
        # the index of '<pad>' is 1
        mask = tgt.eq(1).transpose(0, 1)
        mask = mask.masked_fill(mask == True, float(
            '-inf')).masked_fill(mask == False, float(0.0))
        return mask

    def get_subsequent_mask(self, tgt):
        # torch.triu 上三角
        seq_len = tgt.size(0)
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        mask = mask.masked_fill(mask == 1.0, float(
            '-inf')).masked_fill(mask == 0.0, float(0.0))
        return mask

    def forward(self, x, tgt):
        # src无法计算pad_mask, batch最好是1
        tgt_padding_mask = self.get_paddding_mask(tgt).to(self.device)
        tgt_subsequent_mask = self.get_subsequent_mask(tgt).to(self.device)

        x = x * math.sqrt(self.d_model)
        x = self.dropout(self.pe(x))
        tgt = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.dropout(self.pe(tgt))
        out = self.transformer(
            x, tgt, tgt_mask=tgt_subsequent_mask,
            tgt_key_padding_mask=tgt_padding_mask)

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


class Seq2Seq(nn.Module):
    '''
    Args:
        nclip: the number of clips in a input video
    '''

    def __init__(self, nclip, cnn, transformer, device):
        super(Seq2Seq, self).__init__()
        self.nclip = nclip
        self.device = device

        self.cnn = cnn
        self.transformer = transformer

    def forward(self, x, tgt):
        # cnn_out: [T, N, E]
        cnn_out = torch.zeros(self.nclip, x.size(
            0), self.transformer.d_model).to(self.device)
        x = x.chunk(self.nclip, 2)
        for idx, clip in enumerate(x):
            cnn_out[idx, :, :] = self.cnn(clip)

        y = self.transformer(cnn_out, tgt)
        return y


def greedy_decoder(model, enc_inputs, targets):
    """
        targets: ['sos','w1','w2'...'wn']
        Beam search: K=1
    """
    enc_outputs = model.encoder(enc_inputs)
    dec_inputs = copy.deepcopy(targets)
    for i in range(1, len(dec_inputs)):
        dec_outputs = model.decoder(dec_inputs, enc_inputs, enc_outputs)
        out = model.out(dec_outputs)
        idx = out.max(dim=-1, keepdim=False)[1]
        dec_inputs[i] = idx.data[i-1]
    return dec_inputs


def train(model, train_loader, device, criterion, optimizer, TRG, writer, n_epoch):
    model.train()
    running_loss = 0.0
    running_bleu = 0.0
    running_wer = 0.0
    for batch_idx, batch in enumerate(train_loader):
        inputs = batch['videos'].to(device)
        targets = batch['annotations'].to(device)
        optimizer.zero_grad()
        outputs = model(inputs, targets[:-1, :])
        loss = criterion(outputs.view(-1, outputs.size(-1)),
                         targets[1:, :].view(-1))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_bleu += utils.bleu_count(outputs, targets[1:, :], TRG)
        running_wer += utils.wer_count(outputs, targets[1:, :], TRG)

        if batch_idx % 50 == 49:
            writer.add_scalar('train loss',
                              running_loss / 50,
                              n_epoch * len(train_loader) + batch_idx)
            writer.add_scalar('train bleu',
                              running_bleu / 50,
                              n_epoch * len(train_loader) + batch_idx)
            writer.add_scalar('train wer',
                              running_wer / 50,
                              n_epoch * len(train_loader) + batch_idx)

            running_loss = 0.0
            running_bleu = 0.0
            running_wer = 0.0


def evaluate(model, dev_loader, device, criterion, TRG):
    model.eval()
    epoch_loss = 0.0
    epoch_bleu = 0.0
    epoch_wer = 0.0
    for batch_idx, batch in enumerate(dev_loader):
        inputs = batch['videos'].to(device)
        targets = batch['annotations'].to(device)
        dec_inputs = greedy_decoder(model, inputs, targets[:-1, :])
        outputs = model(inputs, dec_inputs)
        loss = criterion(outputs.view(-1, outputs.size(-1)),
                         targets[1:, :].view(-1))

        epoch_loss += loss.item()
        epoch_bleu += utils.bleu_count(outputs, targets[1:, :], TRG)
        epoch_wer += utils.wer_count(outputs, targets[1:, :], TRG)

    return epoch_loss / len(dev_loader), epoch_bleu / len(dev_loader), epoch_wer / (len(dev_loader))