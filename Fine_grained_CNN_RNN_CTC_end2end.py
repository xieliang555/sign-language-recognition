import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence
from torchtext.data import Field
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import os
import pandas as pd
from itertools import groupby
from jiwer import wer

from dataset import PhoenixDataset
from models.fine_grained_cnn import list_loss, ranking_loss
from models.fine_grained_cnn_rnn import Fine_grained_CNN_RNN


# -----------------Load PhoenixDataset----------------------
BSZ = 1
FrameSize = 224
root = '/mnt/data/public/datasets'
PROPOSAL_NUM = 6

transform = transforms.Compose([
    transforms.RandomResizedCrop(FrameSize, (0.8, 1)),
    transforms.ToTensor()])

TRG = Field(sequential=True, use_vocab=True,
            init_token=None, eos_token=None,
            lower=True, tokenize='spacy',
            tokenizer_language='de')

csv_dir = os.path.join(root, 'phoenix2014-release/phoenix-2014-multisigner')
csv_dir = os.path.join(csv_dir, 'annotations/manual/train.corpus.csv')
csv_file = pd.read_csv(csv_dir)
tgt_sents = [csv_file.iloc[i, 0].lower().split('|')[3].split()
             for i in range(len(csv_file))]
TRG.build_vocab(tgt_sents, min_freq=1)
VocabSize = len(TRG.vocab)


def my_collate(batch):
    videos = [item['video'] for item in batch]
    video_lens = torch.tensor([len(v) for v in videos])
    videos = pad_sequence(videos, batch_first=True)

    annotations = [item['annotation'].split() for item in batch]
    anno_lens = torch.tensor([len(a) for a in annotations])
    annotations = TRG.process(annotations)

    return {'videos': videos, 'annotations': annotations,
            'video_lens': video_lens, 'anno_lens': anno_lens}


train_loader = DataLoader(
    PhoenixDataset(root, 'train', transform),
    batch_size=BSZ, num_workers=BSZ, shuffle=True,
    pin_memory=True, collate_fn=my_collate)

dev_loader = DataLoader(
    PhoenixDataset(root, 'dev', transform),
    batch_size=BSZ, num_workers=BSZ, shuffle=False,
    pin_memory=True, collate_fn=my_collate)

test_loader = DataLoader(
    PhoenixDataset(root, 'test', transform),
    batch_size=BSZ, num_workers=BSZ, shuffle=False,
    pin_memory=True, collate_fn=my_collate)


# --------------------------Define train-------------------------
def train(net, train_loader, optimizer, criterion_ctc, criterion_cnn, epoch, writer):
    net.train()
    running_loss = 0.0
    running_wer = 0.0

    for batch_idx, batch in enumerate(train_loader):
        inputs = batch['videos'].cuda()
        targets = batch['annotations'].permute(1, 0).contiguous().cuda()
        input_lens = batch['video_lens'].cuda()
        target_lens = batch['anno_lens'].cuda()
        n, t, c, h, w = inputs.size()

        optimizer.zero_grad()
        raw_logits, concat_logits, part_logits, top_n_prob, outs = net(inputs)
        cnn_targets = outs.max(-1)[1].permute(1, 0).contiguous().view(-1).data
        loss_raw = criterion_cnn(raw_logits, cnn_targets)
        loss_concat = criterion_cnn(concat_logits, cnn_targets)
        # ?简化？
        loss_partcls = criterion_cnn(
            part_logits.view(n*t*PROPOSAL_NUM, -1),
            cnn_targets.unsqueeze(1).repeat(1, PROPOSAL_NUM).view(-1))
        # ?简化
        part_targets = list_loss(
            part_logits.view(n*t*PROPOSAL_NUM, -1),
            cnn_targets.unsqueeze(1).repeat(1, PROPOSAL_NUM).view(-1)).view(n*t, PROPOSAL_NUM)
        loss_rank = ranking_loss(top_n_prob, part_targets)
        loss_ctc = criterion_ctc(outs, targets, input_lens, target_lens)
        loss_total = loss_raw + loss_concat + loss_partcls + loss_rank + loss_ctc
        loss_total.backward()

        # ignore batch that lead gradient exploration
        flag = False
        for name, param in net.named_parameters():
            if param.grad != None and torch.isnan(param.grad).any():
                flag = True
                break
        if flag:
            print(batch_idx)
            continue

        optimizer.step()

        outs = outs.max(-1)[1].permute(1, 0).contiguous().view(-1)
        outs = ' '.join([TRG.vocab.itos[i]
                         for i, _ in groupby(outs) if i != VocabSize])
        targets = ' '.join([TRG.vocab.itos[i] for i in targets.view(-1)])
        running_wer += wer(targets, outs, standardize=True)
        running_loss += loss_total.item()

        N = len(train_loader) // 10
        if batch_idx % N == N-1:
            writer.add_scalar('train loss',
                              running_loss/N,
                              epoch*len(train_loader)+batch_idx)
            writer.add_scalar('train wer',
                              running_wer/N,
                              epoch*len(train_loader)+batch_idx)

            running_loss = 0.0
            running_wer = 0.0


# --------------------------Define dev-------------------------
def dev(net, dev_loader, criterion_ctc, criterion_cnn, epoch, writer):
    net.eval()
    epoch_loss = 0.0
    epoch_wer = 0.0
    with torch.no_grad():
        for batch_idx, batch in enumerate(dev_loader):
            inputs = batch['videos'].cuda()
            targets = batch['annotations'].permute(1, 0).contiguous().cuda()
            input_lens = batch['video_lens'].cuda()
            target_lens = batch['anno_lens'].cuda()
            n, t, c, h, w = inputs.size()

            raw_logits, concat_logits, part_logits, top_n_prob, outs = net(inputs)
            cnn_targets = outs.max(-1)[1].permute(1, 0).contiguous().view(-1).data
            loss_raw = criterion_cnn(raw_logits, cnn_targets)
            loss_concat = criterion_cnn(concat_logits, cnn_targets)
            # ?简化？
            loss_partcls = criterion_cnn(
                part_logits.view(n*t*PROPOSAL_NUM, -1),
                cnn_targets.unsqueeze(1).repeat(1, PROPOSAL_NUM).view(-1))
            # ?简化
            part_targets = list_loss(
                part_logits.view(n*t*PROPOSAL_NUM,-1), 
                cnn_targets.unsqueeze(1).repeat(1, PROPOSAL_NUM).view(-1)).view(n*t, PROPOSAL_NUM)
            loss_rank = ranking_loss(top_n_prob, part_targets)
            loss_ctc = criterion_ctc(outs, targets, input_lens, target_lens)
            loss_total = loss_raw + loss_concat + loss_partcls + loss_rank + loss_ctc

            outs = outs.max(-1)[1].permute(1, 0).contiguous().view(-1)
            outs = ' '.join([TRG.vocab.itos[i]
                             for i, _ in groupby(outs) if i != VocabSize])
            targets = ' '.join([TRG.vocab.itos[i] for i in targets.view(-1)])
            epoch_wer += wer(targets, outs, standardize=True)
            epoch_loss += loss_total.item()

    epoch_wer /= len(dev_loader)
    epoch_loss /= len(dev_loader)
    if writer:
        writer.add_scalar('dev loss', epoch_loss, epoch)
        writer.add_scalar('dev wer', epoch_wer, epoch)

    return epoch_wer, epoch_loss


# --------------------------train dev test-------------------------
resume_training = True

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
LR = 1e-4
WD = 1e-4
save_root = '/home/xieliang/Data/sign-language-recognition'
if resume_training:
    save_dict = torch.load(os.path.join(
        save_root, 'save/Fine_grained_CNN_RNN6.pth'))
    start_epoch = save_dict['epoch']+1
    best_dev_wer = save_dict['best dev wer']
    net = save_dict['net'].cuda()
else:
    start_epoch = 0
    best_dev_wer = 2
    net = Fine_grained_CNN_RNN(VocabSize).cuda()

criterion_ctc = nn.CTCLoss(blank=VocabSize)
criterion_cnn = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=LR, weight_decay=WD)
writer = SummaryWriter(os.path.join(
    save_root, 'log/Fine_grained_CNN_RNN_CTC6'))

for epoch in range(start_epoch, 1000):
    train(net, train_loader, optimizer, criterion_ctc, criterion_cnn, epoch, writer)
    dev_wer, dev_loss = dev(net, dev_loader, criterion_ctc, criterion_cnn, epoch, writer)
    print(f'epoch: {epoch} | dev wer: {dev_wer} | dev loss: {dev_loss}')
    
    if dev_wer < best_dev_wer:
        best_dev_wer = dev_wer
        torch.save({'epoch': epoch,
                    'net': net,
                    'dev wer': dev_wer,
                    'dev loss': dev_loss,
                    'best dev wer': best_dev_wer}, 
                   os.path.join(save_root, 'save/Fine_grained_CNN_RNN6.pth'))
        print(f'model saved with best dev wer: {best_dev_wer} in epoch {epoch}') 