import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence
from torchtext.data import Field
from torch.utils.tensorboard import SummaryWriter
from torchsummaryX import summary

import warnings
warnings.filterwarnings("ignore")
from itertools import groupby
from jiwer import wer
import pandas as pd
import os

from Core import cnn_rnn
from Core.dataset import Phoenix_Full_TrackedHand


#-----------------------------load dataset--------------------------
TRG = Field(sequential=True, use_vocab=True,
            init_token=None, eos_token= None,
            lower=True, tokenize='spacy',
            tokenizer_language='de')


root = '/mnt/data/public/datasets'
csv_dir = os.path.join(root, 'phoenix2014-release/phoenix-2014-multisigner')
csv_dir = os.path.join(csv_dir, 'annotations/manual/train.corpus.csv')
csv_file = pd.read_csv(csv_dir)
tgt_sents = [csv_file.iloc[i, 0].lower().split('|')[3].split()
             for i in range(len(csv_file))]
TRG.build_vocab(tgt_sents, min_freq=1)
VocabSize = len(TRG.vocab)

def collate_full_trackedHand(batch):
    fullVideos = [item['fullVideo'] for item in batch]
    video_lens = torch.tensor([len(v) for v in fullVideos])
    fullVideos = pad_sequence(fullVideos, batch_first=True)
    trackedHandVideos = [item['trackedHandVideo'] for item in batch]
    trackedHandVideos = pad_sequence(trackedHandVideos, batch_first=True)
    annotations = [item['annotation'].split() for item in batch]
    annotation_lens = torch.tensor([len(anno) for anno in annotations])
    annotations = TRG.process(annotations)
    return {'fullVideos': fullVideos,
            'trackedHandVideos':trackedHandVideos,
            'annotations': annotations,
            'video_lens': video_lens,
            'annotation_lens': annotation_lens}

fullFrameSize = 224
trackedHandFrameSize = 112
BSZ = 1
interval = 2

fullTransform = transforms.Compose([
    transforms.RandomResizedCrop(fullFrameSize, (0.8,1)),
    transforms.ToTensor()])

trackedHandTransform = transforms.Compose([
    transforms.RandomResizedCrop(trackedHandFrameSize, (0.8,1)), 
    transforms.ToTensor()])

train_loader = DataLoader(
    Phoenix_Full_TrackedHand(
        root, mode='train', interval=interval, fullTransform=fullTransform,
        trackedHandTransform = trackedHandTransform),
    batch_size=BSZ, shuffle=True, num_workers=BSZ,
    collate_fn=collate_full_trackedHand, pin_memory=True)

dev_loader = DataLoader(
    Phoenix_Full_TrackedHand(
        root, mode='dev', interval = interval, fullTransform=fullTransform,
        trackedHandTransform = trackedHandTransform),
    batch_size=BSZ, shuffle=False, num_workers=BSZ,
    collate_fn=collate_full_trackedHand, pin_memory=True)

test_loader = DataLoader(
    Phoenix_Full_TrackedHand(
        root, mode='test', interval = interval, fullTransform=fullTransform,
        trackedHandTransform = trackedHandTransform),
    batch_size=BSZ, shuffle=False, num_workers=BSZ,
    collate_fn=collate_full_trackedHand, pin_memory=True)


#-----------------------------define train--------------------------
def train(net, train_loader, criterion, optimizer, epoch, writer):
    net.train()
    running_wer = 0.0
    running_loss = 0.0
    for batch_idx, batch in enumerate(train_loader):
        # ?
        fullVideos = batch['fullVideos'].cuda()
        trackedHandVideos = batch['trackedHandVideos'].cuda()
        targets = batch['annotations'].permute(1,0).contiguous().cuda()
        input_lens = batch['video_lens'].cuda()
        target_lens = batch['annotation_lens'].cuda()
        
        optimizer.zero_grad()
        outs = net(fullVideos, trackedHandVideos)
        loss = criterion(outs, targets, input_lens, target_lens)
        loss.backward()

        flag = False
        for name, param in net.named_parameters():
            if torch.isnan(param.grad).any():
                flag = True
                break
        if flag:
            print(batch_idx)
            continue
            
        optimizer.step()
        
        
        outs = outs.max(-1)[1].permute(1,0).contiguous().view(-1)
        # best ctc_decoder?
        outs = ' '.join([TRG.vocab.itos[k] for k, _ in groupby(outs) if k != VocabSize])
        targets = targets.view(-1)
        targets = ' '.join([TRG.vocab.itos[k] for k in targets])
        running_wer += wer(targets, outs, standardize=True)
        running_loss += loss.item()
        
        N = len(train_loader) // 10
        if batch_idx % N == N-1:
            writer.add_scalar('train wer',
                              running_wer/N,
                              epoch*len(train_loader)+batch_idx)
            writer.add_scalar('train loss',
                              running_loss/N,
                              epoch*len(train_loader)+batch_idx)
        
            running_wer = 0.0
            running_loss = 0.0    


#-----------------------------define dev--------------------------
def val(net, dev_loader, criterion, epoch, writer):
    net.eval()
    epoch_wer = 0.0
    epoch_loss = 0.0
    with torch.no_grad():
        for batch_idx, batch in enumerate(dev_loader):
            fullVideos = batch['fullVideos'].cuda()
            trackedHandVideos = batch['trackedHandVideos'].cuda()
            targets = batch['annotations'].permute(1,0).contiguous().cuda()
            input_lens = batch['video_lens'].cuda()
            target_lens = batch['annotation_lens'].cuda()
            
            outs = net(fullVideos, trackedHandVideos)
            loss = criterion(outs, targets, input_lens, target_lens)
            
            outs = outs.max(-1)[1].permute(1,0).contiguous().view(-1)
            outs = ' '.join([TRG.vocab.itos[k] for k, _ in groupby(outs) if k != VocabSize])
            targets = targets.view(-1)
            targets = ' '.join([TRG.vocab.itos[k] for k in targets])
            epoch_wer += wer(targets, outs, standardize=True)
            epoch_loss += loss.item()
            
            
        epoch_wer /= len(dev_loader)
        epoch_loss /= len(dev_loader)
        if writer:
            writer.add_scalar('dev wer', epoch_wer, epoch)
            writer.add_scalar('dev loss', epoch_loss, epoch)
          
    return epoch_loss, epoch_wer


#-----------------------------train dev test--------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
save_root = '/home/xieliang/Data/sign-language-recognition'
save_model = os.path.join(save_root, 'save/CNN_RNN_CTC_Combined.pth')
save_log = os.path.join(save_root, 'log/CNN_RNN_CTC_Combined')

nHID = 1024
nLAYER = 1
resume_training = True
if resume_training:
    save_dict = torch.load(save_model)
    start_epoch = save_dict['epoch']+1
    best_dev_wer = save_dict['best_dev_wer']
    net = save_dict['net'].cuda()
else:
    start_epoch = 0
    best_dev_wer = 2
    net = cnn_rnn.CNN_RNN_Concat(VocabSize, nHID, nLAYER, dropout=0.5).cuda()

LR = 1e-7
WD = 1e-4
criterion = nn.CTCLoss(blank=VocabSize)
optimizer = optim.Adam(net.parameters(), lr=LR, weight_decay=WD)
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [60, 100])
writer = SummaryWriter(save_log)

print(f'start training from epoch {start_epoch} with best dev wer {best_dev_wer}')
for epoch in range(start_epoch, 1000):
    train(net, train_loader, criterion, optimizer, epoch, writer)
    dev_loss, dev_wer = val(net, dev_loader, criterion, epoch, writer)
    lr_scheduler.step()
    print(f'epoch:{epoch} | dev_loss:{dev_loss} | dev_wer:{dev_wer} | lr:{lr_scheduler.get_lr()}')

    if dev_wer < best_dev_wer:
        best_dev_wer = dev_wer
        torch.save({'epoch': epoch, 'best_dev_wer': best_dev_wer,
                    'net': net}, save_model)
        print(f'model saved with best dev wer: {best_dev_wer} in epoch {epoch}')

