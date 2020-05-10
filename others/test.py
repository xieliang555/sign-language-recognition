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

import utils
from models import seq2seq
from dataset import PhoenixDataset, RandomResizedCropVideo, ToTensorVideo


# build vocab
TRG = Field(sequential=True, use_vocab=True,
            init_token=None, eos_token= None,
            lower=True, tokenize='spacy',
            tokenizer_language='de')


root = '/mnt/data/public/datasets'
csv_dir = os.path.join(root, 'phoenix2014-release/phoenix-2014-multisigner')
# train -> test?
csv_dir = os.path.join(csv_dir, 'annotations/manual/train.corpus.csv')
csv_file = pd.read_csv(csv_dir)

tgt_sents = [csv_file.iloc[i, 0].lower().split('|')[3].split()
             for i in range(len(csv_file))]

# # ?
# tgt_sents = [tgt_sents[0]]
# print(tgt_sents)

# ？
TRG.build_vocab(tgt_sents, min_freq=1)
VocabSize = len(TRG.vocab)


# process batch
# 视频长度1/4？ dataset
def collate_fn(batch):
    '''
        batch: [{'video':video, 'annotation':annotation}...]
        video: [C, T, H, W]
        pad the video sequence to fixed length
        numericalize the annotation
    '''
    # ? delete permute
#     videos = [item['video'].permute(1,0,2,3) for item in batch]
    videos = [item['video'] for item in batch]
    video_lens = torch.tensor([len(v) for v in videos])
    videos = pad_sequence(videos, batch_first=True)
    
    annotations = [item['annotation'].split() for item in batch]
    annotation_lens = torch.tensor([len(anno) for anno in annotations])
    annotations = TRG.process(annotations)
    
    return {'videos': videos,
            'annotations': annotations,
            'video_lens': video_lens,
            'annotation_lens': annotation_lens}


# load dataset
FrameSize = 224
BatchSize = 2

# ?
# transform = transforms.Compose(
#     [ToTensorVideo(), RandomResizedCropVideo(FrameSize)])

transform = transforms.Compose([
    transforms.RandomResizedCrop(FrameSize),
    transforms.ToTensor()])

# shuffle
train_loader = DataLoader(
    PhoenixDataset(root, mode='train', transform=transform),
    batch_size=BatchSize, shuffle=True, num_workers=2,
    collate_fn=collate_fn, pin_memory=True)

val_loader = DataLoader(
    PhoenixDataset(root, mode='dev', transform=transform),
    batch_size=BatchSize, shuffle=True, num_workers=2,
    collate_fn=collate_fn, pin_memory=True)

test_loader = DataLoader(
    PhoenixDataset(root, mode='test', transform=transform),
    batch_size=BatchSize, shuffle=True, num_workers=2,
    collate_fn=collate_fn, pin_memory=True)


# define model
# ?
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
pretrained_path = './save/Res18_Pretrained_Phoenix_FrameWise.pth'
net = torch.load('./save/CNN_GRU_CTC_total9.pth', map_location=device)
criterion = nn.CTCLoss(blank=VocabSize)


def val(net, val_loader, criterion, epoch, writer, device):
    net.eval()
    epoch_wer = 0.0
    epoch_loss = 0.0
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            inputs = batch['videos'].to(device)
            targets = batch['annotations'].permute(1,0).contiguous().to(device)
            input_lens = batch['video_lens'].to(device)
            target_lens = batch['annotation_lens'].to(device)
            
            outs = net(inputs)
            loss = criterion(outs, targets, input_lens, target_lens)
            
            outs = outs.max(-1)[1].permute(1,0).contiguous().view(-1)
            outs = ' '.join([TRG.vocab.itos[k] for k, _ in groupby(outs) if k != VocabSize])
            targets = targets.view(-1)
            targets = ' '.join([TRG.vocab.itos[k] for k in targets])
            epoch_wer += wer(targets, outs, standardize=True)
            epoch_loss += loss.item()
            
            
        epoch_wer /= len(val_loader)
        epoch_loss /= len(val_loader)
        if writer:
            writer.add_scalar('val wer', epoch_wer, epoch)
            writer.add_scalar('val loss', epoch_loss, epoch)
          
    return epoch_loss, epoch_wer



if __name__ == '__main__':
    test_loss, test_wer = val(net, val_loader, criterion, epoch=0, writer=None, device=device)
    print(f'test loss:{test_loss} | test wer:{test_wer}')