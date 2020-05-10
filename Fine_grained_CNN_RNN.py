import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence
from torchtext.data import Field
from torch.utils.tensorboard import SummaryWriter
from torchsummaryX import summary

import os
import pandas as pd
from PIL import Image

from models.config import INPUT_SIZE
from dataset import PhoenixDataset
from models.fine_grained_cnn import attention_net, list_loss, ranking_loss

PROPOSAL_NUM = 6


#-------------------------------------load dataset-----------------------------------
BSZ = 1
root = '/mnt/data/public/datasets'

train_transform = transforms.Compose([
    transforms.Resize(INPUT_SIZE, Image.BILINEAR),
    transforms.RandomCrop(INPUT_SIZE), transforms.ToTensor()])

test_transform = transforms.Compose([
    transforms.Resize(INPUT_SIZE, Image.BILINEAR), 
    transforms.CenterCrop(INPUT_SIZE), transforms.ToTensor()])

TRG = Field(sequential=True, use_vocab=True,
            init_token=None, eos_token=None,
            lower=True, tokenize='spacy',
            tokenizer_language='de')

csv_path = os.path.join(root, 'phoenix2014-release/phoenix-2014-multisigner')
csv_path = os.path.join(csv_path, 'annotations/manual/train.corpus.csv')
csv_file = pd.read_csv(csv_path)
train_sents = [csv_file.iloc[i, 0].lower().split('|')[3].split()
               for i in range(len(csv_file))]
TRG.build_vocab(train_sents, min_freq=1)
VocabSize = len(TRG.vocab)


def my_collate(batch):
    videos = [item['video'] for item in batch]
    videos = pad_sequence(videos, batch_first=True)
    annotations = [item['annotation'].split() for item in batch]
    annotations = TRG.process(annotations)
    return {'videos': videos, 'annotations': annotations}


train_loader = DataLoader(
    PhoenixDataset(root, 'train', train_transform),
    batch_size=BSZ, num_workers=BSZ, pin_memory=True,
    shuffle=True, collate_fn=my_collate)

dev_loader = DataLoader(
    PhoenixDataset(root, 'dev', test_transform),
    batch_size=BSZ, num_workers=BSZ, pin_memory=True,
    shuffle=False, collate_fn=my_collate)

test_loader = DataLoader(
    PhoenixDataset(root, 'test', test_transform),
    batch_size=BSZ, num_workers=BSZ, pin_memory=True,
    shuffle=False, collate_fn=my_collate)


#-------------------------------------define train-----------------------------------
def train(fine_grained_cnn_net, cnn_rnn_ctc_net, train_loader,
          optimizer, criterion, epoch, writer):
    running_total_loss = 0.0
    running_raw_loss = 0.0
    running_concat_loss = 0.0
    running_partcls_loss = 0.0
    running_ranking_loss = 0.0
    running_acc = 0.0
    fine_grained_cnn_net.eval()
    cnn_rnn_ctc_net.train()
    for batch_idx, batch in enumerate(train_loader):
        inputs = batch['videos'].cuda()
        targets = cnn_rnn_ctc_net(inputs)
        targets = targets.max(-1)[1].permute(1, 0).contiguous().view(-1).data

        optimizer.zero_grad()
        n, t, c, h, w = inputs.size()
        inputs = inputs.view(-1, c, h, w)
        raw_logits, concat_logits, part_logits, _, top_n_prob, _ = fine_grained_cnn_net(inputs)
        # part_logits shape改变？
        raw_loss = criterion(raw_logits, targets)
        concat_loss = criterion(concat_logits, targets)
        partcls_loss = criterion(part_logits, targets.unsqueeze(
            1).repeat(1, PROPOSAL_NUM).view(-1))
        part_loss = list_loss(part_logits, targets.unsqueeze(
            1).repeat(1, PROPOSAL_NUM).view(-1)).view(-1, PROPOSAL_NUM)
        rank_loss = ranking_loss(top_n_prob, part_loss)
        total_loss =raw_loss + concat_loss + partcls_loss + rank_loss
        
        total_loss.backward()
        optimizer.step()
        
        outs = concat_logits.max(-1)[1]
        running_acc += targets.eq(outs).sum().item()/len(targets)
        running_total_loss += total_loss.item()
        running_raw_loss += raw_loss.item()
        running_concat_loss += concat_loss.item()
        running_partcls_loss += partcls_loss.item()
        running_ranking_loss += rank_loss.item()
        
        
        N = len(train_loader) // 10
        if batch_idx % N == N-1:
            writer.add_scalar('train acc',
                              running_acc/N,
                              epoch*len(train_loader)+batch_idx)
            writer.add_scalar('train total loss',
                              running_total_loss/N,
                              epoch*len(train_loader)+batch_idx)
            writer.add_scalar('train raw loss',
                              running_raw_loss/N,
                              epoch*len(train_loader)+batch_idx)
            writer.add_scalar('train concat loss',
                              running_concat_loss/N,
                              epoch*len(train_loader)+batch_idx)
            writer.add_scalar('train partcls loss',
                              running_partcls_loss/N,
                              epoch*len(train_loader)+batch_idx)
            writer.add_scalar('train rank loss',
                              running_ranking_loss/N,
                              epoch*len(train_loader)+batch_idx)
            running_acc = 0.0
            running_total_loss = 0.0
            running_raw_loss =0.0
            running_concat_loss = 0.0
            running_partcls_loss = 0.0
            running_ranking_loss = 0.0


#-------------------------------------define dev-----------------------------------
def val(fine_grained_cnn_net, cnn_rnn_ctc_net, dev_loader, 
        criterion, epoch, writer):
    epoch_total_loss = 0.0
    epoch_raw_loss = 0.0
    epoch_concat_loss =0.0
    epoch_partcls_loss = 0.0
    epoch_ranking_loss = 0.0
    epoch_acc = 0.0
    fine_grained_cnn_net.eval()
    cnn_rnn_ctc_net.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(dev_loader):
            inputs = batch['videos'].cuda()
            targets = cnn_rnn_ctc_net(inputs)
            targets = targets.max(-1)[1].permute(1,0).contiguous().view(-1).data
            
            n,t,c,h,w = inputs.size()
            inputs = inputs.view(-1,c,h,w)
            raw_logits, concat_logits, part_logits, _, top_n_prob, _ = fine_grained_cnn_net(inputs)
            raw_loss = criterion(raw_logits, targets)
            concat_loss = criterion(concat_logits, targets)
            partcls_loss = criterion(part_logits, targets.unsqueeze(
                1).repeat(1, PROPOSAL_NUM).view(-1))
            part_loss = list_loss(part_logits, targets.unsqueeze(
                1).repeat(1, PROPOSAL_NUM).view(-1)).view(-1, PROPOSAL_NUM)
            rank_loss = ranking_loss(top_n_prob, part_loss)
            total_loss = raw_loss + concat_loss + partcls_loss + rank_loss
            
            outs = concat_logits.max(-1)[1]
            epoch_acc += targets.eq(outs).sum().item()/len(targets)
            epoch_total_loss += total_loss.item()
            epoch_raw_loss += raw_loss.item()
            epoch_concat_loss += concat_loss.item()
            epoch_partcls_loss += partcls_loss.item()
            epoch_ranking_loss += rank_loss.item()
            
    epoch_acc /= len(dev_loader)
    epoch_total_loss /= len(dev_loader)
    epoch_raw_loss /= len(dev_loader)
    epoch_concat_loss /= len(dev_loader)
    epoch_partcls_loss /= len(dev_loader)
    epoch_ranking_loss /= len(dev_loader)
    if writer:
        writer.add_scalar('dev acc', epoch_acc, epoch)
        writer.add_scalar('dev total loss', epoch_total_loss, epoch)
        writer.add_scalar('dev raw loss', epoch_raw_loss, epoch)
        writer.add_scalar('dev concat loss', epoch_concat_loss, epoch)
        writer.add_scalar('dev partcls loss', epoch_partcls_loss, epoch)
        writer.add_scalar('dev ranking loss', epoch_ranking_loss, epoch)
    return epoch_acc, epoch_total_loss



#-------------------------------------main-----------------------------------
def main(device, resume_training):
    os.environ["CUDA_VISIBLE_DEVICES"] = device
    save_dict1 = torch.load(
        '/home/xieliang/Data/sign-language-recognition/save/CNN_RNN_CTC.pth')
    cnn_rnn_ctc_net = save_dict1['net'].cuda()
    for parmas in cnn_rnn_ctc_net.parameters():
        parmas.requires_grad=False

    save_root = '/home/xieliang/Data/sign-language-recognition'
    save_model = os.path.join(save_root, 'save/fine_grained_cnn5.pth')
    save_log = os.path.join(save_root, 'log/fine_grained_cnn5')

    if resume_training:
        save_dict2 = torch.load(save_model)
        start_epoch = save_dict2['epoch']+1
        best_dev_acc = save_dict2['best_dev_acc']
        fine_grained_cnn_net = save_dict2['net'].cuda()
    else:
        start_epoch = 0
        best_dev_acc = 0
        fine_grained_cnn_net = attention_net(VocabSize, PROPOSAL_NUM).cuda()

    LR = 1e-4
    WD = 1e-4
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(fine_grained_cnn_net.parameters(), lr=LR, weight_decay=WD)
    writer = SummaryWriter(save_log)

    # summary(fine_grained_cnn_net, torch.zeros(54, 3, 448, 448).cuda())

    print(f'start training in epoch {start_epoch} with best dev acc {best_dev_acc}')
    for epoch in range(start_epoch, 1000):
        train(fine_grained_cnn_net, cnn_rnn_ctc_net, train_loader,
              optimizer, criterion, epoch, writer)
        dev_acc, dev_loss = val(fine_grained_cnn_net, cnn_rnn_ctc_net, 
            dev_loader, criterion, epoch, writer)
        print(f'epoch: {epoch} | dev acc: {dev_acc} | dev loss: {dev_loss}')

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            torch.save({'epoch': epoch, 'best_dev_acc':best_dev_acc, 
                        'net':fine_grained_cnn_net}, save_model)
            print(f'model saved with best dev acc {best_dev_acc} in epoch {epoch}')
        
        

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default="0", type=str, help='the index of cuda')
    parser.add_argument('--resume_training', action='store_true', help='if action, resume training')
    args = parser.parse_args()
    device = args.device
    resume_training = args.resume_training
    main(device, resume_training)