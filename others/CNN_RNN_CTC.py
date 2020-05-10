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

from models import cnn_rnn
from dataset import PhoenixDataset



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

# scale?
transform = transforms.Compose([
    transforms.RandomResizedCrop(FrameSize, scale = (0.8,1)),
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



# define train
def train(net, train_loader, criterion, optimizer, epoch, writer, device):
    net.train()
    running_wer = 0.0
    running_loss = 0.0
    for batch_idx, batch in enumerate(train_loader):
        inputs = batch['videos'].to(device)
        targets = batch['annotations'].permute(1,0).contiguous().to(device)
        input_lens = batch['video_lens'].to(device)
        target_lens = batch['annotation_lens'].to(device)
        
        optimizer.zero_grad()
        outs = net(inputs)
        loss = criterion(outs, targets, input_lens, target_lens)
        loss.backward()
        
#         # ?
#         print('inputs:', inputs.shape)
#         print('targets:', ' '.join([TRG.vocab.itos[i] for i in targets[0]]))
#         outs_ = outs.max(-1)[1].permute(1,0)[0]
#         print(outs_)
#         print('outs:', ' '.join(TRG.vocab.itos[i] for i in outs_ if i != VocabSize))
        
        # 为什么clip后仍然出现NaN? 注释掉看
#         torch.nn.utils.clip_grad_norm_(net.parameters(), 2, 'inf')

        flag = False
        for name, param in net.named_parameters():
            if torch.isnan(param.grad).any():
                flag = True
                break
                
                
#             if torch.isnan(param.grad).any():
#                 print('1batch_idx:', batch_idx)
#                 writer.add_video('video', inputs, epoch*(len(train_loader)+batch_idx))
#                 writer.add_text('text', ' '.join([TRG.vocab.itos[i] for i in targets[0]]),
#                                epoch*(len(train_loader)+batch_idx))
#                 print(f'1 find NaN in param.grad of {name} layer')
#             if torch.isnan(param.data).any():
#                 print('1 batch_idx:', batch_idx)
#                 print(f'1 find NaN in param.data of {name} layer')
                
            # assert后这个epoch结束，运行val()函数 ？ jupyter 报错也继续运行
#             assert not torch.isnan(param.data).any(), f'find Nan in param of {name} layer'
#             assert not torch.isnan(param.grad).any(), f'find NaN in grad of {name} layer'
            

    
        if flag:
            print(batch_idx)
            writer.add_video('video', inputs, epoch*(len(train_loader)+batch_idx))
            writer.add_text('text', ' '.join([TRG.vocab.itos[i] for i in targets[0]]),
                            epoch*(len(train_loader)+batch_idx))
            continue
            
        optimizer.step()
        
        
        outs = outs.max(-1)[1].permute(1,0).contiguous().view(-1)
        # best ctc_decoder?
        outs = ' '.join([TRG.vocab.itos[k] for k, _ in groupby(outs) if k != VocabSize])
        targets = targets.view(-1)
        targets = ' '.join([TRG.vocab.itos[k] for k in targets])
        running_wer += wer(targets, outs, standardize=True)
        running_loss += loss.item()
        

        # ?
        if batch_idx % 300 == 299:
            writer.add_scalar('train wer',
                              running_wer/300,
                              epoch*len(train_loader)+batch_idx)
            writer.add_scalar('train loss',
                              running_loss/300,
                              epoch*len(train_loader)+batch_idx)
#             writer.add_histogram('feature.0.weight',
#                                 net.state_dict()['features.0.weight'],
#                                 epoch*len(train_loader)+batch_idx)
        
            running_wer = 0.0
            running_loss = 0.0    
            
            
# define val
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
            epoch_loss += loss.item()
            
            outs = outs.max(-1)[1].permute(1,0).contiguous().view(-1)
            outs = ' '.join([TRG.vocab.itos[k] for k, _ in groupby(outs) if k != VocabSize])
            targets = targets.view(-1)
            targets = ' '.join([TRG.vocab.itos[k] for k in targets])
            epoch_wer += wer(targets, outs, standardize=True)

        epoch_wer /= len(val_loader)
        epoch_loss /= len(val_loader)
        if writer:
            writer.add_scalar('val wer', epoch_wer, epoch)
            writer.add_scalar('val loss', epoch_loss, epoch)
          
    return epoch_loss, epoch_wer


# train and val
def train_val():
    nHID = 1024
    nLAYER = 1
    LR = 1e-4
    WD = 1e-4
    # ?
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # ?
    # net = seq2seq.CNN_GRU(VocabSize, nHID, nLAYER, dropout=0.5).to(device)
    # ?
    # net.apply(init)
    # ?
    net = torch.load('./save/CNN_GRU_CTC_total9.pth', map_location=device)
    criterion = nn.CTCLoss(blank=VocabSize)
    optimizer = optim.Adam(net.parameters(), lr = LR, weight_decay=WD)
    # ?
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [60,100])
    # ?
    writer = SummaryWriter('./log/CNN_GRU_CTC_total10')
    nEPOCH = 10000
    # ?
    best_wer = 0.7
    # ?
    for epoch in range(0,nEPOCH):
        lr_scheduler.step()
        # ?
        train(net, train_loader, criterion, optimizer, epoch, writer, device)
        val_loss, val_wer = val(net, val_loader, criterion, epoch, writer, device)
        print(f'epoch:{epoch} | val_loss:{val_loss} | val_loss:{val_wer}')

        if val_wer < best_wer:
            best_wer = val_wer
            torch.save(net, './save/CNN_GRU_CTC_total9.pth')
            writer.add_hparams(hparam_dict={'epoch':epoch, 'val wer':val_wer},
                               metric_dict={'best val wer':best_wer})
            print(f'model saved with best wer: {best_wer} in epoch {epoch}')  
            
            
def test():
    criterion = nn.CTCLoss(blank=VocabSize)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_ft = torch.load('./save/CNN_GRU_CTC_total9.pth', map_location=device)
    test_loss, test_wer = val(model_ft, test_loader, criterion, epoch=0, writer=None, device=device)
    print(f'test loss:{test_loss} | test wer:{test_wer}')
           
        
        

if __name__ == '__main__':
#     train_val()
    test()
    
    
    
    
    
    '''
    
    some trick:
    时间复杂度：输入：2*180*3*224*224， num_worker=10, pin_memory=true, 0<gpu_utils<1
    空间复杂的：输入: 2*180*3*224*224, 6000~7000, 看序列长度
    训练第一个epoch前面loss下降后面出现NaN（梯度爆炸？）
        1） lr从1e-4 -> 1e-6 学习速度太慢，loss下降后基本不变，仍然出现cnn第一层conv出现NaN
        2） 梯度裁剪: 仍然出现grad NaN(feature.0.weight正常)
        3)  参数初始化：仍然出现NaN
        4） 使用GN: 仍然出现NaN
        5) 神经网络层数(减小一层GRU层): grad仍然出现NaN，发现是一个样本导致NaN。但是loss没有再出现NaN
        6) 某一个batch导致梯度爆炸，从而导致参数出现NaN，进而lossNaN, 为什么这个batch会导致梯度爆炸（参数累计的原因还是样本的原因）？（work）
    过拟合（train wer 0.2674, val wer:0.6; train loss: 1.066, wer loss: 3.34）？
        1) assert后没有训练后面的数据，导致数据集减少（wer减少7%， work）
        2）先在小数据集上找到最优的时间复杂度，空间复杂度，结果（样本规模：1->测试集->训练集）
        3) 数据集增强（randomreisizedcrop，收敛变慢，wer无明显变化）
        4) ImageNet与手语相差大导致过拟合，使用参数初始化代替预训练模型(从头训练)
        5) lr schedule
        6) weight decay
        7) 使用ResNet代替AlexNet（work）
        8) 增加layer
        9）减小词典大小
    一些调试方法
        1) 使用一个样本快速看算法是否有问题
        2）使用shuffle=False检查数据集是否有问题
        3）小样本集（测试集作为训练集）找超参数
        4) 增加网络层数可以提升精度，前提要防止梯度爆炸和梯度消失，过拟合
    收敛速度：
        1）预训练模型可以加快收敛（work）
        2）即使只训练一个样本，当模型很大时，从零训练起也很难训练
        3）输入长度变为1/2，相比原长度加快收敛（减少学习一些不必要的冗余）
        4）输入长度变为1/4，收敛最快，loss最低
        5) 输入长度变为1/6， 收敛速度最快，loss最低
    继续训练：
        1）导入模型
        2）更改起始epoch
        3）更改best_wer
    速度太慢：
        1）I/O 多线程（work？）
        2）网络延迟，使用python脚本代替jupyter notebook（work）
        3) 重新启动
        4）num_worker与batch_size（注意不是视频帧数）最好1:1
    改进：
        1) 可视化
        2）transformer -> lstm
        3) fine-grained
    
    '''