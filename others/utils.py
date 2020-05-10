import numpy as np
import os
import pandas as pd
import glob

from torchtext.data.metrics import bleu_score
from jiwer import wer



def DatasetStatistic(root, mode='train'):
    '''
    analysis the statistic of the RWTH-PHOENIX-Weather 2014 dataset
    Args:
        root: the whole data root
        mode: 'train','dev', 'test'
    return: 
        max_len: maximum video length 
    '''
    
    root = os.path.join(root, 'phoenix2014-release/phoenix-2014-multisigner')
    video_root = os.path.join(root, 'features/fullFrame-210x260px/' + mode)
    csv_path = os.path.join(root, 'annotations/manual/' + mode + '.corpus.csv')

    csv_file = pd.read_csv(csv_path)
    video_paths = [os.path.join(video_root, csv_file.iloc[i, 0].split('|')[1])
                   for i in range(csv_file.shape[0])]
    
    video_lens = [len(glob.glob(path)) for path in video_paths]
    return [max(video_lens), min(video_lens),
            np.mean(video_lens), np.std(video_lens)]


def itos(idx_seq, TRG):
    '''
    Description:
        denumericalize: convert the index sequence to the text sequence
    '''
    return [TRG.vocab.itos[idx] for idx in idx_seq]


# √
def bleu_count(outputs, targets, TRG):
    '''
    corpus level
    shape:
        outputs: [T, N, E]
        targets: [T, N]
    '''
    outputs = outputs.max(2)[1].transpose(0,1)
    targets = targets.transpose(0,1)
    
    mask = targets.ne(TRG.vocab.stoi['<pad>'])
    outputs = outputs.masked_select(mask)
    targets = targets.masked_select(mask)
    
    candidate_corpus = [itos(outputs, TRG)]
    references_corpus = [[itos(targets, TRG)]]
    return bleu_score(candidate_corpus, references_corpus)


def wer_count(outputs, targets, TRG):
    '''
    word error rate
    Ref: https://pypi.org/project/jiwer/
    shape:
        outputs: [T, N, E]
        targets: [T, N]
    '''
    outputs = outputs.max(2)[1].transpose(0,1)
    targets = targets.transpose(0,1)
    
    mask = targets.ne(TRG.vocab.stoi['<pad>'])
    outputs = outputs.masked_select(mask)
    targets = targets.masked_select(mask)
    
    candidate_corpus = [' '.join(itos(outputs, TRG))]
    reference_corpus = [' '.join(itos(targets, TRG))]
    return wer(reference_corpus, candidate_corpus, standardize = True)



# ?
def uniform_temporal_segment(src, src_padding_mask, clip_size):
    '''
        equally split video to clips along temporal dimension
        src: [N, C, T, H, W]
        src_padding_mask: [N, T]
    Return:
        src: [NCLIP,N,C,clip_size,H,W]
        src_padding_mask: [N, NCLIP]
    '''
    src = src.split(clip_size, 2)
    src_padding_mask = src_padding_mask[:,::clip_size]
    return src, src_padding_mask


# ?
def overlap_temporal_segment(src, src_padding_mask, clip_size):
    '''
        split video to clips using sliding window with 50% overlap
    '''
    N, C, T, H, W = src.size()
    NCLIP = (T-clip_size) // (clip_size // 2) +1
    ret = torch.zeros(NCLIP, N, C, clip_size, H, W)
    for i in range(NCLIP):
        ret[i,...] = src[:,:, i*clip_size//2:i*clip_size//2+clip_size, :,:]
    src_padding_mask = src_padding_mask[:,::clip_size//2][:,:NCLIP]
    return ret, src_padding_mask
