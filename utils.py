import numpy as np
import os
import pandas as pd
import glob

from torchtext.data.metrics import bleu_score
from jiwer import wer


def get_csv(root):
    root = os.path.join(root, 'phoenix2014-release/phoenix-2014-multisigner')
    root = os.path.join(root, 'annotations/manual/train.corpus.csv')
    return pd.read_csv(root)


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
