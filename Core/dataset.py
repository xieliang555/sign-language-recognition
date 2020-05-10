import torch
from torch.utils.data import Dataset

import os
import pandas as pd
from PIL import Image
import threading
import numpy as np

    
class PhoenixDataset(Dataset):
    def __init__(self, root, mode, interval, transform=None):
        '''
        Args:
            root: the original root where the downloaded dataset are placed
            mode: 'train', 'dev', 'test'
            interval: read a frame at intervals
        '''
        root = os.path.join(
            root, 'phoenix2014-release/phoenix-2014-multisigner')
        csv_path = os.path.join(
            root, 'annotations/manual/' + mode + '.corpus.csv')
        self.csv_file = pd.read_csv(csv_path)
        self.video_root = os.path.join(
            root, 'features/fullFrame-210x260px/' + mode)
        self.interval = interval
        self.transform = transform

    def __len__(self):
        return len(self.csv_file)
    
    # read frames through multi-threading
    def read_frames(self, frame_paths, iThread, frames, lock):
        frames_tmp = []
        for p in frame_paths:
            frame = Image.open(p)
            if self.transform:
                frame = self.transform(frame)
            frames_tmp.append(frame)
        lock.acquire()
        frames.append((iThread, frames_tmp))
        lock.release()
        
    def __getitem__(self, idx):
        
        video_path = os.path.join(
            self.video_root, self.csv_file.iloc[idx, 0].split('|')[0])

        video_path = os.path.join(video_path, '1')
        paths = sorted(os.listdir(video_path))
        paths = [os.path.join(video_path, p) for i, p in enumerate(paths) if i%self.interval==0]
        
        frames = []
        lock=threading.Lock()
        nFrames_per_thread = 10
        paths = [paths[i:i+nFrames_per_thread] 
                 for i in range(0,len(paths), nFrames_per_thread)]
        threads = [threading.Thread(target=self.read_frames,args=([paths[i], i, frames, lock]))
                      for i in range(len(paths))]
        [thread.start() for thread in threads]
        [thread.join() for thread in threads]
        frames = torch.stack([p for t in sorted(frames) for p in t[1]],dim=0)
        
        annotation = self.csv_file.iloc[idx,0].split('|')[3].lower()
        return {'video': frames, 'annotation': annotation}


class Phoenix_Full_TrackedHand(Dataset):
    def __init__(self, root, mode, interval, fullTransform=None, trackedHandTransform=None):
        root = os.path.join(
            root, 'phoenix2014-release/phoenix-2014-multisigner')
        csv_path = os.path.join(
            root, 'annotations/manual/' + mode + '.corpus.csv')
        self.csv_file = pd.read_csv(csv_path)
        self.fullVideo_root = os.path.join(
            root, 'features/fullFrame-210x260px/' + mode)
        self.trackedHandVideo_root = os.path.join(
            root, 'features/trackedRightHand-92x132px/' + mode)
        self.interval = interval
        self.fullTransform = fullTransform
        self.trackedHandTransform = trackedHandTransform
    
    def __len__(self):
        return len(self.csv_file)
    
    def read_frames(self, paths, iThread, Video, lock, is_full):
        temp = []
        for p in paths:
            img = Image.open(p)
            if is_full:
                img = self.fullTransform(img)
            else:
                img = self.trackedHandTransform(img)
            temp.append(img)
        lock.acquire()
        Video.append((iThread, temp))
        lock.release()
    
    def __getitem__(self, idx):
        fullVideo_dir = os.path.join(
            self.fullVideo_root, self.csv_file.iloc[idx, 0].split('|')[0], '1')
        fullVideo_paths = sorted(os.listdir(fullVideo_dir))
        fullVideo_paths = [
            os.path.join(fullVideo_dir, p) for i, p in enumerate(
                fullVideo_paths) if i%self.interval==0]
        nFrames_per_thread = 30
        fullVideo_paths = [fullVideo_paths[
            i:i+nFrames_per_thread] for i in range(
            0, len(fullVideo_paths), nFrames_per_thread)]
        fullVideo = []
        lock = threading.Lock()
        threads = [threading.Thread(
            target=self.read_frames, args=([
            fullVideo_paths[i], i, fullVideo, lock, True])) for i in range(len(fullVideo_paths))]
        [t.start() for t in threads]
        [t.join() for t in threads]
        fullVideo = torch.stack([p for t in sorted(fullVideo) for p in t[1]], dim=0)
        
        trackedHandVideo_dir = os.path.join(
            self.trackedHandVideo_root, self.csv_file.iloc[idx, 0].split('|')[0], '1')
        trackedHandVideo_paths = sorted(os.listdir(trackedHandVideo_dir))
        trackedHandVideo_paths = [
            os.path.join(trackedHandVideo_dir, p) for i, p in enumerate(
                trackedHandVideo_paths) if i%self.interval==0]
        trackedHandVideo_paths = [trackedHandVideo_paths[
            i:i+nFrames_per_thread] for i in range(
            0, len(trackedHandVideo_paths), nFrames_per_thread)]
        trackedHandVideo = []
        threads = [threading.Thread(
            target=self.read_frames, args=([
                trackedHandVideo_paths[i], i, trackedHandVideo, lock, False])) 
                   for i in range(len(trackedHandVideo_paths))]
        [t.start() for t in threads]
        [t.join() for t in threads]
        trackedHandVideo = torch.stack([p for t in sorted(trackedHandVideo) for p in t[1]], dim=0)
            
        annotation = self.csv_file.iloc[idx,0].split('|')[3].lower()
        return {'fullVideo':fullVideo, 'trackedHandVideo':trackedHandVideo, 'annotation':annotation}
        
        
    
class PhoenixFrame(Dataset):
    '''
        the dataset for frame-wise classification
    '''
    def __init__(self, root, transform=None):
        super(PhoenixFrame, self).__init__()
        root = os.path.join(
            root, 'phoenix2014-release/phoenix-2014-multisigner')
        txt_file_path = os.path.join(
            root, 'annotations/automatic/train.alignment')
        self.txt_file = pd.read_table(txt_file_path)
        self.frame_root = root
        self.transform = transform

    def __len__(self):
        return len(self.txt_file)

    def __getitem__(self,idx):
        frame_path, target = self.txt_file.iloc[idx, 0].split()
        frame_path = os.path.join(self.frame_root, frame_path)
        frame = Image.open(frame_path)
        # combine the 3 hidden state as 1
        target = int(target) // 3
        
        if self.transform:
            frame = self.transform(frame)
            
        return {'frame':frame, 'target':target}
    
    
    
class CSLDataset(Dataset):
    '''
        one signer performs one video(sentence) for one time
        50 signers perform 100 video totally counts 5000 videos
        返校后在学校下载数据集写
    '''
    def __init__(self, root, transform=None):
        self.video_root = os.path.join(root, 'color_frame')
        self.txt_file = pd.read_table(os.path.join(root, 'corpus.txt'))
    
    def __len__(self):
        return 5000
    
    def __getitem__(self, idx):
        pass
        
        
        
        
        
        
        
        
        