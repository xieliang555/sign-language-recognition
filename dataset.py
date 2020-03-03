import torch
from torch.utils.data import Dataset
from torchvision.transforms import RandomResizedCrop

import os
import pandas as pd
from skimage import io
        

    
class ToTensorVideo(object):
    '''
        convert ndarrays to Tensors
    '''
    def __call__(self, video):
        # / 255.0 ?
        video = video / 255.0
        return torch.from_numpy(video)
    

class RandomResizedCropVideo(RandomResizedCrop):
    '''
        crop first , then resize(scale)

        custom transforms for video clip
        Ref: https://github.com/pytorch/vision/tree/master/torchvision/transforms
        Ref: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    '''
    def __init__(self, size, scale = (0.08,1),
                 ratio = (3.0 / 4.0, 4.0 / 3.0),
                 interpolation_mode = 'bilinear'):
        if isinstance(size, tuple):
            self.size = size
        else:
            self.size = (size, size)
        self.scale = scale
        self.ratio = ratio
        self.interpolation_mode = interpolation_mode
    
    def __call__(self, video):
        # transform the video shape from [T, H, W, C] to [C, T, H, W]
        video = video.permute(3, 0, 1, 2)
        i, j, h, w = self.get_params(video, self.scale, self.ratio)
        video = video[:, :, i:i+h, j:j+w]
        video = torch.nn.functional.interpolate(
            video.float(), size = self.size, mode = self.interpolation_mode)
        return video
        
    

class PhoenixDataset(Dataset):
    def __init__(self, root, mode, transform=None):
        '''
        Args:
            root: the original root where the downloaded dataset are placed
            mode: 'train', 'dev', 'test'
            transforms: custom transforms for video
        Return:
            video: [T, H, W, C]
            annotation: str
        '''
        root = os.path.join(
            root, 'phoenix2014-release/phoenix-2014-multisigner')
        csv_path = os.path.join(
            root, 'annotations/manual/' + mode + '.corpus.csv')
        self.csv_file = pd.read_csv(csv_path)
        self.video_root = os.path.join(
            root, 'features/fullFrame-210x260px/' + mode)
        self.transform = transform

    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self, idx):
        video_path = os.path.join(
            self.video_root, self.csv_file.iloc[idx, 0].split('|')[1])
        video = io.imread_collection(video_path).concatenate()
        video = video[0::2,:,:,:]
        annotation = self.csv_file.iloc[idx, 0].split('|')[3].lower()

        if self.transform:
            video = self.transform(video)

        return {'video': video, 'annotation': annotation}