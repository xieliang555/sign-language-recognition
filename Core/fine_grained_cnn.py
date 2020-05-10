from torch import nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable

import os
import numpy as np

from . import cnn as resnet
from .anchors import generate_default_anchor_maps, hard_nms

PROPOSAL_NUM = 6
CAT_NUM = 4

class ProposalNet(nn.Module):
    def __init__(self):
        super(ProposalNet, self).__init__()
        # 2048->512
        self.down1 = nn.Conv2d(512, 128, 3, 1, 1)
        self.down2 = nn.Conv2d(128, 128, 3, 2, 1)
        self.down3 = nn.Conv2d(128, 128, 3, 2, 1)
        self.ReLU = nn.ReLU()
        self.tidy1 = nn.Conv2d(128, 6, 1, 1, 0)
        self.tidy2 = nn.Conv2d(128, 6, 1, 1, 0)
        self.tidy3 = nn.Conv2d(128, 9, 1, 1, 0)

    def forward(self, x):
        batch_size = x.size(0)
        d1 = self.ReLU(self.down1(x))
        d2 = self.ReLU(self.down2(d1))
        d3 = self.ReLU(self.down3(d2))
        t1 = self.tidy1(d1).view(batch_size, -1)
        t2 = self.tidy2(d2).view(batch_size, -1)
        t3 = self.tidy3(d3).view(batch_size, -1)
        # confidence of default proposal box
        return torch.cat((t1, t2, t3), dim=1)


class attention_net(nn.Module):
    def __init__(self, vocabSize, topN=4):
        super(attention_net, self).__init__()
        # 50->18
        self.pretrained_model = resnet.resnet18(pretrained=True)
        self.pretrained_model.avgpool = nn.AdaptiveAvgPool2d(1)
        # ? 4->1, 200->vocabSize+1
        self.pretrained_model.fc = nn.Linear(512 * 1, vocabSize+1)
        self.proposal_net = ProposalNet()
        self.topN = topN
        # ? 2048->512, 200->vocabSize+1
        self.concat_net = nn.Linear(512 * (CAT_NUM + 1), vocabSize+1)
        # ? 4->1, 200->vocabSize+1
        self.partcls_net = nn.Linear(512 * 1, vocabSize+1)
        # edge_anchors: [426,4]
        _, edge_anchors, _ = generate_default_anchor_maps()
        # ? 224->112
        self.pad_side = 112
        # ? 224->112
        self.edge_anchors = (edge_anchors + 112).astype(np.int)

    def forward(self, x):
        resnet_out, rpn_feature, feature = self.pretrained_model(x)
        x_pad = F.pad(x, (self.pad_side, self.pad_side, self.pad_side, self.pad_side), mode='constant', value=0)
        batch = x.size(0)
        # [n, 426]
        rpn_score = self.proposal_net(rpn_feature.detach())
        all_cdds = [
            np.concatenate((x.reshape(-1, 1), self.edge_anchors.copy(), np.arange(0, len(x)).reshape(-1, 1)), axis=1)
            for x in rpn_score.data.cpu().numpy()]
        top_n_cdds = [hard_nms(x, topn=self.topN, iou_thresh=0.25) for x in all_cdds]
        # top_n_cdds :[n*t, topN, 6]
        top_n_cdds = np.array(top_n_cdds)
        # top_n_index: [n*t, topN]
        top_n_index = top_n_cdds[:, :, -1].astype(np.int)
        # ?删除.cuda()
        top_n_index = torch.from_numpy(top_n_index).cuda()
        # [n*t, topN]
        top_n_prob = torch.gather(rpn_score, dim=1, index=top_n_index)
        # ? 224->112
        # ? 删除.cuda()
        part_imgs = torch.zeros([batch, self.topN, 3, 112, 112]).cuda()
        for i in range(batch):
            for j in range(self.topN):
                [y0, x0, y1, x1] = top_n_cdds[i][j, 1:5].astype(np.int)
                # ? 224->112
                part_imgs[i:i + 1, j] = F.interpolate(x_pad[i:i + 1, :, y0:y1, x0:x1], size=(112, 112), mode='bilinear',
                                                      align_corners=True)
        # ? 224->112
        part_imgs = part_imgs.view(batch * self.topN, 3, 112, 112)
        _, _, part_features = self.pretrained_model(part_imgs.detach())
        part_feature = part_features.view(batch, self.topN, -1)
        part_feature = part_feature[:, :CAT_NUM, ...].contiguous()
        part_feature = part_feature.view(batch, -1)
        # raw_logits: [N*T, C]
        # concat_logits: [N*T, C]
        # part_logits: [N*T*topN, C]
        # concat_out: [N*T, F]
        concat_out = torch.cat([part_feature, feature], dim=1)
        concat_logits = self.concat_net(concat_out)
        raw_logits = resnet_out
        part_logits = self.partcls_net(part_features)
        #part_logits = self.partcls_net(part_features).view(batch, self.topN, -1)
        #top_n_index for visualize bbx
        return [raw_logits, concat_logits, part_logits, top_n_index, top_n_prob, concat_out]


def list_loss(logits, targets):
    temp = F.log_softmax(logits, -1)
    loss = [-temp[i][targets[i].item()] for i in range(logits.size(0))]
    return torch.stack(loss)


def ranking_loss(score, targets, proposal_num=PROPOSAL_NUM):
    loss = Variable(torch.zeros(1).cuda())
    batch_size = score.size(0)
    for i in range(proposal_num):
        targets_p = (targets > targets[:, i].unsqueeze(1)).type(torch.cuda.FloatTensor)
        pivot = score[:, i].unsqueeze(1)
        loss_p = (1 - pivot + score) * targets_p
        loss_p = torch.sum(F.relu(loss_p))
        loss += loss_p
    return loss / batch_size
