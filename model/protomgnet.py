import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from briks import ASPP, TFM, FEF
from memory import PEMR


class ProtoMGNet(nn.Module):
    def __init__(self,
                 mode,
                 in_channels,
                 out_channels,
                 feats_channels,
                 num_classes,
                 aspp_dilations=None,
                 ):
        super(ProtoMGNet, self).__init__()
        assert self.mode in ['TRAIN', 'TEST']
        self.mode = mode
        # backbone
        self.backbone = models.resnet50(pretrained=True)

        # aspp
        self.aspp = ASPP(in_channels=in_channels, out_channels=out_channels, dilations=aspp_dilations)
        # fdn
        self.fdn = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=feats_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.SyncBatchNorm(feats_channels),
            nn.ReLU(inplace=True),
        )
        # memory
        self.memory_module = PEMR(num_classes=num_classes, feats_channels=feats_channels, out_channels=out_channels)
        # tfm
        self.fuse3 = TFM(high_in_channels=1024, low_in_channels=1024, out_channels=256, up_factor=1)
        self.fuse2 = TFM(high_in_channels=512, low_in_channels=512, out_channels=256, up_factor=1)
        self.fuse1 = TFM(high_in_channels=256, low_in_channels=512, out_channels=256, up_factor=2)

        # fef
        self.edge_net = FEF(in_channels_list=[256, 512, 1024], hidden_channels=512) # 3072

        # decoder


    '''forward'''
    def forward(self, x, targets=None, **kwargs):
        img_size = x.size(2), x.size(3)

        # backbone
        backbone_outputs = self.transforminputs(self.backbone_net(x), selected_indices=self.cfg['backbone'].get('selected_indices'))

        # aspp
        feats_aspp = self.aspp(backbone_outputs[-1])
        wm = self.decoder1(feats_aspp)

        # memory
        stored_memory, memory_output, selected = self.memory_module(backbone_outputs[-1], wm)

        # fef
        edge, edge_feat = self.edge_net(backbone_outputs[:-1])

        # tfm
        lateral_outputs = backbone_outputs[:-1]
        lateral_outputs.append(memory_output)
        for i in range(len(lateral_outputs) - 1, 0, -1):
            lateral_outputs[i - 1] = getattr(self, f'fuse{i}')(lateral_outputs[i], lateral_outputs[i - 1])
        lateral_outputs = [F.interpolate(out, size=lateral_outputs[0].size()[2:], mode='bilinear') for out in lateral_outputs]

        # output
        outputs = [edge_feat]
        outputs.append(lateral_outputs)
        preds = self.decoder2(torch.cat(outputs, dim=1)) # 试试只使用最高层进行预测和更新

        if self.mode == 'TRAIN':
            fdn = self.fdn(backbone_outputs[-1])
            fdn = backbone_outputs[-1] + fdn
            with torch.no_grad():
                self.memory_module.update(features=F.interpolate(fdn, size=img_size, mode='bilinear'), segmentation=targets['seg_target'])
                edge = F.interpolate(edge, size=img_size, mode='bilinear')
        return preds, edge