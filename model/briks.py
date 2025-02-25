import torch
import torch.nn as nn
import torch.nn.functional as F

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, dilations, align_corners=False):
        super(ASPP, self).__init__()
        self.align_corners = align_corners
        self.parallel_branches = nn.ModuleList()
        for idx, dilation in enumerate(dilations):
            if dilation == 1:
                branch = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=dilation, bias=False),
                    nn.SyncBatchNorm(out_channels),
                    nn.ReLU(inplace=True),
                )
            else:
                branch = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation, bias=False),
                    nn.SyncBatchNorm(out_channels),
                    nn.ReLU(inplace=True),
                )
            self.parallel_branches.append(branch)
        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.SyncBatchNorm(out_channels),
            nn.ReLU(inplace=True),
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(out_channels * (len(dilations) + 1), out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.SyncBatchNorm(out_channels),
            nn.ReLU(inplace=True),
        )
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        size = x.size()
        outputs = []
        for branch in self.parallel_branches:
            outputs.append(branch(x))
        global_features = self.global_branch(x)
        global_features = F.interpolate(global_features, size=(size[2], size[3]), mode='bilinear', align_corners=self.align_corners)
        outputs.append(global_features)
        features = torch.cat(outputs, dim=1)
        features = self.bottleneck(features)
        return features


class TFM(nn.Module):
    def __init__(self, high_in_channels, low_in_channels, out_channels, up_factor) -> None:
        super().__init__()
        self.up_dwc_low = nn.Sequential(
            nn.Upsample(scale_factor=up_factor),
            nn.Conv2d(low_in_channels, low_in_channels, kernel_size=7, stride=1, padding=3, groups=low_in_channels, bias=False),
            nn.SyncBatchNorm(low_in_channels),
            nn.ReLU(inplace=True)
        )
        self.pwc_low = nn.Sequential(
            nn.Conv2d(low_in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        )
        self.dwc_high = nn.Sequential(
            nn.Conv2d(high_in_channels, high_in_channels, kernel_size=3, stride=1, padding=1, groups=high_in_channels, bias=False),
            nn.SyncBatchNorm(high_in_channels),
            nn.ReLU(inplace=True)
        )
        self.pwc_high = nn.Sequential(
            nn.Conv2d(high_in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        )
        self.conv_fuse = nn.Sequential(
            nn.Conv2d(out_channels*2, out_channels, kernel_size=5, stride=1, padding=2, groups=out_channels, bias=True),
            nn.SyncBatchNorm(out_channels)
        )
        self.out_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.SyncBatchNorm(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, *inp_feats):
        low_feats, high_feats = inp_feats
        low = self.pwc_low(self.up_dwc_low(low_feats))
        high = self.pwc_high(self.dwc_high(high_feats))

        fuse = torch.cat([low, high], dim=1)
        # fuse = channel_shuffle(fuse, fuse.shape[1])
        sig = self.conv_fuse(fuse).sigmoid()
        out = low * sig
        out = self.out_conv(out)
        return out


class FEF(nn.Module):
    def __init__(self, in_channels_list=[256, 512, 1024], hidden_channels=256):
        super(FEF, self).__init__()
        # 1.去掉频域中的卷积层
        # 2.省去拆分实数和复数的步骤 直接对复数进行滤波
        # 3.拟增加一个边缘增强模块
        self.branches = nn.ModuleList()
        for in_channels in in_channels_list:
            self.branches.append(nn.Sequential(
                nn.Conv2d(in_channels, hidden_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.SyncBatchNorm(hidden_channels),
                nn.ReLU(True),
            ))
        self.edge_fuse = nn.Sequential(
                nn.Conv2d(hidden_channels * len(in_channels_list), hidden_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.SyncBatchNorm(hidden_channels),
                nn.ReLU(True),
        )
        self.edge_filter = nn.Parameter(torch.ones((1,hidden_channels, 64, 33)), requires_grad=True)
        self.edge_decoder = nn.Sequential(
            nn.Dropout2d(0.1),
            nn.Conv2d(in_channels=hidden_channels, out_channels=2, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        # 256 512 1024 512
        xsize = x[0].size()[2:]
        assert len(x) == len(self.branches)
        x = [F.interpolate(feat, size=xsize, mode='bilinear', align_corners=False) for feat in x]
        edge, edges_feats = [], []
        for i in range(len(x)):
            edge_feat = self.branches[i](x[i])
            batch, c, h, w = edge_feat.size()
            # 进行二维傅里叶变换
            ffted = torch.fft.rfft2(edge_feat, norm='ortho')
            # 复频域自适应滤波
            selected_ffted = ffted * self.edge_filter
            # 进行逆傅里叶变换
            selected_ffted = torch.fft.irfft2(selected_ffted, s=(h, w), norm='ortho')
            edges_feats.append(selected_ffted)
        edge_feats = torch.cat(edges_feats, dim=1)
        edge_feats = self.edge_fuse(edge_feats)
        edge = self.edge_decoder(edge_feats)
        return edge, edge_feats