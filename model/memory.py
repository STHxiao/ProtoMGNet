import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


class PEMR(nn.Module):
    def __init__(self, num_classes, feats_channels, out_channels):
        super(PEMR, self).__init__()
        # initialization
        self.memory = nn.Parameter(torch.cat([
            torch.zeros(num_classes, 1, dtype=torch.float), torch.ones(num_classes, 1, dtype=torch.float),
        ], dim=1), requires_grad=False)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(feats_channels * 3 + 1, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.SyncBatchNorm(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv_mask = nn.Conv2d(feats_channels, 1, kernel_size=1)

    def att_pool(self, x):
        batch, channel, height, width = x.size()
        #[N, D, C, 1]
        input_x = x
        input_x = input_x.view(batch, channel, height*width) # [N, D, C]
        input_x = input_x.unsqueeze(1) # [N, 1, D, C]

        context_mask = self.conv_mask(x) # [N, 1, C, 1]
        context_mask = context_mask.view(batch, 1, height*width) # [N, 1, C]
        context_mask = F.softmax(context_mask, dim=2) # [N, 1, C]
        context_mask = context_mask.unsqueeze(3) # [N, 1, C, 1]
        context = torch.matmul(input_x, context_mask)# [N, 1, D, 1]
        context = context.view(batch, channel, 1, 1) # [N, D, 1, 1]
        return context

    def forward(self, feats, preds=None):
        batch_size, num_channels, h, w = feats.size()
        # wm
        wm = preds.permute(0, 2, 3, 1).contiguous()
        wm = wm.reshape(-1, self.num_classes)
        wm = F.softmax(wm, dim=-1)

        # initialization
        memory_means = self.memory.data[:, 0]
        memory_stds = self.memory.data[:, 1]
        memory = []
        for idx in range(self.num_classes):
            torch.manual_seed(idx)
            cls_memory = torch.normal(
                mean=torch.full((1, self.feats_channels), memory_means[idx]),
                std=torch.full((1, self.feats_channels), memory_stds[idx])
            )
            memory.append(cls_memory)
        memory = torch.cat(memory, dim=0).type_as(wm)

        # --(B*H*W, num_classes) * (num_classes, C) --> (B*H*W, C)
        selected_memory = torch.matmul(wm, memory)
        # calculate selected_memory
        # --(B*H*W, C) --> (B, H, W, C)
        selected_memory = selected_memory.view(batch_size, h, w, num_channels)
        # --(B, H, W, C) --> (B, C, H, W)
        selected_memory = selected_memory.permute(0, 3, 1, 2).contiguous()
        selected = selected_memory

        # att pool
        context = self.att_pool(feats)
        context_proto = context.expand_as(selected_memory)

        # enhancement
        cos_sim_map = F.cosine_similarity(context_proto, selected_memory, dim=1, eps=1e-7)  # b x h x w
        cos_sim_map = cos_sim_map.unsqueeze(1)# b x 1 x h x w
        memory_output = self.bottleneck(torch.concat([feats, context_proto, cos_sim_map, selected_memory], dim=1))

        return memory.data, memory_output, selected

    def update(self, features, segmentation, momentum, ignore_index=255):
        batch_size, num_channels, h, w = features.size()
        # use features to update memory
        segmentation = segmentation.long()
        features = features.permute(0, 2, 3, 1).contiguous() # 占内存
        features = features.view(batch_size * h * w, num_channels)
        clsids = segmentation.unique()
        for clsid in clsids:
            if clsid == ignore_index: continue
            # --(B, H, W) --> (B*H*W,)
            seg_cls = segmentation.view(-1)
            # --extract the corresponding feats: (K, C)
            feats_cls = features[seg_cls == clsid]
            # --update memory
            feats_cls = feats_cls.mean(0)
            mean, std = feats_cls.mean(), feats_cls.std()
            self.memory[clsid][0] = (1 - momentum) * self.memory[clsid][0].data + momentum * mean
            self.memory[clsid][1] = (1 - momentum) * self.memory[clsid][1].data + momentum * std
        memory = self.memory.data.clone()
        dist.all_reduce(memory.div_(dist.get_world_size()))
        self.memory = nn.Parameter(memory, requires_grad=False)