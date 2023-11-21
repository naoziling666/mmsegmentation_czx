import torch.nn as nn
from mmseg.registry import MODELS
import torch

class GlobalAvgPool2DBaseline(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2DBaseline, self).__init__()

    def forward(self, x):
        x_pool = torch.mean(x.view(x.size(0), x.size(1), x.size(2) * x.size(3)), dim=2)

        x_pool = x_pool.view(x.size(0), x.size(1), 1, 1).contiguous()
        return x_pool
    
    
    
    
@MODELS.register_module()
class SceneRelation(nn.Module):
    def __init__(self,
                 in_channels,
                 channel_list,
                #  out_channels,
                 scale_aware_proj=True):
        super(SceneRelation, self).__init__()
        self.scale_aware_proj = scale_aware_proj

        if scale_aware_proj:
            self.scene_encoder = nn.ModuleList(
                [nn.Sequential(
                    nn.Conv2d(in_channels, channel, 1),
                    nn.ReLU(True),
                    nn.Conv2d(channel, channel, 1),
                ) for channel in channel_list]
            )
            # self.scene_encoder = nn.ModuleList(
            #     [nn.Sequential(
            #         nn.Conv2d(in_channels, out_channels, 1),
            #         nn.ReLU(True),
            #         nn.Conv2d(out_channels, out_channels, 1),
            #     ) for _ in range(len(channel_list))]
            # )
        # else:
        #     # 2mlp
        #     self.scene_encoder = nn.Sequential(
        #         nn.Conv2d(in_channels, out_channels, 1),
        #         nn.ReLU(True),
        #         nn.Conv2d(out_channels, out_channels, 1),
        #     )
        self.content_encoders = nn.ModuleList()
        self.feature_reencoders = nn.ModuleList()
        for c in channel_list:
            self.content_encoders.append(
                nn.Sequential(
                    nn.Conv2d(c, c, 1),
                    nn.BatchNorm2d(c),
                    nn.ReLU(True)
                )
            )
            self.feature_reencoders.append(
                nn.Sequential(
                    nn.Conv2d(c, c, 1),
                    nn.BatchNorm2d(c),
                    nn.ReLU(True)
                )
            )

            # self.content_encoders.append(
            #     nn.Sequential(
            #         nn.Conv2d(c, out_channels, 1),
            #         nn.BatchNorm2d(out_channels),
            #         nn.ReLU(True)
            #     )
            # )
            # self.feature_reencoders.append(
            #     nn.Sequential(
            #         nn.Conv2d(c, out_channels, 1),
            #         nn.BatchNorm2d(out_channels),
            #         nn.ReLU(True)
            #     )
            # )

        self.normalizer = nn.Sigmoid()
        self.gap = GlobalAvgPool2DBaseline()

    def forward(self, features: list):
        scene_feature = self.gap(features[-1])
        content_feats = [c_en(p_feat) for c_en, p_feat in zip(self.content_encoders, features)]
        if self.scale_aware_proj:
            scene_feats = [op(scene_feature) for op in self.scene_encoder]
            relations = [self.normalizer((sf * cf).sum(dim=1, keepdim=True)) for sf, cf in
                         zip(scene_feats, content_feats)]
        else:
            scene_feat = self.scene_encoder(scene_feature)
            relations = [self.normalizer((scene_feat * cf).sum(dim=1, keepdim=True)) for cf in content_feats]

        p_feats = [op(p_feat) for op, p_feat in zip(self.feature_reencoders, features)]

        refined_feats = [r * p for r, p in zip(relations, p_feats)]
        return refined_feats
    
