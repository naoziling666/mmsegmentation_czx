import math 
import torch.nn as nn
from .decode_head import BaseDecodeHead
from mmseg.registry import MODELS
from ..utils import resize

@MODELS.register_module()
class AssymetricDecoder(BaseDecodeHead):
    def __init__(self,
                 in_feat_output_strides=(2, 4, 8, 16),
                 out_feat_output_stride=2,
                 norm_fn=nn.BatchNorm2d,
                 num_groups_gn=None,
                 **kwargs):
        super(AssymetricDecoder, self).__init__(**kwargs)
        if norm_fn == nn.BatchNorm2d:
            norm_fn_args = dict(num_features=self.channels)
        elif norm_fn == nn.GroupNorm:
            if num_groups_gn is None:
                raise ValueError('When norm_fn is nn.GroupNorm, num_groups_gn is needed.')
            norm_fn_args = dict(num_groups=num_groups_gn, num_channels=self.channels)
        else:
            raise ValueError('Type of {} is not support.'.format(type(norm_fn)))

        self.out_channels = self.num_classes
        self.blocks = nn.ModuleList()
        for in_feat_os in in_feat_output_strides:
            num_upsample = int(math.log2(int(in_feat_os))) - int(math.log2(int(out_feat_output_stride)))
            # num_upsample = 0,1,2,3
            num_layers = num_upsample if num_upsample != 0 else 1
            # num_layers = 1,1,2,3
            self.blocks.append(nn.Sequential(*[
                nn.Sequential(
                    nn.Conv2d(self.in_channels[num_upsample] if idx == 0 else self.channels, self.channels, 3, 1, 1, bias=False),
                    norm_fn(**norm_fn_args) if norm_fn is not None else nn.Identity(),
                    nn.ReLU(inplace=True),
                    nn.UpsamplingBilinear2d(scale_factor=2) if num_upsample != 0 else nn.Identity(),
                )
                for idx in range(num_layers)]))

    def forward(self, feat_list: list):
        inner_feat_list = []
        h, w = feat_list[0].shape[2], feat_list[0].shape[3]
        # print(h, w)
        for idx, block in enumerate(self.blocks):
            decoder_feat = block(feat_list[idx])
            decoder_feat = resize(
                decoder_feat,
                size=(w, h),
                mode='bilinear',
                align_corners=self.align_corners)
            inner_feat_list.append(decoder_feat)

        out_feat = sum(inner_feat_list) / 4.
        output = self.cls_seg(out_feat)
        return output
    
