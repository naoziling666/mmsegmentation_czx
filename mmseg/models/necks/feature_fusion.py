import torch.nn as nn
import sys
from mmseg.registry import MODELS
import torch
import copy



@MODELS.register_module()
class Feature_Fusion(nn.Module):
    def __init__(self, channel_list, r=4):
        super(Feature_Fusion, self).__init__()
        self.loal_attentions = nn.ModuleList()
        self.global_attentions = nn.ModuleList()
        for channels in channel_list:
            inter_channels = int(channels // r)
    
            # 局部注意力
            local_att = nn.Sequential(
                nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(inter_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(channels),
            )
            self.loal_attentions.append(local_att)
 
            # 全局注意力
            global_att = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),  # senet中池化
                nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(inter_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(channels),
            )
            self.global_attentions.append(global_att)
 
        self.sigmoid = nn.Sigmoid()

        

    def forward(self, features):
        inputs_image = features[0]
        inputs_flow = features[1]
        inputs = []
        for i in range(len(inputs_image)):
            inputs.append(inputs_image[i]+inputs_flow[i])
        outputs = []
        for i in range(len(inputs_image)):
            xl = self.loal_attentions[i](inputs[i])
            xg = self.global_attentions[i](inputs[i])
            xlg = xl + xg
            attention = self.sigmoid(xlg)
            outputs.append(inputs[i] * attention)
        return outputs