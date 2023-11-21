import torch.nn as nn
import sys
sys.path.insert(0, '/home/ps/CZX/mmsegmentation_czx')
from mmseg.registry import MODELS
import torch
import copy


# def apply_attention(feature_map):
#     batch_size = feature_map.shape[0]
#     channel = feature_map.shape[1]
#     pooling = nn.AdaptiveAvgPool2d((1,1))
#     layer = nn.Sequential(
#                 nn.Linear(channel, channel),
#                 nn.Sigmoid()
#             )
#     feature_pooling = pooling(feature_map)
#     for i in range(batch_size):
#         x = feature_map[i]
#         x_pooling = feature_pooling[i]
#         x_squeeze = x_pooling.flatten()
#         x_fc = layer(x_squeeze)
#         for j in range(len(x)):
#             feature_map[i][j] = feature_map[i][j]*x_fc[j]

#     return feature_map
# 在mmseg新加组件的时候，在model中的层要定义在具体组件里面
# 不可以定义在模型组件之外的函数中，负责会导致模型的参数在不同的device
@MODELS.register_module()
class ChannelAttention(nn.Module):
    def __init__(self, channel_list):
        super(ChannelAttention, self).__init__()
        self.channel_list = channel_list
        self.pooling = nn.AdaptiveAvgPool2d((1,1))
        self.apply_attention = nn.ModuleList()
        for channel in channel_list:
            self.apply_attention.append(nn.Sequential(
                nn.Linear(channel, channel),
                nn.Sigmoid()))
        

    def forward(self, inputs:list):
        outputs = []
        for i in range(len(inputs)):
            feature_pooling = self.pooling(inputs[i])
            feature_squeeze = feature_pooling.squeeze(3).squeeze(2)
            # 比如定义的全连接层输入和输出为64，那么这个全连接层就是对应的是一个64*64的矩阵
            # 所以可以用任意batch_size的特征去做，比如shape为(4,64)，其中batch_size为4，
            # 他就对应的是一个4*64的矩阵
            feature_fc = self.apply_attention[i](feature_squeeze)
            update_feature = torch.mul(inputs[i], feature_fc.unsqueeze(2).unsqueeze(3))
            outputs.append(update_feature)

        return outputs
            
            
            
            
            
            
            
            
            
            
            
            
# # some testing code           
# if __name__ == "__main__":
#     x = torch.rand((2,64,32,32))
#     y = torch.rand((2,128,16,16))
#     z = torch.rand((2,320,8,8))
#     k = torch.rand((2,64,4,4))
#     input = [x,y,z,k]
#     channel_attention = ChannelAttention([64,128,320,64])
#     output = channel_attention.forward(input)
#     print(output.shape)
