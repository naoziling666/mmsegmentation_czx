import torch.nn as nn
import sys
import torch.nn.functional as F
from mmseg.registry import MODELS
import torch
import copy
import warnings
import math
def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)

# 在mmseg新加组件的时候，在model中的层要定义在具体组件里面
# 不可以定义在模型组件之外的函数中，负责会导致模型的参数在不同的device
@MODELS.register_module()
class Time_feat_query(nn.Module):
    def __init__(self,  in_index=[1,2,3],num_query=64, embed_dims = 64):
        super(Time_feat_query, self).__init__()
        self.num_query = num_query
        self.in_index = in_index
        self.sampling_offset = nn.Sequential(
                                nn.Linear(embed_dims, embed_dims//2),
                                nn.ReLU(),
                                nn.Linear(embed_dims//2, 2))
        self.scale_weights = nn.Sequential(
                                nn.Linear(embed_dims, embed_dims//2),
                                nn.ReLU(),
                                nn.Linear(embed_dims//2, 1))
        self._init_layers()
    # def _init_layers(self):
    #     self.init_query_bbox = nn.Embedding(self.num_query, 2) # (x,y,w,l)
    #     grid_size = int(math.sqrt(self.num_query))
    #     assert grid_size * grid_size == self.num_query
    #     x = y = torch.arange(grid_size)
    #     xx, yy = torch.meshgrid(x, y, indexing='ij')  # [0, grid_size - 1]
    #     xy = torch.cat([xx[..., None], yy[..., None]], dim=-1)
    #     xy = (xy + 0.5) / grid_size  # [0.5, grid_size - 0.5] / grid_size ~= (0, 1)
    #     with torch.no_grad():
    #         self.init_query_bbox.weight[:, :2] = xy.reshape(-1, 2)  # [Q, 2]
    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            Tensor: The transformed inputs
        """


        inputs = [inputs[i] for i in self.in_index]
        upsampled_inputs = [
            resize(
                input=x,
                size=inputs[0].shape[2:],
                mode='bilinear',
                align_corners=False) for x in inputs
        ]
        inputs = torch.cat(upsampled_inputs, dim=1)


        return inputs

    def _init_layers(self):
        self.init_query_bbox = nn.Embedding(self.num_query, 2) # (x,y,w,l)
        grid_size = int(math.sqrt(self.num_query))
        self.grid_size = grid_size
        assert grid_size * grid_size == self.num_query
        x = y = torch.arange(grid_size)
        xx, yy = torch.meshgrid(x, y, indexing='ij')  # [0, grid_size - 1]
        xy = torch.cat([xx[..., None], yy[..., None]], dim=-1)
        xy = (xy + 0.5) / grid_size  # [0.5, grid_size - 0.5] / grid_size ~= (0, 1)
        with torch.no_grad():
            self.init_query_bbox.weight[:, :2] = xy.reshape(-1, 2)  # [Q, 2]

    def make_query_feat(self, center_points, width, features):
        batch_size = features.shape[0]
        center_point = center_points[0]
        features_out = []
        row_starts = center_point[:, 0]-width
        row_ends = center_point[:, 0]+width
        col_starts = center_point[:, 1]-width
        col_ends = center_point[:, 1]+width
        batch_size = features.shape[0]
        features_per_sample = []
        for i in range(len(row_starts)):
            patch_feature = features[:, row_starts[i]:row_ends[i], col_starts[i]:col_ends[i]].flatten(-2)
            features_per_sample.append(patch_feature)
        features_out = torch.cat(features_per_sample, dim=0).reshape(features.shape[0], len(center_point),-1)

        return features_out



    def fuse_query_feat(self, features, query_bbox, scale_weights, width):
        B = features.shape[0]
        # 将query_bbox按特征图尺寸放大并转换为整数坐标
        centre_points = query_bbox * features.shape[-1]
        centre_points = centre_points.to(dtype=torch.int32)
        
        # 创建一个与original features尺寸相同的零张量来存储新的特征
        new_features = features.clone()  # 这里使用了clone来确保非in-place操作
        
        for b in range(B):
            centre_point = centre_points[b]
            scale_weight = scale_weights[b]
            row_starts = centre_point[:, 0] - width
            row_ends = centre_point[:, 0] + width
            col_starts = centre_point[:, 1] - width
            col_ends = centre_point[:, 1] + width
            
            for i in range(len(row_starts)):
                # 复制原features对应区域的块，并乘以相应的scale_weight
                try:
                    new_features[b, :, row_starts[i]:row_ends[i], col_starts[i]:col_ends[i]] = \
                        features[b, :, row_starts[i]:row_ends[i], col_starts[i]:col_ends[i]] * scale_weight[i]
                except IndexError:
                    # 若选择的索引超出边界，则跳过
                    continue

        return new_features


    def fuse_feat(self,query_bbox, features, features_next): # 尝试设置features的requires_grad = False
        # query_bbox [b, q, 2]
        # query_feat [b, q, embed_dim]
        B = features.shape[0]
        Q = query_bbox.shape[1]
        shape = features.shape[-1]
        width_of_query = int(shape / self.grid_size/2)
        center_point = query_bbox *shape
        center_point = center_point.to(dtype=torch.int32)
        query_feat = self.make_query_feat(center_point, width_of_query, torch.mean(features, dim=1)) 
        sampling_offset = self.sampling_offset(query_feat)
        scale_weight = self.scale_weights(query_feat)
        delta_offset = sampling_offset * width_of_query
        query_bbox_new = (center_point + delta_offset)/shape
        features_time = self.fuse_query_feat(features, query_bbox_new, scale_weight, width_of_query)

        return features_time+features_next


    def forward(self, inputs:list):
        
        inputs_resize = []
        for i in range(len(inputs)):
            inputs_resize.append(self._transform_inputs(inputs[i]))
        query_bbox = self.init_query_bbox.weight.clone()
        query_bbox = query_bbox.repeat((inputs[0][0].shape[0], 1, 1))
        out = self.fuse_feat(query_bbox, inputs_resize[0], inputs_resize[-1])

        return out
            
            
            
            
            
            
            
            
            
            
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
