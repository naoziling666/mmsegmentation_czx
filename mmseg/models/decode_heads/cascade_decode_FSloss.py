import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import copy
import warnings
from mmcv.cnn import ConvModule
from mmengine.device import get_device
from .decode_head import BaseDecodeHead
from typing import List, Tuple
from torch import Tensor
from mmseg.utils import ConfigType, SampleList
from mmseg.registry import MODELS
from mmseg.models.losses import accuracy
import numpy as np
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





class Matrix_Decomposition_2D_Base(nn.Module):
    """Base class of 2D Matrix Decomposition.

    Args:
        MD_S (int): The number of spatial coefficient in
            Matrix Decomposition, it may be used for calculation
            of the number of latent dimension D in Matrix
            Decomposition. Defaults: 1.
        MD_R (int): The number of latent dimension R in
            Matrix Decomposition. Defaults: 64.
        train_steps (int): The number of iteration steps in
            Multiplicative Update (MU) rule to solve Non-negative
            Matrix Factorization (NMF) in training. Defaults: 6.
        eval_steps (int): The number of iteration steps in
            Multiplicative Update (MU) rule to solve Non-negative
            Matrix Factorization (NMF) in evaluation. Defaults: 7.
        inv_t (int): Inverted multiple number to make coefficient
            smaller in softmax. Defaults: 100.
        rand_init (bool): Whether to initialize randomly.
            Defaults: True.
    """

    def __init__(self,
                 MD_S=1,
                 MD_R=64,
                 train_steps=6,
                 eval_steps=7,
                 inv_t=100,
                 rand_init=True):
        super().__init__()

        self.S = MD_S
        self.R = MD_R

        self.train_steps = train_steps
        self.eval_steps = eval_steps

        self.inv_t = inv_t

        self.rand_init = rand_init

    def _build_bases(self, B, S, D, R, device=None):
        raise NotImplementedError

    def local_step(self, x, bases, coef):
        raise NotImplementedError

    def local_inference(self, x, bases):
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        coef = torch.bmm(x.transpose(1, 2), bases)
        coef = F.softmax(self.inv_t * coef, dim=-1)

        steps = self.train_steps if self.training else self.eval_steps
        for _ in range(steps):
            bases, coef = self.local_step(x, bases, coef)

        return bases, coef

    def compute_coef(self, x, bases, coef):
        raise NotImplementedError

    def forward(self, x, return_bases=False):
        """Forward Function."""
        B, C, H, W = x.shape

        # (B, C, H, W) -> (B * S, D, N)
        D = C // self.S
        N = H * W
        x = x.view(B * self.S, D, N)
        if not self.rand_init and not hasattr(self, 'bases'):
            bases = self._build_bases(1, self.S, D, self.R, device=x.device)
            self.register_buffer('bases', bases)

        # (S, D, R) -> (B * S, D, R)
        if self.rand_init:
            bases = self._build_bases(B, self.S, D, self.R, device=x.device)
        else:
            bases = self.bases.repeat(B, 1, 1)

        bases, coef = self.local_inference(x, bases)

        # (B * S, N, R)
        coef = self.compute_coef(x, bases, coef)

        # (B * S, D, R) @ (B * S, N, R)^T -> (B * S, D, N)
        x = torch.bmm(bases, coef.transpose(1, 2))

        # (B * S, D, N) -> (B, C, H, W)
        x = x.view(B, C, H, W)

        return x


class NMF2D(Matrix_Decomposition_2D_Base):
    """Non-negative Matrix Factorization (NMF) module.

    It is inherited from ``Matrix_Decomposition_2D_Base`` module.
    """

    def __init__(self, args=dict()):
        super().__init__(**args)

        self.inv_t = 1

    def _build_bases(self, B, S, D, R, device=None):
        """Build bases in initialization."""
        if device is None:
            device = get_device()
        bases = torch.rand((B * S, D, R)).to(device)
        bases = F.normalize(bases, dim=1)

        return bases

    def local_step(self, x, bases, coef):
        """Local step in iteration to renew bases and coefficient."""
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        numerator = torch.bmm(x.transpose(1, 2), bases)
        # (B * S, N, R) @ [(B * S, D, R)^T @ (B * S, D, R)] -> (B * S, N, R)
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))
        # Multiplicative Update
        coef = coef * numerator / (denominator + 1e-6)

        # (B * S, D, N) @ (B * S, N, R) -> (B * S, D, R)
        numerator = torch.bmm(x, coef)
        # (B * S, D, R) @ [(B * S, N, R)^T @ (B * S, N, R)] -> (B * S, D, R)
        denominator = bases.bmm(coef.transpose(1, 2).bmm(coef))
        # Multiplicative Update
        bases = bases * numerator / (denominator + 1e-6)

        return bases, coef

    def compute_coef(self, x, bases, coef):
        """Compute coefficient."""
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        numerator = torch.bmm(x.transpose(1, 2), bases)
        # (B * S, N, R) @ (B * S, D, R)^T @ (B * S, D, R) -> (B * S, N, R)
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))
        # multiplication update
        coef = coef * numerator / (denominator + 1e-6)

        return coef


class Hamburger(nn.Module):
    """Hamburger Module. It consists of one slice of "ham" (matrix
    decomposition) and two slices of "bread" (linear transformation).

    Args:
        ham_channels (int): Input and output channels of feature.
        ham_kwargs (dict): Config of matrix decomposition module.
        norm_cfg (dict | None): Config of norm layers.
    """

    def __init__(self,
                 ham_channels=512,
                 ham_kwargs=dict(),
                 norm_cfg=None,
                 **kwargs):
        super().__init__()
        self.ham_in = ConvModule(
            ham_channels, ham_channels, 1, norm_cfg=None, act_cfg=None)

        self.ham = NMF2D(ham_kwargs)

        self.ham_out = ConvModule(
            ham_channels, ham_channels, 1, norm_cfg=norm_cfg, act_cfg=None)

    def forward(self, x):
        enjoy = self.ham_in(x)
        enjoy = F.relu(enjoy, inplace=True)
        enjoy = self.ham(enjoy)
        enjoy = self.ham_out(enjoy)
        ham = F.relu(x + enjoy, inplace=True)

        return ham
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.stride = stride
        self.res_plus_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        self.res_plus_norm = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Apply skip connection if needed
        if self.stride != 1 or x.shape[1] != out.shape[1]:
            identity = self.res_plus_conv(x)
            identity = self.res_plus_norm(identity)

        out += identity
        out = self.relu(out)
        return out   
    
class course_decode(nn.Module):
    def __init__(self, in_channels):
        super(course_decode, self).__init__()
        self.in_channel = sum(in_channels)
        self.decode = nn.ModuleList()
        self.decode.append(ResidualBlock(self.in_channel, 512))
        self.decode.append(ResidualBlock(512, 256))
        self.decode.append(ResidualBlock(256, 64))
        self.decode.append(ResidualBlock(64, 2))
    def forward(self, inputs:list):
        inputs = torch.cat(inputs, dim=1)
        for i, layer in enumerate(self.decode):
            if i == 0:
                output = layer(inputs)
            else:
                output = layer(output)
        return output

@MODELS.register_module()
class Cascade_Decode_FSloss(BaseDecodeHead):
    def __init__(self, ham_channels=512, foreground_index = [0,1,2], background_index = [3], ham_kwargs=dict(), **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)
        self.decode_foreground = course_decode(self.in_channels)
        self.ham_channels = ham_channels
        self.foreground_index = foreground_index
        self.background_index = background_index
        self.squeeze = ConvModule(
            sum(self.in_channels)+2,
            self.ham_channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.hamburger = Hamburger(ham_channels, ham_kwargs, **kwargs)

        self.align = ConvModule(
            self.ham_channels,
            self.channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def compute_adjustment(self, inputs, tro):
        B = inputs.shape[0]
        pass
    def predict(self, inputs: Tuple[Tensor], batch_img_metas: List[dict],
                test_cfg: ConfigType) -> Tensor:
        """Forward function for prediction.

        Args:
            inputs (Tuple[Tensor]): List of multi-level img features.
            batch_img_metas (dict): List Image info where each dict may also
                contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Outputs segmentation logits map.
        """
        seg_logits, seg_logits_foreground = self.forward(inputs)
        post_logit_adjust=True
        if post_logit_adjust:
            # tro_post_range = [0.25, 0.5, 0.75, 1, 1.5, 2]
            tro = 0.115
            frequency = np.array([0.15608681, 0.33011826, 0.03359617, 0.48019876])
            adjustments = np.log(frequency ** tro + 1e-12)
            adjustments = torch.tensor(adjustments).to(seg_logits.device)
            seg_logits[:,0,:,:] -= adjustments[0]
            seg_logits[:,1,:,:] -= adjustments[1]
            seg_logits[:,2,:,:] -= adjustments[2]
            seg_logits[:,3,:,:] -= adjustments[3]

        return self.predict_by_feat(seg_logits, batch_img_metas)

    def _stack_batch_gt(self, batch_data_samples: SampleList) -> Tensor:
        gt_semantic_segs = [
            data_sample.gt_sem_seg.data for data_sample in batch_data_samples
        ]
        return torch.stack(gt_semantic_segs, dim=0)
    def loss_by_feat(self, seg_logits: Tensor, seg_logits_foreground: Tensor,
                     batch_data_samples: SampleList) -> dict:
        """Compute segmentation loss.

        Args:
            seg_logits (Tensor): The output from decode head forward function.
            batch_data_samples (List[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `metainfo` and `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        seg_label = self._stack_batch_gt(batch_data_samples)
        
        
        loss = dict()
        seg_logits = resize(
            input=seg_logits,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        seg_logits_foreground = resize(
            input=seg_logits_foreground,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logits, seg_label)
        else:
            seg_weight = None
        seg_label = seg_label.squeeze(1)
        seg_label_foreground = seg_label.clone()
        for item in self.background_index:
            seg_label_foreground[seg_label_foreground==item] = 256
        for item in self.foreground_index:
            seg_label_foreground[seg_label_foreground==item] = 1
        seg_label_foreground[seg_label_foreground==256] = 0
        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode
        seg_logits_foreground_ = seg_logits_foreground.clone()
        seg_logits_foreground_.detach()
        # joint calculate loss start
        softmax = nn.Softmax(dim=1)
        seg_logits_foreground_ =softmax(seg_logits_foreground_)
        # seg_logits_clone = seg_logits.clone()
        # seg_logits_clone.detach()
        seg_logits_final = seg_logits.clone()
        for item in self.background_index:
            for i in range(seg_logits.shape[0]):
                seg_logits_final[i,item,:,:] = seg_logits[i,item,:,:]*seg_logits_foreground_[i,0,:,:]
        for item in self.foreground_index:
            for i in range(seg_logits.shape[0]):
                seg_logits_final[i,item,:,:] = seg_logits[i,item,:,:]*seg_logits_foreground_[i,1,:,:]
        # joint calculate loss end
        # one loss start
        for loss_decode in losses_decode:
            if loss_decode.loss_name not in loss:
                loss[loss_decode.loss_name] = loss_decode(
                    seg_logits_final,
                    seg_label)
            else:
                loss[loss_decode.loss_name] += loss_decode(
                    seg_logits_final,
                    seg_label)
        # one loss end
        # # two loss start
        # loss[losses_decode[0].loss_name] = losses_decode[0](
        #         seg_logits_final,
        #         seg_label,
        #         weight=seg_weight,
        #         ignore_index=self.ignore_index)
        # loss[losses_decode[1].loss_name] = losses_decode[1](
        #         seg_logits_foreground,
        #         seg_label_foreground,
        #         weight=seg_weight,
        #         ignore_index=self.ignore_index)
        # # two loss end
        loss['acc_seg'] = accuracy(
            seg_logits, seg_label, ignore_index=self.ignore_index)
        return loss

    def loss(self, inputs: Tuple[Tensor], batch_data_samples: SampleList,
             train_cfg: ConfigType) -> dict:
        """Forward function for training.

        Args:
            inputs (Tuple[Tensor]): List of multi-level img features.
            batch_data_samples (list[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `img_metas` or `gt_semantic_seg`.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        seg_logits, coarse_foreground = self.forward(inputs)
        losses = self.loss_by_feat(seg_logits, coarse_foreground, batch_data_samples)
        return losses
        
        
    def forward(self, inputs):
        inputs = self._transform_inputs(inputs)
        inputs = [
            resize(
                level,
                size=inputs[0].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners) for level in inputs
        ]
        # 只有两个通道
        coarse_foreground = self.decode_foreground(inputs)
        # coarse_foreground =  resize(coarse_foreground,
        #                             size=inputs[0].shape[2:],
        #                             mode='bilinear',
        #                             align_corners=self.align_corners)

        inputs.append(coarse_foreground)
        inputs = torch.cat(inputs, dim=1)
        x = self.squeeze(inputs)
        # apply hamburger module
        x = self.hamburger(x)

        # apply a conv block to align feature map
        output = self.align(x)
        output = self.cls_seg(output)
        return output, coarse_foreground
        # batch_size 为4时
        # coarse_foreground.shape = [4,2,75,75]
        # output.shape = [4,4,75,75]
        
# if __name__ == "__main__":
#     ham_norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)        
#     decode = Cascade_Decode_test(in_channels=[128, 320, 512],
#         in_index=[1, 2, 3],
#         channels=1024,
#         ham_channels=1024,
#         dropout_ratio=0.1,
#         num_classes=4,
#         norm_cfg=ham_norm_cfg,
#         align_corners=False,
#         loss_decode=dict(
#             type='FocalLoss', use_sigmoid=True, loss_weight=1.0),
#         ham_kwargs=dict(
#             MD_S=1,
#             MD_R=16,
#             train_steps=6,
#             eval_steps=7,
#             inv_t=100,
#             rand_init=True))
#     q = torch.rand((4,64,256, 256))
#     a = torch.rand((4,128,128, 128))
#     b = torch.rand((4,320,64, 64))
#     c = torch.rand((4,512,32, 32))
#     input = [q,a,b,c]
#     out = decode.forward(input)