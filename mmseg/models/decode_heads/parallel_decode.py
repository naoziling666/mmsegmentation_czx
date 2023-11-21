import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import copy
import warnings
from mmcv.cnn import ConvModule
from mmengine.device import get_device
from .decode_head import BaseDecodeHead

from mmseg.registry import MODELS


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
class Cascade_Decode(BaseDecodeHead):
    def __init__(self, ham_channels=512, ham_kwargs=dict(), **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)
        self.decode_foreground = course_decode(self.in_channels)
        self.ham_channels = ham_channels

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

        # inputs.append(coarse_foreground)
        # inputs = torch.cat(inputs, dim=1)
        x = self.squeeze(inputs)
        # apply hamburger module
        x = self.hamburger(x)

        # apply a conv block to align feature map
        output = self.align(x)
        output = self.cls_seg(output)
        
        
        return output