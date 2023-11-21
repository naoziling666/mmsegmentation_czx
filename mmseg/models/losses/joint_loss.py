import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.registry import MODELS



MODELS.register_module()
class JointLoss(nn.Module):

    def __init__(self,
                 ratio = 0.2,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 foreground_index = [1,2,3],
                 loss_name='loss_joint'):
        """`Focal Loss <https://arxiv.org/abs/1708.02002>`_
        Args:
            use_sigmoid (bool, optional): Whether to the prediction is
                used for sigmoid or softmax. Defaults to True.
            gamma (float, optional): The gamma for calculating the modulating
                factor. Defaults to 2.0.
            alpha (float | list[float], optional): A balanced form for Focal
                Loss. Defaults to 0.5. When a list is provided, the length
                of the list should be equal to the number of classes.
                Please be careful that this parameter is not the
                class-wise weight but the weight of a binary classification
                problem. This binary classification problem regards the
                pixels which belong to one class as the foreground
                and the other pixels as the background, each element in
                the list is the weight of the corresponding foreground class.
                The value of alpha or each element of alpha should be a float
                in the interval [0, 1]. If you want to specify the class-wise
                weight, please use `class_weight` parameter.
            reduction (str, optional): The method used to reduce the loss into
                a scalar. Defaults to 'mean'. Options are "none", "mean" and
                "sum".
            class_weight (list[float], optional): Weight of each class.
                Defaults to None.
            loss_weight (float, optional): Weight of loss. Defaults to 1.0.
            loss_name (str, optional): Name of the loss item. If you want this
                loss item to be included into the backward graph, `loss_` must
                be the prefix of the name. Defaults to 'loss_focal'.
        """
        super().__init__()
        assert reduction in ('none', 'mean', 'sum'), \
            "AssertionError: reduction should be 'none', 'mean' or " \
            "'sum'"
        assert isinstance(loss_weight, float), \
            'AssertionError: loss_weight should be of type float'
        assert isinstance(loss_name, str), \
            'AssertionError: loss_name should be of type str'
        assert isinstance(foreground_index, list), \
            'AssertionError: foreground_index should be of type list'
        assert isinstance(class_weight, list) or class_weight is None, \
            'AssertionError: class_weight must be None or of type list'
        self.ratio = ratio
        self.foreground_index = foreground_index
        self.reduction = reduction
        self.class_weight = class_weight
        self.loss_weight = loss_weight
        self._loss_name = loss_name

    def forward(self,
                pred,
                target,
                reduction_override=None,
                ignore_index=255,
                **kwargs):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction with shape
                (N, C) where C = number of classes, or
                (N, C, d_1, d_2, ..., d_K) with K≥1 in the
                case of K-dimensional loss.
            target (torch.Tensor): The ground truth. If containing class
                indices, shape (N) where each value is 0≤targets[i]≤C−1,
                or (N, d_1, d_2, ..., d_K) with K≥1 in the case of
                K-dimensional loss. If containing class probabilities,
                same shape as the input.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to
                average the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used
                to override the original reduction method of the loss.
                Options are "none", "mean" and "sum".
            ignore_index (int, optional): The label index to be ignored.
                Default: 255
        Returns:
            torch.Tensor: The calculated loss
        """
        assert isinstance(ignore_index, int), \
            'ignore_index must be of type int'
        assert reduction_override in (None, 'none', 'mean', 'sum'), \
            "AssertionError: reduction should be 'none', 'mean' or " \
            "'sum'"
        assert pred.shape == target.shape or \
               (pred.size(0) == target.size(0) and
                pred.shape[2:] == target.shape[1:]), \
               "The shape of pred doesn't match the shape of target"

        original_shape = pred.shape
        bi_pred = pred[0]
        cls_pred = pred[1:]
        fgp = torch.sigmoid(bi_pred)
        clsp = torch.softmax(cls_pred, dim=1)
        num_class = clsp.shape[1]
        joint_prob = torch.clone(clsp)
        for i in range(num_class):
            if i in self.foreground_index:
                joint_prob[:, i:, :, :] = fgp * clsp[:, i:, :, :]
            else:
                joint_prob[:, i, :, :] = (1-fgp).squeeze(dim=1) * clsp[:, i, :, :]
        # # normalization factor, [B x 1 X H X W]
        Z = torch.sum(joint_prob, dim=1, keepdim=True)
        # cls prob, [B, N, H, W]
        p_ci = joint_prob / Z
        
        
    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.
        Returns:
            str: The name of this loss item.
        """
        return self._loss_name
