
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS
from .utils import get_class_weight, weight_reduce_loss


@MODELS.register_module()
class Banlanced_Softmax_binary(nn.Module):
    """CrossEntropyLoss.

    Args:
        use_sigmoid (bool, optional): Whether the prediction uses sigmoid
            of softmax. Defaults to False.
        use_mask (bool, optional): Whether to use mask cross entropy loss.
            Defaults to False.
        reduction (str, optional): . Defaults to 'mean'.
            Options are "none", "mean" and "sum".
        class_weight (list[float] | str, optional): Weight of each class. If in
            str format, read them from a file. Defaults to None.
        loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
        loss_name (str, optional): Name of the loss item. If you want this loss
            item to be included into the backward graph, `loss_` must be the
            prefix of the name. Defaults to 'loss_ce'.
        avg_non_ignore (bool): The flag decides to whether the loss is
            only averaged over non-ignored targets. Default: False.
            `New in version 0.23.0.`
    """

    def __init__(self,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 loss_name='loss_bs',
                 avg_non_ignore=False):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = get_class_weight(class_weight)
        self.avg_non_ignore = avg_non_ignore
        if not self.avg_non_ignore and self.reduction == 'mean':
            warnings.warn(
                'Default ``avg_non_ignore`` is False, if you would like to '
                'ignore the certain label and average loss over non-ignore '
                'labels, which is the same with PyTorch official '
                'cross_entropy, set ``avg_non_ignore=True``.')

        self._loss_name = loss_name

    def extra_repr(self):
        """Extra repr."""
        s = f'avg_non_ignore={self.avg_non_ignore}'
        return s

    def forward(self,
                cls_score,
                label,
                reduction_override=None,
                ignore_index=255,
                **kwargs):
        """Forward function."""
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        label_one_hot = F.one_hot(label, num_classes=cls_score.shape[0])
        cls_score = cls_score.permute(0,2,3,1)
        loss = F.binary_cross_entropy_with_logits(cls_score, label_one_hot, reduction = none, ignore_index=255)
        loss = loss.mean(dim=2)
        num_0 = torch.sum(label == 0).item()/torch.sum(label!=ignore_index).item()
        num_1 = torch.sum(label == 1).item()/torch.sum(label!=ignore_index).item()
        num_2 = torch.sum(label == 2).item()/torch.sum(label!=ignore_index).item()
        num_3 = torch.sum(label == 3).item()/torch.sum(label!=ignore_index).item()

        loss[label==0] *= num_0
        loss[label==1] *= num_1
        loss[label==2] *= num_2
        loss[label==3] *= num_3
        # cls_score_sum = cls_score_exp.sum(dim=1).unsqueeze(1)
        # cls_score_exp = cls_score_exp/cls_score_sum
        # valid_mask = (label != ignore_index)
        # label[label==255] = 0
        # label_one_hot = torch.nn.functional.one_hot(label)
        # loss = cls_score_exp*label_one_hot.permute(0,3,1,2)
        # loss = loss.sum(dim=1)*valid_mask
        loss *= self.loss_weight
        loss = loss.mean()
        return loss

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