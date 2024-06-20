
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS
# from .utils import get_class_weight, weight_reduce_loss


@MODELS.register_module()
class EQL_loss(nn.Module):


    def __init__(self,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 loss_name='loss_eql',
                 avg_non_ignore=False):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        # self.class_weight = get_class_weight(class_weight)
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

        valid_mask = (label != ignore_index)
        label[label==255] = 0
        label_one_hot = torch.nn.functional.one_hot(label)
        loss_binary = F.binary_cross_entropy_with_logits(cls_score, label)

        return loss_binary

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
    



# some testing code 

if __name__ == "__main__":
    input = torch.rand((4,4,600,600))
    target = torch.randint(0,4,(600,600))
    loss = EQL_loss()
    loss.forward(input, target)