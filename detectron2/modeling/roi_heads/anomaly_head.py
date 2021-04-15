# Copyright (c) Facebook, Inc. and its affiliates.
from typing import List
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d, ConvTranspose2d, ShapeSpec, cat, get_norm
from detectron2.structures import Instances
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry

__all__ = [
    "BaseMaskRCNNHead",
    "MaskRCNNConvUpsampleHead",
    "build_mask_head",
    "ROI_MASK_HEAD_REGISTRY",
]


ROI_MASK_HEAD_REGISTRY = Registry("ROI_MASK_HEAD")
ROI_MASK_HEAD_REGISTRY.__doc__ = """
Registry for mask heads, which predicts instance masks given
per-region features.

The registered object will be called with `obj(cfg, input_shape)`.
"""

def zero_fill_center(module: nn.Module, zero_size: int) -> None:
    """
    9x9 Example:
    xxx xxx xxx
    xxx xxx xxx
    xxx xxx xxx
    xxx 000 xxx
    xxx 000 xxx
    xxx 000 xxx
    xxx xxx xxx
    xxx xxx xxx
    xxx xxx xxx

    Args:
        module (torch.nn.Module): module to initialize.
    """
    with torch.no_grad():
        w = module.weight.data.copy()
        w[zero_size:2*zero_size, zero_size:2*zero_size] = 0
        module.weight = w

def anomaly_inference(pred_vals: torch.Tensor, input_vals: torch.Tensor):
    """
    """
    anomaly_vals = torch.abs(pred_vals - input_vals)
    return anomaly_vals

class BaseHead(nn.Module):
    """
    Implement the basic Mask R-CNN losses and inference logic described in :paper:`Mask R-CNN`
    """

    @configurable
    def __init__(self, *, vis_period: int = 0):
        """
        NOTE: this interface is experimental.

        Args:
            vis_period (int): visualization period
        """
        super().__init__()
        self.vis_period = vis_period

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {"vis_period": cfg.VIS_PERIOD}

    def forward(self, x):
        """
        Args:
            x: input region

        Returns:
        """
        y = self.layers(x)
        if self.training:
            return {"loss": F.mse_loss(y, x, reduction="mean")}
        else:
            return anomaly_inference(x, y)

    def layers(self, x):
        """
        Neural network layers that makes predictions from input features.
        """
        raise NotImplementedError


# To get torchscript support, we make the head a subclass of `nn.Sequential`.
# Therefore, to add new layers in this head class, please make sure they are
# added in the order they will be used in forward().
@ROI_MASK_HEAD_REGISTRY.register()
class AnomalyHead(BaseHead, nn.Sequential):
    """
    """

    @configurable
    def __init__(self, input_shape: ShapeSpec, *, num_classes, conv_dims, conv_norm="", **kwargs):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape (ShapeSpec): shape of the input feature
            num_classes (int): the number of foreground classes (i.e. background is not
                included). 1 if using class agnostic prediction.
            conv_dims (list[int]): a list of N>0 integers representing the output dimensions
                of N-1 conv layers and the last upsample layer.
            conv_norm (str or callable): normalization for the conv layers.
                See :func:`detectron2.layers.get_norm` for supported types.
        """
        super().__init__(**kwargs)
        assert len(conv_dims) >= 1, "conv_dims have to be non-empty!"

        self.conv_norm_relus = []

        cur_channels = input_shape.channels
        for k, conv_dim in enumerate(conv_dims):
            conv = Conv2d(
                cur_channels,
                conv_dim,
                kernel_size=9,
                stride=1,
                padding=4,
                bias=not conv_norm,
                norm=get_norm(conv_norm, conv_dim),
                activation=nn.ReLU(),
            )
            self.add_module("anomaly_fcn{}".format(k + 1), conv)
            self.conv_norm_relus.append(conv)
            cur_channels = conv_dim
        
        #cur_channels = conv_dims[-1]

        # Prediction must have the same shape with the input
        self.predictor = Conv2d(cur_channels, input_shape.channels, kernel_size=1, stride=1, padding=0)

        for layer in self.conv_norm_relus:
            anomaly_fill(layer)

        # use normal distribution initialization for prediction layer
        nn.init.normal_(self.predictor.weight, std=0.001)
        if self.predictor.bias is not None:
            nn.init.constant_(self.predictor.bias, 0)

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        conv_dim = cfg.MODEL.ROI_MASK_HEAD.CONV_DIM
        num_conv = cfg.MODEL.ROI_MASK_HEAD.NUM_CONV
        ret.update(
            conv_dims=[conv_dim] * (num_conv + 1),  # +1 for ConvTranspose
            conv_norm=cfg.MODEL.ROI_MASK_HEAD.NORM,
            input_shape=input_shape,
        )
        if cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK:
            ret["num_classes"] = 1
        else:
            ret["num_classes"] = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        return ret

    def layers(self, x):
        for layer in self:
            x = layer(x)
        return x


def build_mask_head(cfg, input_shape):
    """
    Build a mask head defined by `cfg.MODEL.ROI_MASK_HEAD.NAME`.
    """
    name = cfg.MODEL.ROI_MASK_HEAD.NAME
    return ROI_MASK_HEAD_REGISTRY.get(name)(cfg, input_shape)
