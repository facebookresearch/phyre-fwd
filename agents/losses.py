# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The different loss functions used in training the forward models."""
import numpy as np
import torch
from torch import nn

import phyre


def reshape_gt_to_pred(pred_pix, gt_pix):
    """Resize the gt_pix to same size as pred_pix if not.
    Args:
        pred_pix ([Bx7xH1xW1])
        gt_pix ([BxH2xW2])
    Returns:
        gt_pix BxH1xW1
    """
    if pred_pix.shape[2:] == gt_pix.shape[1:]:
        return gt_pix
    gt_pix = nn.functional.interpolate(
        # None of the interpolations are implemented for Long so need
        # convert to float for interpolation. Only use nearest
        # interpolation so that it can be easily converted back to int
        torch.unsqueeze(gt_pix, axis=1).to(torch.float),
        size=pred_pix.shape[2:], mode='nearest') \
        .to(torch.long).to(pred_pix.device).squeeze(1) \
        .detach()
    return gt_pix


class PixCELoss(nn.Module):
    """Pixel level cross entropy loss."""
    def __init__(self, wt_fn, loss_type):
        super().__init__()
        self.wt_fn = getattr(self, wt_fn)
        self.loss_fn = getattr(self, loss_type)

    @classmethod
    def _pix_wts_default(cls, out_dim, gt_pix):
        """This pixel weighting was used until 2020-01-21."""
        cls_weights = np.ones((out_dim, ), dtype=np.float32)
        cls_weights[0] = 0.1  # 10x lower, as it degenerates to white bgs
        return torch.as_tensor(cls_weights, device=gt_pix.device)

    @classmethod
    def _pix_wts_count_reciprocal(cls, out_dim, gt_pix):
        """This pixel weighting is default from 2020-01-22.
        Gave better results on 0018 template."""
        uniq_elts, uniq_cnts = torch.unique(gt_pix, return_counts=True)
        uniq_elts = uniq_elts.cpu().numpy().tolist()
        uniq_cnts = uniq_cnts.cpu().numpy().tolist()
        cls_weights = np.zeros((out_dim, ), dtype=np.float32)
        for i in range(out_dim):
            if i in uniq_elts:
                cls_weights[i] = 1.0 / uniq_cnts[uniq_elts.index(i)]
        return torch.as_tensor(cls_weights, device=gt_pix.device)

    def _per_pixel_softmax(self, pred_pix, gt_pix):
        """
        Args:
            pred_pix ([Bx7xH1xW1])
            gt_pix ([BxH2xW2])
        """
        pred_pix = pred_pix.permute(0, 2, 3, 1)
        out_dim = pred_pix.shape[-1]
        cls_weights = self.wt_fn(out_dim, gt_pix)
        criterion = nn.CrossEntropyLoss(weight=cls_weights)
        return criterion(pred_pix.reshape((-1, out_dim)), gt_pix.reshape(
            (-1, )))

    @classmethod
    def _bce(cls, pred_pix, gt_pix):
        """
        Binary cross entropy
        Args:
            pred_pix ([Bx7xH1xW1])
            gt_pix ([BxH2xW2])
        """
        gt_pix = nn.functional.one_hot(gt_pix,
                                       num_classes=phyre.NUM_COLORS).permute(
                                           0, 3, 1, 2).float()
        criterion = nn.BCELoss()
        return criterion(pred_pix, gt_pix)

    @classmethod
    def _convert_to_distribution(cls, tensor, keep_dims=None, act=None):
        # Keep defines which channels need to predicted/evaluated
        tensor = torch.flatten(torch.flatten(tensor, 0, 1), -2, -1)
        if keep_dims is None:
            keep_dims = (torch.sum(tensor, dim=-1) > 0)
        return act(tensor[keep_dims, ...]), keep_dims

    @classmethod
    def _per_channel_spatial_softmax(
            cls,
            pred_pix,
            gt_pix,
            gt_dist=True,
            # Not exposed, used for a quick expt
            target_temp=1):
        """
        Args:
            pred_pix ([Bx7xH1xW1])
            gt_pix ([BxH2xW2])
        """
        gt_pix = nn.functional.one_hot(gt_pix,
                                       num_classes=phyre.NUM_COLORS).permute(
                                           0, 3, 1, 2).float()
        # flatten, move the channels to batch since each channel will be tested
        # separately, and stretch out the spatial dimensions
        act = nn.Softmax(dim=-1) if gt_dist else lambda x: x  # (identity)
        gt_pix = gt_pix / target_temp
        gt_pix_normed, keep_dims = cls._convert_to_distribution(
            gt_pix, None, act)
        act = nn.LogSoftmax(dim=-1)
        pred_pix_normed, _ = cls._convert_to_distribution(
            pred_pix, keep_dims, act)
        nll = torch.mean(-torch.sum(gt_pix_normed * pred_pix_normed, dim=-1))
        return nll

    @classmethod
    def _per_channel_spatial_softmax_no_gt_dist(cls, pred_pix, gt_pix):
        return cls._per_channel_spatial_softmax(pred_pix,
                                                gt_pix,
                                                gt_dist=False)

    @classmethod
    def _per_channel_spatial_kl(cls, pred_pix, gt_pix):
        """
        Args:
            pred_pix ([Bx7xH1xW1])
            gt_pix ([BxH2xW2])
        """
        gt_pix = nn.functional.one_hot(gt_pix,
                                       num_classes=phyre.NUM_COLORS).permute(
                                           0, 3, 1, 2).float()
        # flatten, move the channels to batch since each channel will be tested
        # separately, and stretch out the spatial dimensions
        gt_pix_normed, keep_dims = cls._convert_to_distribution(
            gt_pix, None, nn.Softmax(dim=-1))
        pred_pix_normed, _ = cls._convert_to_distribution(
            pred_pix, keep_dims, nn.LogSoftmax(dim=-1))
        return nn.KLDivLoss(reduction='batchmean')(pred_pix_normed,
                                                   gt_pix_normed)

    def forward(self, pred_pix, gt_pix):
        assert len(pred_pix.shape) == 4
        assert len(gt_pix.shape) == 3
        gt_pix = reshape_gt_to_pred(pred_pix, gt_pix)
        return self.loss_fn(pred_pix, gt_pix)


class PixL2Loss(nn.Module):
    """Pixel level L2 loss."""
    def forward(self, pred_pix, gt_pix):
        """
        Args:
            pred_pix ([Bx7xH1xW1])
            gt_pix ([BxH2xW2])
        """
        gt_pix = reshape_gt_to_pred(pred_pix, gt_pix)
        # Add the channel dimension to gt_pix
        gt_pix = nn.functional.one_hot(gt_pix,
                                       num_classes=phyre.NUM_COLORS).permute(
                                           0, 3, 1, 2).float()
        criterion = nn.MSELoss()
        return criterion(pred_pix, gt_pix)


class InfoNCELoss(nn.Module):
    """Impl of InfoNCE loss."""
    def __init__(self, in_dim, temp=0.07, l2_norm_feats=True, nce_dim=None):
        """
        Args:
            in_dim: Dimension of the incoming features
            temp (float): temprature
            l2_norm_feats (bool): Whether to normalize feats before
            nce_dim (int): If not None, reduce the dimensionality to this number
        """
        super().__init__()
        self.temp = temp
        self.l2_norm_feats = l2_norm_feats
        if nce_dim is not None and in_dim != nce_dim:
            self.reduce_dim = nn.Sequential(nn.Linear(in_dim, nce_dim),
                                            nn.ReLU(),
                                            nn.Linear(nce_dim, nce_dim))
        else:
            self.reduce_dim = lambda x: x  # Identity

    def forward(self, pred, gt):
        """
        Args:
            pred (BxNobjxDxH'xW')
            gt (BxNobjxDxH'xW')
            From https://arxiv.org/pdf/1911.05722.pdf
        """
        def spatial_to_batch(feat):
            """Move the H', W', Nobj dimension to batch,
            so will do a spatial-obj NCE."""
            feat_dim = feat.shape[2]
            return torch.reshape(torch.transpose(feat, 2, -1), [-1, feat_dim])

        pred_rolled = spatial_to_batch(pred)
        gt_rolled = spatial_to_batch(gt)
        pred_rolled = self.reduce_dim(pred_rolled)
        gt_rolled = self.reduce_dim(gt_rolled)
        # Features are L2 normalized before doing this typically. In case the
        # feature extractor did not normalize the features, normalize it now
        if self.l2_norm_feats:
            pred_rolled = nn.functional.normalize(pred_rolled, p=2, dim=-1)
            gt_rolled = nn.functional.normalize(gt_rolled, p=2, dim=-1)
        logits = torch.mm(pred_rolled, torch.transpose(gt_rolled, 0, 1))
        labels = torch.arange(pred_rolled.shape[0]).to(logits.device)
        criterion = nn.CrossEntropyLoss()
        return criterion(logits / self.temp, labels)
