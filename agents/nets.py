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

import logging
from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torchvision
from omegaconf import OmegaConf
import hydra

import phyre
from phyre_simulator import PhyreSimulator  # pylint: disable=unused-import
from losses import *  # pylint: disable=wildcard-import,unused-wildcard-import
from preproc import *  # pylint: disable=wildcard-import,unused-wildcard-import

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda:0' if USE_CUDA else 'cpu')
np.random.seed(42)

class ActionNetwork(nn.Module):
    def __init__(self, action_size, output_size, hidden_size=256,
                 num_layers=1):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(action_size, hidden_size)])
        for _ in range(1, num_layers):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, tensor):
        for layer in self.layers:
            tensor = nn.functional.relu(layer(tensor), inplace=True)
        return self.output(tensor)


class FilmActionNetwork(nn.Module):
    def __init__(self, action_size, output_size, **kwargs):
        super().__init__()
        self.net = ActionNetwork(action_size, output_size * 2, **kwargs)

    def forward(self, actions, image):
        beta, gamma = torch.chunk(
            self.net(actions).unsqueeze(-1).unsqueeze(-1), chunks=2, dim=1)
        return image * beta + gamma


class SimpleNetWithAction(nn.Module):
    def __init__(self, action_size, action_network_kwargs=None):
        super().__init__()
        action_network_kwargs = action_network_kwargs or {}
        self.stem = nn.Sequential(
            nn.Conv2d(phyre.NUM_COLORS, 3, kernel_size=1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
            nn.Conv2d(3, 64, kernel_size=7, stride=4, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=5, stride=2, padding=2,
                      bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=5, stride=2, padding=2,
                      bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.action_net = ActionNetwork(action_size, 128,
                                        **action_network_kwargs)

    @property
    def device(self):
        if hasattr(self, 'parameters') and next(self.parameters()).is_cuda:
            return 'cuda'
        else:
            return 'cpu'

    def preprocess(self, observations):
        device = self.device
        image = _image_colors_to_onehot(
            observations.to(dtype=torch.long, device=device))
        return dict(features=self.stem(image).squeeze(-1).squeeze(-1))

    def forward(self, observations, actions, preprocessed=None):
        if preprocessed is None:
            preprocessed = self.preprocess(observations)
        return self._forward(actions, **preprocessed)

    def _forward(self, actions, features):
        actions = self.action_net(actions.to(features.device))
        return (actions * features).sum(-1) / (actions.shape[-1]**0.5)

    def ce_loss(self, decisions, targets):
        targets = torch.ByteTensor(targets).float().to(decisions.device)
        return nn.functional.binary_cross_entropy_with_logits(
            decisions, targets)


def _get_fusution_points(fusion_place_spec, max_points):
    if fusion_place_spec == 'all':
        return tuple(range(max_points))
    elif fusion_place_spec == 'none':
        return tuple()
    else:
        return tuple(int(fusion_place_spec), )


class ResNet18FilmAction(nn.Module):
    def __init__(self,
                 action_size,
                 action_layers=1,
                 action_hidden_size=256,
                 fusion_place='last'):
        super().__init__()
        net = torchvision.models.resnet18(pretrained=False)
        conv1 = nn.Conv2d(phyre.NUM_COLORS,
                          64,
                          kernel_size=7,
                          stride=2,
                          padding=3,
                          bias=False)
        self.register_buffer('embed_weights', torch.eye(phyre.NUM_COLORS))
        self.stem = nn.Sequential(conv1, net.bn1, net.relu, net.maxpool)
        self.stages = nn.ModuleList(
            [net.layer1, net.layer2, net.layer3, net.layer4])

        def build_film(output_size):
            return FilmActionNetwork(action_size,
                                     output_size,
                                     hidden_size=action_hidden_size,
                                     num_layers=action_layers)

        assert fusion_place in ('first', 'last', 'all', 'none', 'last_single')

        self.last_network = None
        if fusion_place == 'all':
            self.action_networks = nn.ModuleList(
                [build_film(size) for size in (64, 64, 128, 256)])
        elif fusion_place == 'last':
            # Save module as attribute.
            self._action_network = build_film(256)
            self.action_networks = [None, None, None, self._action_network]
        elif fusion_place == 'first':
            # Save module as attribute.
            self._action_network = build_film(64)
            self.action_networks = [self._action_network, None, None, None]
        elif fusion_place == 'last_single':
            # Save module as attribute.
            self.last_network = build_film(512)
            self.action_networks = [None, None, None, None]
        elif fusion_place == 'none':
            self.action_networks = [None, None, None, None]
        else:
            raise Exception('Unknown fusion place: %s' % fusion_place)
        self.reason = nn.Linear(512, 1)

    @property
    def device(self):
        if hasattr(self, 'parameters') and next(self.parameters()).is_cuda:
            return 'cuda'
        else:
            return 'cpu'

    def preprocess(self, observations):
        image = self._image_colors_to_onehot(observations)
        features = self.stem(image)
        for stage, act_layer in zip(self.stages, self.action_networks):
            if act_layer is not None:
                break
            features = stage(features)
        else:
            features = nn.functional.adaptive_max_pool2d(features, 1)
        return dict(features=features)

    def forward(self, observations, actions, preprocessed=None):
        if preprocessed is None:
            preprocessed = self.preprocess(observations)
        return self._forward(actions, **preprocessed)

    def _forward(self, actions, features):
        actions = actions.to(features.device)
        skip_compute = True
        for stage, film_layer in zip(self.stages, self.action_networks):
            if film_layer is not None:
                skip_compute = False
                features = film_layer(actions, features)
            if skip_compute:
                continue
            features = stage(features)
        if not skip_compute:
            features = nn.functional.adaptive_max_pool2d(features, 1)
        if self.last_network is not None:
            features = self.last_network(actions, features)
        features = features.flatten(1)
        if features.shape[0] == 1 and actions.shape[0] != 1:
            # Haven't had a chance to use actions. So will match batch size as
            # in actions manually.
            features = features.expand(actions.shape[0], -1)
        return self.reason(features).squeeze(-1)

    def ce_loss(self, decisions, targets):
        targets = targets.to(dtype=torch.float, device=decisions.device)
        return nn.functional.binary_cross_entropy_with_logits(
            decisions, targets)

    def _image_colors_to_onehot(self, indices):
        onehot = torch.nn.functional.embedding(
            indices.to(dtype=torch.long, device=self.embed_weights.device),
            self.embed_weights)
        onehot = onehot.permute(0, 3, 1, 2).contiguous()
        return onehot


def _image_colors_to_onehot(indices):
    onehot = torch.nn.functional.embedding(
        indices, torch.eye(phyre.NUM_COLORS, device=indices.device))
    onehot = onehot.permute(0, 3, 1, 2).contiguous()
    return onehot


def gen_dyn_conv(dim_in, dim_out):
    # Switched to 1x1 kernels since I might be running it on 1x1 features too.
    # Using vector features when using object representation
    conv = nn.Conv2d(dim_in,
                     dim_out,
                     kernel_size=1,
                     stride=1,
                     padding=0,
                     bias=False)
    return conv


class DynConcat(nn.Module):
    """Simple dynamics model, that concats the features and 2 layer MLP."""
    def __init__(self, encoder, dim, n, nobj):
        super().__init__()
        del encoder  # This one doesn't need it
        self.dyn = nn.Sequential(gen_dyn_conv(dim * n * nobj, dim * nobj),
                                 nn.ReLU(inplace=True),
                                 gen_dyn_conv(dim * nobj, dim * nobj),
                                 nn.ReLU(inplace=True),
                                 gen_dyn_conv(dim * nobj, dim * nobj))

    def forward(self, features, pixels):
        """
        This dyn model does not use pixels, so will just return the last history
            frame
        Args:
            features: (B, T, Nobj, D, H', W')
            pixels: (B, T, Nobj, C, H, W)
        Returns:
            pred: (B, Nobj, D, H', W')
            pixels: (B, Nobj, C, H, W)
            addl_losses: {}
        """
        cat_feats = torch.reshape(features, (features.shape[0], -1) +
                                  features.shape[-2:])
        future_feat = torch.reshape(self.dyn(cat_feats),
                                    features.shape[:1] + features.shape[2:])
        # Skip connection, add the last frames features, so it stops
        # deleting things
        pred = features[:, -1, ...] + future_feat
        return pred, pixels[:, -1, ...], {}


class MultiSTN(nn.Module):
    """Multi spatial transformer network: predicts multiple transformations
    and applies to parts of the input feature, split on the channel dim."""
    def __init__(self,
                 input_dim,
                 num_tx,
                 dof='affine',
                 inp_type='pix',
                 affine_tx_mode='bilinear',
                 kernel_size=3,
                 stochastic=False):
        """
        Args:
            input_dim (int): Dimension of the features used to predict the STN
                parameters
            num_tx (int): Number of transformations to predict, will apply to
                the tensor, split along some dimension
            dof (str): Controls how generic of a affine matrix to predict.
                If 'affine', will predict a generic 3x2 matrix
                If 'rot-trans-only', it will only predict theta, x, y,
                and use those to construct the affine matrix. So it will force
                the matrix to not do any shear, scale etc.
                Similarly for 'rot-only' and 'trans-only'
            inp_type (str): Defines the type of the input. 'pix' is the default,
                to directly transform the grid and move the pixels. 'pt' is the
                PointNet style format, where the first 2 dimensions of each
                split of the channels must correspond to the X, Y location, and
                the transforms will just modify those dimensions, and not
                touch the pixel values at all.
            affine_tx_mode (str): The mode to use for grid_sample
            kernel_size (int)
            stochastic (bool): If true, predict a distribution over the affine
                matrix, instead of deterministically.
        """
        super().__init__()
        self.num_tx = num_tx
        self.dof = dof
        self.inp_type = inp_type
        self.affine_tx_mode = affine_tx_mode
        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(input_dim,
                      8 * num_tx,
                      kernel_size=kernel_size,
                      padding=kernel_size // 2), nn.ReLU(True),
            nn.Conv2d(8 * num_tx,
                      10 * num_tx,
                      kernel_size=kernel_size,
                      padding=kernel_size // 2), nn.ReLU(True))

        # Regressor for the affine matrices
        # Predicting 3x2 parameters that should be enough for any generic
        # affine transformation, though will subselect in case only few
        # parameters are needed
        self.stochastic = stochastic
        if self.stochastic:
            self.fc_loc_mean = nn.Linear(10 * num_tx, 10 * num_tx)
            self.fc_loc_logvar = nn.Linear(10 * num_tx, 10 * num_tx)
        self.fc_loc = nn.Sequential(nn.Linear(10 * num_tx, 32 * num_tx),
                                    nn.ReLU(True),
                                    nn.Linear(32 * num_tx, num_tx * 3 * 2))

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        if self.dof != 'affine':  # The paramters would be used for rot/trans
            self.fc_loc[2].bias.data.zero_()  # 0 rot/translation by default
        else:
            self.fc_loc[2].bias.data.copy_(
                torch.from_numpy(
                    np.array([1, 0, 0, 0, 1, 0] * num_tx, dtype=np.float)))

    def transform_pix(self, feat, theta, mode='bilinear'):
        """Transform the features using theta."""
        grid = nn.functional.affine_grid(theta,
                                         feat.size(),
                                         align_corners=True)
        return nn.functional.grid_sample(feat,
                                         grid,
                                         mode=mode,
                                         align_corners=True)

    def transform_pt(self, feat, theta):
        """Transform pt-net style feature using theta.
        Here, it assumes the first 2 dimensions of the feature are loc.
        Args:
            feat (B, C, H, W), C >= 2
        Returns:
            tx feat (B, C, H, W)
        """
        assert feat.shape[1] >= 2
        feat_pos = feat[:, :2, ...]
        feat_pos_ones = torch.ones_like(feat[:, :1, ...])
        feat_pos_aug = torch.cat([feat_pos, feat_pos_ones], dim=1)
        feat_pos_aug = feat_pos_aug.view(feat.shape[:1] + (3, -1))
        feat_pos_aug_end = feat_pos_aug.transpose(1, 2).unsqueeze(-1)
        txed = torch.matmul(theta.unsqueeze(1), feat_pos_aug_end)
        tx_feat_pos = txed.squeeze(-1).transpose(1, 2).view(feat_pos.shape)
        # Attach the features to it
        tx_feat = torch.cat([tx_feat_pos, feat[:, 2:, ...]], dim=1)
        return tx_feat

    def _compute_loc_stochastic(self, feat_hist):
        # from https://github.com/pytorch/examples/blob/master/vae/main.py#L53
        mean = self.fc_loc_mean(feat_hist)
        logvar = self.fc_loc_logvar(feat_hist)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        latent_var_z = mean + eps * std
        kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        return self.fc_loc(latent_var_z), kl_loss

    def forward(self, feat_for_tx, feat_to_tx, split_dim=1):
        """
        Args:
            feat_for_tx (B, D, H, W): The features to use to compute the
                transformation
            feat_to_tx (B, D', H, W): Features to apply the tx onto
            split_dim (int): Dimension to split on
        """
        feat_hist_embed = self.localization(feat_for_tx)
        # Average out the spatial dimension
        feat_hist_embed = torch.mean(feat_hist_embed, dim=[-2, -1])
        addl_losses = {}
        if self.stochastic:
            pred, kl_loss = self._compute_loc_stochastic(feat_hist_embed)
            addl_losses['kl'] = kl_loss
        else:
            pred = self.fc_loc(feat_hist_embed)
        if self.dof != 'affine':
            pred = pred.view(-1, self.num_tx, 3 * 2)
            # Say the first number is actual angle, and next 2 are x, y
            angle = pred[..., :1]
            pos_x = pred[..., 1:2]
            pos_y = pred[..., 2:3]
            if self.dof == 'rot-only':
                pos_x = torch.zeros_like(pos_x)
                pos_y = torch.zeros_like(pos_y)
            elif self.dof == 'trans-only':
                angle = torch.zeros_like(angle)
            else:
                assert self.dof == 'rot-trans-only', 'The only other option'
            cos_angle = torch.cos(angle)
            sin_angle = torch.sin(angle)
            # create the 2x3 matrix out of this
            theta = torch.cat(
                [cos_angle, sin_angle, pos_x, -sin_angle, cos_angle, pos_y],
                dim=-1)
            theta = theta.view(theta.shape[:-1] + (2, 3))
        elif self.dof == 'affine':
            theta = pred.view(-1, self.num_tx, 2, 3)
        else:
            raise NotImplementedError('Unknown {}'.format(self.dof))

        # Split the channels of feat_to_tx into num_tx groups, and apply the
        # transformations to each of those groups
        assert feat_to_tx.shape[split_dim] % self.num_tx == 0, (
            'Must be divisible to ensure equal sized chunks')
        # Chunk it
        feat_to_tx_parts = torch.chunk(feat_to_tx, self.num_tx, split_dim)

        # Apply the corresponding transformation to each part
        if self.inp_type == 'pix':
            tx_fn = partial(self.transform_pix, mode=self.affine_tx_mode)
        elif self.inp_type == 'pt':
            tx_fn = self.transform_pt
        else:
            raise NotImplementedError('Unknown type {}'.format(self.inp_type))

        feat_to_tx_parts_txed = [
            tx_fn(el, theta[:, i, ...])
            for i, el in enumerate(feat_to_tx_parts)
        ]
        return torch.cat(feat_to_tx_parts_txed, dim=split_dim), addl_losses


class DynSTN(nn.Module):
    """Spatial Transformer based dynamics model."""
    def __init__(self, encoder, dim, n, nobj, num_tx, base_stn):
        super().__init__()
        del encoder  # This one doesn't need it
        assert nobj == 1 or nobj == num_tx, (
            'Either split the 1 object features and tx, or tx each obj sep')
        self.dyn = hydra.utils.instantiate(base_stn, dim * n * nobj, num_tx)

    def forward(self, features, pixels):
        """
        This dyn model does not use pixels, so will just return the last history
            frame
        Args:
            features: (B, T, Nobj, D, H', W')
            pixels: (B, T, Nobj, C, H, W)
        Returns:
            pred: (B, Nobj, D, H', W')
            pix
            addl_losses
        """
        cat_feats = torch.reshape(features, (features.shape[0], -1) +
                                  features.shape[-2:])
        # For > 1 objs, just flatten Nobj and D channels, and the STN class
        # will split it back to do the transformations
        feat_obj_flat = torch.flatten(features, 2, 3)
        new_feat, addl_loses = self.dyn(cat_feats, feat_obj_flat[:, -1, ...])
        future_feat = torch.reshape(new_feat,
                                    features.shape[:1] + features.shape[2:])
        return future_feat, pixels[:, -1, ...], addl_loses


class DynSTNPixels_DEPRECATED(nn.Module):
    """Spatial Transformer based dynamics model, applied on pixels.
    Use DynSTNPixelChannelsDetBg"""
    def __init__(self, encoder, dim, n, nobj, num_tx, base_stn):
        super().__init__()
        self.enc = encoder
        self.dyn = hydra.utils.instantiate(base_stn, dim * n * nobj, num_tx)
        self.num_tx = num_tx
        # A network to predict num_tx attention maps
        self.attention = nn.Sequential(
            gen_deconv(dim * n * nobj, num_tx),
            *([gen_deconv(num_tx, num_tx, upsample_factor=4)] * 2),
            nn.Conv2d(num_tx, num_tx, kernel_size=1, padding=0, bias=False),
            nn.Softmax(dim=1))

    def forward(self, features, pixels):
        """
        Args:
            features: (B, T, Nobj, D, H', W')
            pixels: (B, T, C, H, W)
        Returns:
            pred: (B, Nobj, D, H', W')
        """
        raise NotImplementedError('Deal with objectified pixel input. '
                                  'Also deal with addl losses. ')
        cat_feats = torch.reshape(features, (features.shape[0], -1) +
                                  features.shape[-2:])
        assert features.shape[2] == 1, 'Not implemented yet for >1 objs'
        # Repmat the image channels num_tx times, so STN can predict those many
        # transformations
        pixels_tiled = pixels.repeat(1, 1, self.num_tx, 1, 1)
        future_pixels_tiled = self.dyn(cat_feats, pixels_tiled[:, -1, ...])
        # Compute attention maps for compositing
        attention_maps = self.attention(cat_feats)
        # Do a weighted sum of the channels using the attention maps
        attention_maps_split = torch.chunk(attention_maps, self.num_tx, 1)
        future_pixels_split = torch.chunk(future_pixels_tiled, self.num_tx, 1)
        weighted = [
            att * pix
            for att, pix in zip(attention_maps_split, future_pixels_split)
        ]
        future_pixels = torch.mean(torch.stack(weighted), dim=0)
        # Since this is a new image being generated, need to pass through the
        # encoder to get the features for this image
        future_feat = self.enc(future_pixels.unsqueeze(1))[:, 0, ...]
        return future_feat, future_pixels


class DynSTNPixelChannels_DEPRECATED(nn.Module):
    """Spatial Transformer based dynamics model, applied on channels of img.
    Use DynSTNPixelChannelsDetBg"""
    def __init__(self, encoder, dim, n, nobj, base_stn):
        super().__init__()
        self.enc = encoder
        self.num_tx = phyre.NUM_COLORS  # One tx per color
        self.dyn = hydra.utils.instantiate(base_stn, dim * n * nobj,
                                           self.num_tx)

    def forward(self, features, pixels):
        """
        Args:
            features: (B, T, Nobj, D, H', W')
            pixels: (B, T, C, H, W)
        Returns:
            pred: (B, Nobj, D, H', W')
        """
        raise NotImplementedError('Deal with objectified pixel input. '
                                  'Also deal with addl losses. ')
        assert (pixels.shape[2] == self.num_tx or
                pixels.shape[2] == self.num_tx * 3), 'In pix or pt mode so far'
        cat_feats = torch.reshape(features, (features.shape[0], -1) +
                                  features.shape[-2:])
        assert features.shape[2] == 1, 'Not implemented yet for >1 objs'
        future_pixels = self.dyn(cat_feats, pixels[:, -1, ...])
        # Since this is a new image being generated, need to pass through the
        # encoder to get the features for this image
        future_feat = self.enc(future_pixels.unsqueeze(1))[:, 0, ...]
        return future_feat, future_pixels


class DynSTNPixelChannelsGenBg_DEPRECATED(nn.Module):
    """Spatial Transformer based dynamics model, applied on channels of img.
    Generates the background.
    Use DynSTNPixelChannelsDetBg
    """
    def __init__(self, encoder, dim, n, nobj, base_stn):
        super().__init__()
        self.enc = encoder
        # One tx per color, except background that is generated since it's not
        # an object that can be moved like others. Just a 1x1 convolution on
        # the predicted image to gen the last channel
        self.num_tx = phyre.NUM_COLORS - 1
        self.dyn = hydra.utils.instantiate(base_stn, dim * n * nobj,
                                           self.num_tx)
        # Just a couple layer should suffice, over the last frame, and new frame
        # feature
        self.bg_dec = nn.Sequential(
            nn.Conv2d(2 * phyre.NUM_COLORS - 1,
                      8,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=False), nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=1, stride=1, padding=0, bias=False))

    def forward(self, features, pixels):
        """
        Args:
            features: (B, T, Nobj, D, H', W')
            pixels: (B, T, C, H, W)
        Returns:
            pred: (B, Nobj, D, H', W')
        """
        raise NotImplementedError('Deal with objectified pixel input. '
                                  'Also deal with addl losses. ')
        assert (pixels.shape[2] - 1) == self.num_tx
        cat_feats = torch.reshape(features, (features.shape[0], -1) +
                                  features.shape[-2:])
        assert features.shape[2] == 1, 'Not implemented yet for >1 objs'
        future_pixels_obj = self.dyn(cat_feats, pixels[:, -1, 1:, ...])
        future_pixels_bg = self.bg_dec(
            torch.cat([pixels[:, -1, ...], future_pixels_obj], dim=1))
        future_pixels = torch.cat([future_pixels_bg, future_pixels_obj], dim=1)
        # Since this is a new image being generated, need to pass through the
        # encoder to get the features for this image
        future_feat = self.enc(future_pixels.unsqueeze(1))[:, 0, ...]
        return future_feat, future_pixels


class DynSTNPixelChannelsDetBg(nn.Module):
    """Spatial Transformer based dynamics model, applied on channels of img.
    Generates the background deterministically, using the change.
    """
    def __init__(self,
                 encoder,
                 dim,
                 n,
                 nobj,
                 base_stn,
                 movable_ch,
                 movable_only=False):
        super().__init__()
        self.enc = encoder
        self.movable_only = movable_only
        # One tx per color (or movable colors, if that is set),
        # except background that is generated since it's not
        # an object that can be moved like others.
        if self.movable_only:
            self.movable_channels = torch.LongTensor(movable_ch)
        else:
            self.movable_channels = torch.arange(1, phyre.NUM_COLORS)
        self.num_tx = len(self.movable_channels)
        self.nobj = nobj
        self.dyn = hydra.utils.instantiate(base_stn, dim * n * nobj,
                                           self.num_tx * nobj)

    def forward(self, features, pixels):
        """
        Args:
            features: (B, T, Nobj, D, H', W')
            pixels: (B, T, Nobj, C, H, W)
        Returns:
            pred: (B, Nobj, D, H', W')
            pix
            addl_losses
        """
        assert pixels.shape[3] >= self.num_tx
        cat_feats = torch.reshape(features, (features.shape[0], -1) +
                                  features.shape[-2:])
        pixels_movable = pixels[:, -1, :, self.movable_channels, ...]
        # combine all channels of objects and transform
        pixels_movable_flat = torch.flatten(pixels_movable, 1, 2)
        future_pixels_flat_movable, addl_losses = self.dyn(
            cat_feats, pixels_movable_flat)
        future_pixels_movable = future_pixels_flat_movable.view(
            pixels_movable.shape)
        future_pixels = pixels[:, -1, ...]  # Copy most of the channels
        future_pixels[:, :, self.movable_channels, ...] = future_pixels_movable
        # Compute the background deterministically, where all other channels
        # are 0s, it has to be 1. So make channels sum to 1.
        future_pixels_bg = 1.0 - torch.sum(
            future_pixels[:, :, 1:, ...], dim=2, keepdims=True)
        future_pixels[:, :, :1, ...] = future_pixels_bg
        # Since this is a new image being generated, need to pass through the
        # encoder to get the features for this image
        future_feat = self.enc(future_pixels.unsqueeze(1))[:, 0, ...]
        return future_feat, future_pixels, addl_losses


def gen_deconv(in_dim,
               out_dim,
               stride=1,
               kernel_size=3,
               padding=1,
               upsample_factor=2,
               inst_norm=False,
               activation=nn.ReLU(inplace=True)):
    return nn.Sequential(
        nn.ConvTranspose2d(in_dim,
                           out_dim,
                           kernel_size=kernel_size,
                           stride=stride,
                           padding=padding,
                           bias=False),
        # nn.Sequential() simulates identity, if no instance norm to be added
        nn.InstanceNorm2d(out_dim, affine=True)
        if inst_norm else nn.Sequential(),
        activation,
        nn.Upsample(scale_factor=upsample_factor,
                    mode='bilinear',
                    align_corners=True))


class BasicDecoder(nn.Module):
    """Simple decoder, goes from features to frame representation."""
    def __init__(self, in_dim, out_dim, nlayers, kernel_size, padding,
                 upsample_factor, decode_from, backprop_feat_ext, inst_norm,
                 activation):
        super().__init__()
        decoder_dim = 256
        self.backprop_feat_ext = backprop_feat_ext
        self.decode_from = decode_from
        assert self.decode_from in ['pixels', 'features']
        if self.decode_from == 'pixels':
            in_dim = phyre.NUM_COLORS
            decoder_dim = 16
        activation = hydra.utils.instantiate(activation)
        logging.warning('Using %s activation for decoders', activation)
        inter_layers = [
            gen_deconv(decoder_dim,
                       decoder_dim,
                       1,
                       kernel_size,
                       padding,
                       upsample_factor,
                       inst_norm,
                       activation=activation) for _ in range(nlayers)
        ]
        self.deconv_net = nn.Sequential(
            gen_deconv(in_dim,
                       decoder_dim,
                       1,
                       kernel_size,
                       padding,
                       upsample_factor,
                       activation=activation), *inter_layers,
            gen_deconv(
                decoder_dim,
                out_dim,
                1,
                kernel_size,
                padding,
                upsample_factor,
                activation=nn.Sequential()))  # No activation on the last

    def forward(self, features, pixels):
        """
        Args:
            features (BxNobjxDxH'xW'): Features to be decoded
            pixels (BxNobjxCxHxW): Pixels generated by the dynamics model
        Returns:
            imgs (BxNobjxD_outxHxW): Output frames (per obj, aggregation is
                done later in the Fwd class)
        """
        if self.decode_from == 'pixels':
            decode_feature = pixels
        else:
            decode_feature = features
        if not self.backprop_feat_ext:
            # Means train the decoder separately from the rest of the network,
            # don't backprop gradients to the feature extractor
            decode_feature = decode_feature.detach()
        # Summing the features over all the objects, and doing one decode.
        # Separate decodes takes just way too much time, so need to do it once
        decode_feature = torch.sum(decode_feature, dim=1, keepdims=True)
        features_flatten_obj = torch.flatten(decode_feature, 0, 1)
        images = self.deconv_net(features_flatten_obj)
        # Reshape back into object level
        out = torch.reshape(images,
                            decode_feature.shape[:2] + images.shape[1:])
        return out


class TrivialDecoder(nn.Module):
    """Trivial decoder, simply outputs the frames from the dynamics model."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        del in_dim, out_dim

    def forward(self, features, pixels):
        """
        Args:
            features (BxNobjxDxH'xW'): Features to be decoded
            pixels (BxNobjxCxHxW): Pixels generated by the dynamics model
        Returns:
            imgs (BxNobjxCxHxW): Output frames
        """
        del features  # assumes the dynamics model will do all decoding
        return pixels


def average_losses(all_losses):
    """Average the losses into one dict of losses.
    Args:
        all_losses: List of dictionary of losses.
    Returns:
        combined: A dictionary with same keys as individual dicts, with
            all losses combined.
    """
    if len(all_losses) == 0:
        return {}
    combined = {}
    for key, val in all_losses[0].items():
        if not isinstance(val, torch.Tensor):
            # If it's none or sth.. eg some loss was not active
            combined[key] = val
        else:
            # Average all the values
            stkd = torch.stack([el[key] for el in all_losses])
            # Average the losses that are positive, since I set undefined
            # losses to -1 (where not enough GT is available, etc)
            combined[key] = torch.mean(stkd * (stkd >= 0), dim=0)
    return combined


class BasicObjEncoder(nn.Module):
    """Takes objectified representation, and puts it through more layers."""
    def __init__(self,
                 in_dim,
                 out_dim,
                 nlayers,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 spatial_mean=True):
        super().__init__()
        if nlayers > 0:
            self.out_dim = out_dim
        else:
            logging.warning('Ignoring the out_dim (%d) for ObjEncoder',
                            out_dim)
            self.out_dim = in_dim
        layers_lst = [[
            nn.Conv2d(in_dim if i == 0 else out_dim,
                      out_dim,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding,
                      bias=False),
            nn.ReLU(inplace=True)
        ] for i in range(nlayers)]
        layers_lst_flat = [item for sublist in layers_lst for item in sublist]
        if len(layers_lst_flat) > 0:
            layers_lst_flat = layers_lst_flat[:-1]  # Remove the last relu
            self.encoder = nn.Sequential(*layers_lst_flat)
        else:
            self.encoder = None
        self.spatial_mean = spatial_mean

    def forward(self, feat):
        """
        Args:
            feat: (B, T, Nobj, D, H', W')
        """
        if self.encoder:
            feat_flat = torch.flatten(feat, 0, 2)
            obj_embed_flat = self.encoder(feat_flat)
            obj_embed = torch.reshape(
                obj_embed_flat, feat.shape[:3] + obj_embed_flat.shape[1:])
        else:
            obj_embed = feat
        if self.spatial_mean:
            obj_embed = torch.mean(obj_embed, dim=[-1, -2], keepdims=True)
        return obj_embed


class ContextGatingObjectifier(nn.Module):
    """Takes intermediate representation and converts into object-level rep."""
    def __init__(self, dim, obj_encoder, nobj=1):
        super().__init__()
        self.obj_mapper = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0,
                      bias=False), nn.ReLU(inplace=True),
            nn.Conv2d(dim,
                      nobj,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=False))
        self.obj_encoder = hydra.utils.instantiate(obj_encoder, dim)
        self.out_dim = self.obj_encoder.out_dim

    def forward(self, vid_feat):
        """
        Decompose the video features into object level representation.
        Args:
            vid_feat: (BxTxDxH'xW')
            nobj (int): Max number of objects in the scene. The hope is that the
                extra channels will just have some degenerate information
        Returns:
            BxTxNobjxDxH''xW''
        """
        raise NotImplementedError('The inp is now objfied, TODO deal with it')
        batch_size = vid_feat.shape[0]
        # Use context gating: generate a heatmap for each object at each time
        # step, and weight using that heatmap to get an object representation
        flatten_feat = torch.flatten(vid_feat, 0, 1)
        # Unsqueeze to add a channel dimension to the attention maps
        obj_map = self.obj_mapper(flatten_feat).unsqueeze(2)
        # Add a 1-D object dimension
        flatten_feat = flatten_feat.unsqueeze(1)
        # Weight the feats with the attention maps to get the object-features
        mapped_feat = flatten_feat * obj_map
        # Reshape to add the time dimension back
        mapped_feat = torch.reshape(mapped_feat,
                                    (batch_size, -1) + mapped_feat.shape[1:])
        final_feat = self.obj_encoder(mapped_feat)
        return final_feat


class ChannelSplitObjectifier(nn.Module):
    """Splits the channel of image representation to get obj rep."""
    def __init__(self, dim, obj_encoder, nobj=1):
        super().__init__()
        self.nobj = nobj
        self.obj_encoder = hydra.utils.instantiate(obj_encoder, dim // nobj)
        self.out_dim = self.obj_encoder.out_dim

    def forward(self, vid_feat):
        """
        Decompose the video features into object level representation.
        Args:
            vid_feat: (BxTxNobjxDxH'xW')
        Returns:
            BxTxNobjx(D/Nobj)xH'xW'
        """
        assert vid_feat.shape[2] == 1, (
            'Channel split can not deal with pre objectified {} input'.format(
                vid_feat.shape[2]))
        assert vid_feat.shape[3] % self.nobj == 0, 'Must be divisible'
        # Reshape the channel dimension to split into an object dimension
        objed = vid_feat.view(vid_feat.shape[:2] + (self.nobj, -1) +
                              vid_feat.shape[-2:])
        assert objed.shape[2] == self.nobj
        assert objed.shape[3] == vid_feat.shape[3] / self.nobj
        # Apply a little network to get a flat feature
        obj_encoded = self.obj_encoder(objed)
        return obj_encoded


class TrivialObjectifier(nn.Module):
    """Simply returns the feature.

    Earlier version would unsqueeze, but since the component splitting the
        input at least has 1 obj, so no need to unsqueeze it further.
    """
    def __init__(self, dim, obj_encoder, nobj=1):
        super().__init__()
        del obj_encoder
        self.nobj = nobj
        self.out_dim = dim

    def forward(self, vid_feat):
        assert vid_feat.shape[2] == self.nobj, ('{} != {}'.format(
            vid_feat.shape[2], self.nobj))
        return vid_feat


class SimpleBaseEncoder(nn.Module):
    """Simple network, simplified from Anton's version."""
    def __init__(self, in_dim, width_scale_factor):
        """Simple encoder weights.
        For a 256x256 input, it'll give a 4x4 output."""
        super().__init__()
        self.width_scale_factor = width_scale_factor
        _s = self._scale_int
        self.stem = nn.Sequential(
            nn.Conv2d(in_dim, 3, kernel_size=1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
            nn.Conv2d(3,
                      _s(64),
                      kernel_size=7,
                      stride=2,
                      padding=3,
                      bias=False),
            nn.BatchNorm2d(_s(64)),
            nn.ReLU(inplace=True),
            nn.Conv2d(_s(64),
                      _s(64),
                      kernel_size=5,
                      stride=2,
                      padding=2,
                      bias=False),
            nn.BatchNorm2d(_s(64)),
            nn.ReLU(inplace=True),
            nn.Conv2d(_s(64),
                      _s(64),
                      kernel_size=5,
                      stride=2,
                      padding=2,
                      bias=False),
            nn.BatchNorm2d(_s(64)),
            nn.ReLU(inplace=True),
            nn.Conv2d(_s(64),
                      _s(64),
                      kernel_size=5,
                      stride=2,
                      padding=2,
                      bias=False),
            nn.BatchNorm2d(_s(64)),
            nn.ReLU(inplace=True),
            nn.Conv2d(_s(64),
                      _s(128),
                      kernel_size=5,
                      stride=2,
                      padding=2,
                      bias=False),
            nn.BatchNorm2d(_s(128)),
            nn.ReLU(inplace=True),
        )
        self.out_dim = _s(128)

    def _scale_int(self, n):
        """Scale the number by a factor. To control width of this network."""
        return int(self.width_scale_factor * n)

    def forward(self, image):
        return self.stem(image)


class ResNetBaseEncoder(nn.Module):
    """ResNet based feature extractor."""
    def __init__(self, in_dim, base_model, nlayers):
        super().__init__()
        net = hydra.utils.instantiate(base_model)
        conv1 = nn.Conv2d(in_dim,
                          64,
                          kernel_size=7,
                          stride=2,
                          padding=3,
                          bias=False)
        self.stem = nn.Sequential(conv1, net.bn1, net.relu, net.maxpool)
        self.stages = nn.ModuleList(
            [getattr(net, 'layer%d' % (i + 1)) for i in range(nlayers)])
        last_stage = self.stages[-1][-1]
        if hasattr(last_stage, 'bn3'):
            self.out_dim = last_stage.bn3.num_features
        elif hasattr(last_stage, 'bn2'):
            self.out_dim = last_stage.bn2.num_features
        else:
            raise ValueError('This should not happen')

    def forward(self, image):
        features = self.stem(image)
        for stage in self.stages:
            features = stage(features)
        return features


class BasicEncoder(nn.Module):
    """Encode pixels to features."""
    def __init__(self, in_dim, nobj, feat_ext, objectifier, obj_encoder,
                 spatial_mean, feat_ext_eval_mode, process_objs_together):
        """
        Args:
            obj_before_enc: If true, do the objectify in the input (pixel) space
                before running the encode (so each object is encoded separately)
            spatial_mean: Avg pool the features to 1x1
            feat_ext_eval_mode: Set the feature extractor to eval mode for BN,
                dropout etc
            process_objs_together: If true, it will concatenate all objs on the
                channel dimension, extract features, and split the features
                in channel dimensions to get features for each obj
        """
        super().__init__()
        self.nobj = nobj
        self.process_objs_together = process_objs_together
        # The image embedding model
        self.feat_ext = hydra.utils.instantiate(
            feat_ext, in_dim * nobj if self.process_objs_together else in_dim)
        initial_dim = self.feat_ext.out_dim
        # The objects model
        self.objectifier = hydra.utils.instantiate(objectifier, initial_dim,
                                                   obj_encoder)
        self.out_dim = self.objectifier.out_dim
        if self.process_objs_together:
            assert self.out_dim % nobj == 0
            self.out_dim //= nobj
        self.spatial_mean = spatial_mean
        self.feat_ext_eval_mode = feat_ext_eval_mode

    def _forward_vid(self, batch_vid_obs, l2_norm_feats=False):
        """
        Convert a video into images to run the forward model.
        Args:
            batch_vid_obs: BxTxCxHxW or BxTxNobjxCxHxW
        Returns:
            features: BxTxDxH'xW' or BxTxNobjxDxH'xW'
        """
        # Add an object dimension, so the rest of the code doesn't have to
        # deal with edge cases
        added_obj_dim = False
        if len(batch_vid_obs.shape) == 4:
            added_obj_dim = True
            batch_vid_obs = batch_vid_obs.unsqueeze(2)  # BxTxNobjxCxHxW
        # Flatten videos into frames to extract out the features
        # resulting shape B'xC'xHxW
        if self.process_objs_together:
            # resulting shape B' = B * T, C' = Nobj * C
            flat_obs = batch_vid_obs.reshape((-1, ) + batch_vid_obs.shape[-4:])
            flat_obs = torch.flatten(flat_obs, 1, 2)
        else:
            # resulting shape B' = B * T * Nobj, C' = C
            flat_obs = batch_vid_obs.reshape((-1, ) + batch_vid_obs.shape[-3:])
        # Extract features
        if self.feat_ext_eval_mode:
            self.feat_ext.eval()
        features = self.feat_ext(flat_obs)
        if self.spatial_mean:
            # Mean over spatial dimensions
            features = torch.mean(features, dim=[-2, -1], keepdims=True)
        if l2_norm_feats:
            # L2 normalize the features -- MemoryBank, MoCo and PIRL do that
            features = nn.functional.normalize(features, p=2, dim=-1)
        # Reshape back to original batch dimension
        if self.process_objs_together:
            features_batched = features.reshape(batch_vid_obs.shape[:2] +
                                                (self.nobj, -1) +
                                                features.shape[-2:])
        else:
            features_batched = features.reshape(batch_vid_obs.shape[:-3] +
                                                features.shape[1:])
        if added_obj_dim:
            features_batched = features_batched.squeeze(2)
        assert features_batched.shape[-3] == self.out_dim
        return features_batched

    def forward(self, vid):
        """
        Args:
            vid (B, T, Nobj, C, H, W): Input video, in preprocessed form; i.e.
                one-hot
        Returns:
            obj_feat (B, T, Nobj', D, H', W'): Features with objects, if needed
        """
        vid_feat = self._forward_vid(vid)
        vid_feat = self.objectifier(vid_feat)
        return vid_feat


def combine_obj_pixels(obj_pix, obj_dim):
    """Combine obj-split pixels into a single image.
    Args:
        obj_pix: B, ..., Nobj, ..., C, H, W
        obj_dim: The dimension to reduce over -- which corresponds to objs
    Returns
        B, ..., ..., C, H, W
    """
    if obj_pix is None:
        return None
    return torch.max(obj_pix, dim=obj_dim)[0]


class MLPClassifier(nn.Module):
    """Simple classifier on top of the intermediate features."""
    def __init__(self, in_dim, nlayers, match_inp_sz_layer=False):
        super().__init__()
        self.nlayers = nlayers
        if nlayers == 0:
            return
        # First linear layer, to project to the in_dim dimension, if not
        self.match_inp_sz_layer = match_inp_sz_layer
        if self.match_inp_sz_layer:
            raise NotImplementedError('Doesnt work with multi-gpu yet..')
            self.register_parameter('init_linear_wt', None)
        self.in_dim = in_dim
        layers = [[nn.Linear(in_dim, in_dim),
                   nn.ReLU(inplace=True)] for _ in range(nlayers - 1)]
        layers_flat = [item for sublist in layers for item in sublist]
        self.cls = nn.Sequential(*(layers_flat[:-1] + [nn.Linear(in_dim, 1)]))

    def reset_parameters(self, inp, in_dim, out_dim):
        self.init_linear_wt = nn.Parameter(
            inp.new(in_dim, out_dim).normal_(0, 1))

    def forward(self, preds, pixs, process_all_frames=False):
        """
        Run the classifier on the predictions.
        Args:
            preds: (BxTx1xDxH'xW')
            pixs: (BxTx1xDxHxW)
        Retuns:
            solved: (BxT)
            process_all_frames: Set true when used by other classifiers for
                intermediate feature extraction, so to get features for each
                frame.
        """
        del pixs  # This does not use it
        if self.nlayers == 0:
            return preds
        # Since this classifier doesn't take into account context and the final
        # _cls is going to look at the last frame, so might as well only process
        # that last frame
        if not process_all_frames:
            preds = preds[:, -1:, ...]
        mean_feat = torch.mean(preds, axis=[2, -1, -2])
        if self.match_inp_sz_layer:
            if self.init_linear_wt is None:
                logging.warning(
                    'Creating a linear layer to map the input '
                    'dims (%d) to MLP input dim (%d)', mean_feat.shape[-1],
                    self.in_dim)
                self.reset_parameters(preds, self.in_dim,
                                      preds.shape[1] * preds.shape[3])
            mean_feat = nn.functional.linear(mean_feat, self.init_linear_wt)
            mean_feat = nn.ReLU(inplace=True)(mean_feat)
        return self.cls(mean_feat).squeeze(-1)


class ConvNetClassifier(nn.Module):
    """ConvNet classifier on top of the intermediate features."""
    def __init__(self, feat_in_dim, num_conv_blocks, num_fc_layers):
        super().__init__()
        del feat_in_dim
        nobj = 1
        self.enc = BasicEncoder(
            phyre.NUM_COLORS,
            nobj,
            OmegaConf.create({
                'class': 'nets.ResNetBaseEncoder',
                'params': {
                    'base_model': {
                        'class': 'torchvision.models.resnet18',
                        'params': {
                            'pretrained': False,
                        }
                    },
                    'nlayers': num_conv_blocks,
                }
            }),
            OmegaConf.create({
                'class': 'nets.TrivialObjectifier',
                'params': {
                    'nobj': nobj,  # will sum into 1 obj
                }
            }),
            OmegaConf.create({
                'class': 'nets.BasicObjEncoder',
                'params': {
                    'out_dim': 16,
                    'nlayers': 0,
                    'spatial_mean': True,
                }
            }),
            spatial_mean=False,
            feat_ext_eval_mode=False,
            process_objs_together=False,  # Doesn't matter, 1 obj
        )
        self.cls = MLPClassifier(self.enc.out_dim, num_fc_layers)

    def forward(self, preds, pixs, process_all_frames=False):
        """
        Run the classifier on the predictions.
        Args:
            preds: (BxTx1xDxH'xW')
            pixs: (BxTx1xDxHxW)
            process_all_frames: Set true when used by other classifiers for
                intermediate feature extraction, so to get features for each
                frame.
        Retuns:
            solved: (BxT)
        """
        # Not enforcing the assert here if pred is None, since this module
        # is usually used by other modules as a way to extract features,
        # and it might pass in None for preds. But rest assured, this check
        # would have been done on the caller side.
        assert preds is None or preds.shape[1] == pixs.shape[1], (
            'Must pass in run_decode=True if using a pixel-based classifier!!')
        del preds  # This does not use it
        # Since this classifier doesn't take into account context and the final
        # _cls is going to look at the last frame, so might as well only process
        # that last frame
        if not process_all_frames:
            pixs = pixs[:, -1:, ...]
        obj_feats = self.enc(pixs)
        return self.cls(obj_feats, None, process_all_frames=process_all_frames)


class TxClassifier(nn.Module):
    """Transformer on top of the intermediate features over time."""
    def __init__(self, in_dim, nheads, nlayers):
        super().__init__()
        self.tx_enc = TxEncoder(in_dim, nheads, nlayers)
        self.cls = nn.Linear(self.tx_enc.out_dim, 1)

    def forward(self, preds, pixs):
        """
        Run the classifier on the predictions.
        Args:
            preds: (BxTx1xDxH'xW')
            pixs: (BxTx1xDxHxW)
        Retuns:
            solved: (Bx1)
        """
        del pixs  # This does not use it
        # Spatial mean the features
        stacked_mean_feat = torch.flatten(torch.mean(preds, axis=[-1, -2]), 1,
                                          2)
        feat_enc_time = self.cls(self.tx_enc(stacked_mean_feat))
        # Max pool over time to get the final prediction
        # Keepdims since the output format expects a time dimension and does
        # a max pool over it at the end
        cls_pred = torch.max(feat_enc_time, dim=1,
                             keepdims=True)[0].squeeze(-1)
        return cls_pred


class ConvTxClassifier(nn.Module):
    """Transformer on top of the Conv features learned over time."""
    def __init__(self, in_dim, nconvblocks, nheads, nlayers):
        super().__init__()
        self.conv_feat = ConvNetClassifier(in_dim, nconvblocks, 0)
        self.tx_cls = TxClassifier(self.conv_feat.enc.out_dim, nheads, nlayers)

    def forward(self, preds, pixs):
        """
        Run the classifier on the predictions.
        Args:
            preds: (BxTx1xDxH'xW')
            pixs: (BxTx1xDxHxW)
        Retuns:
            solved: (Bx1)
        """
        assert preds.shape[1] == pixs.shape[1], (
            'Must pass in run_decode=True if using a pixel-based classifier!!')
        del preds
        feats = self.conv_feat(None, pixs, process_all_frames=True)
        preds = self.tx_cls(feats, None)
        return preds


class Conv3dClassifier(nn.Module):
    """3D conv over features learned over time."""
    def __init__(self, in_dim, num_3d_layers):
        super().__init__()
        layers = [[
            nn.Conv3d(in_dim, in_dim, 3, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True)
        ] for _ in range(num_3d_layers - 1)]
        layers_flat = [item for sublist in layers for item in sublist]
        self.enc = nn.Sequential(*(layers_flat[:-1]))
        self.cls = nn.Linear(in_dim, 1)

    def forward(self, preds, pixs):
        """
        Run the classifier on the predictions.
        Args:
            preds: (BxTx1xDxH'xW')
            pixs: (BxTx1xDxHxW)
        Retuns:
            solved: (Bx1)
        """
        del pixs
        enc_preds = self.enc(preds.squeeze(2).transpose(1, 2))
        cls_preds = self.cls(torch.mean(enc_preds, [-1, -2, -3]))
        # It has 1 extra dim in the end from the fc layer which should be
        # removed, but since I need to add a time dimension anyway, just leave
        # this there (will end up the same)
        return cls_preds


class ConvConv3dClassifier(nn.Module):
    """Conv3D on top of the Conv features learned over time."""
    def __init__(self, in_dim, nconvblocks, n3dlayers):
        super().__init__()
        self.conv_feat = ConvNetClassifier(in_dim, nconvblocks, 0)
        self.td_cls = Conv3dClassifier(self.conv_feat.enc.out_dim, n3dlayers)

    def forward(self, preds, pixs):
        """
        Run the classifier on the predictions.
        Args:
            preds: (BxTx1xDxH'xW')
            pixs: (BxTx1xDxHxW)
        Retuns:
            solved: (Bx1)
        """
        assert preds.shape[1] == pixs.shape[1], (
            'Must pass in run_decode=True if using a pixel-based classifier!!')
        del preds
        feats = self.conv_feat(None, pixs, process_all_frames=True)
        preds = self.td_cls(feats, None)
        return preds


class ConcatClassifier(nn.Module):
    """Concat the features and classify."""
    def __init__(self, in_dim, nlayers):
        super().__init__()
        self.cls = MLPClassifier(in_dim, nlayers, match_inp_sz_layer=True)

    def forward(self, preds, pixs):
        """
        Run the classifier on the predictions.
        Args:
            preds: (BxTx1xDxH'xW')
            pixs: (BxTx1xDxHxW)
        Retuns:
            solved: (Bx1)
        """
        del pixs
        # Concatenate over the time dimension
        preds_flat = preds.view(preds.shape[0], 1, 1, -1, preds.shape[-2],
                                preds.shape[-1])
        return self.cls(preds_flat, None, process_all_frames=True)


class ConvConcatClassifier(nn.Module):
    """Concat the Conv features and classify."""
    def __init__(self, in_dim, nconvblocks, nclslayers):
        super().__init__()
        self.conv_feat = ConvNetClassifier(in_dim, nconvblocks, 0)
        self.concat_cls = ConcatClassifier(self.conv_feat.enc.out_dim,
                                           nclslayers)

    def forward(self, preds, pixs):
        """
        Run the classifier on the predictions.
        Args:
            preds: (BxTx1xDxH'xW')
            pixs: (BxTx1xDxHxW)
        Retuns:
            solved: (Bx1)
        """
        assert preds.shape[1] == pixs.shape[1], (
            'Must pass in run_decode=True if using a pixel-based classifier!!')
        del preds
        feats = self.conv_feat(None, pixs, process_all_frames=True)
        preds = self.concat_cls(feats, None)
        return preds


class TrivialInteractor(nn.Module):
    """Model interactions btw objects: do nothing."""
    def __init__(self, in_dim):
        super().__init__()
        del in_dim

    @classmethod
    def forward(cls, feat):
        """
        Args:
            feat: (B, T, Nobj, C, H', W')
        Returns:
            feat as is
        """
        return feat


class TxEncoder(nn.Module):
    """Transformer based encoder, generates a feature combining the context."""
    def __init__(self, in_dim, nheads, nlayers, maintain_dim=False):
        """
        Args:
            maintain_dim (bool): If true, it maps the final output to the same
                dimensionality as the input
        """
        super().__init__()
        # Very basic position encoding
        self.loc_embed = nn.Sequential(nn.Linear(1, 4), nn.ReLU(inplace=True),
                                       nn.Linear(4, 8))
        self.nheads = nheads
        self.nlayers = nlayers
        in_dim_loc = in_dim + 8 * nheads
        self.loc_mixer = nn.Linear(in_dim_loc, in_dim_loc)
        layer = nn.TransformerEncoderLayer(in_dim_loc, nheads)
        self.encoder = nn.TransformerEncoder(layer, nlayers)
        if maintain_dim:
            self.back_to_orig_dim = nn.Linear(in_dim_loc, in_dim)
            self.out_dim = in_dim
        else:
            self.back_to_orig_dim = lambda x: x  # Identity
            self.out_dim = in_dim_loc

    def forward(self, feat):
        """
        Args:
            feat: (B, T, C)
        Returns:
            Same shape as input
        """
        # Add a location embedding (over time), since time axis will flatten
        loc_embedding = self.loc_embed(
            torch.arange(feat.shape[1],
                         device=feat.device).unsqueeze(-1).float())
        # Make into the shape of the feature
        loc_embedding = loc_embedding.unsqueeze(0).repeat(
            feat.shape[0], 1, self.nheads)
        feat = torch.cat([feat, loc_embedding], dim=-1)
        # Mix up the location information throughout the features so each head
        # would have it
        mixed_feat = self.loc_mixer(feat)
        # Transformer encoder expects the time dimension as the 0th! So gotta
        # permute things around
        return self.back_to_orig_dim(
            self.encoder(mixed_feat.permute(1, 0, 2)).permute(1, 0, 2))


class TxInteractor(nn.Module):
    """Model interactions btw objects: using Transformer."""
    def __init__(self, in_dim, nheads, nlayers):
        super().__init__()
        self.in_dim = in_dim
        self.tx_enc = TxEncoder(in_dim, nheads, nlayers, maintain_dim=True)

    def forward(self, feat):
        """
        Args:
            feat: (B, T, Nobj, C, H', W')
        Returns:
            Same shape as input
        """
        # Mean reduce the spatial dimensions for tx, then add it back to the
        # original feature as a residual connection
        feat_spat_mean = torch.mean(feat, dim=[-1, -2])
        feat_flat = feat_spat_mean.flatten(1, 2)
        tx_feat = self.tx_enc(feat_flat)
        tx_feat = tx_feat.view(
            feat_spat_mean.shape).unsqueeze(-1).unsqueeze(-1)
        return feat + tx_feat


class TrivialSpatialAttention(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        del in_dim

    def forward(self, feat):
        return feat


class TxSpatialAttention(nn.Module):
    def __init__(self, in_dim, nheads, nlayers):
        super().__init__()
        self.tx_enc = TxEncoder(in_dim, nheads, nlayers, maintain_dim=True)

    def forward(self, feat):
        """
        Args:
            feats (B, T, Nobj, D, H', W')
        """
        feat_flat = torch.flatten(torch.flatten(feat, 0, 2), -2, -1)
        feat_att = self.tx_enc(feat_flat.transpose(1, 2)).transpose(1, 2)
        return feat_att.view(feat.shape)


class Fwd(nn.Module):
    """The master class with Forward model."""
    def __init__(self, agent_cfg):
        """
        Args:
            dyn_type: The type of dynamics model to use.
            dyn_n: Number of previous features used for prediction.
        """
        super().__init__()
        # The image embedding model
        self.preproc = VideoPreprocessor(agent_cfg)
        self.enc = hydra.utils.instantiate(agent_cfg.encoder,
                                           self.preproc.out_dim,
                                           agent_cfg.nobj)
        dim = self.enc.out_dim
        self.interactor = hydra.utils.instantiate(agent_cfg.interactor, dim)
        # The dynamics model
        self.dyn = hydra.utils.instantiate(agent_cfg.dyn, self.enc, dim)
        # Classifier model
        self.nframes_to_cls = agent_cfg.nframes_to_cls
        # A attention of the latent features before passing them through the
        # classifier.
        self.spat_att = hydra.utils.instantiate(agent_cfg.spat_att, dim)
        self.cls = hydra.utils.instantiate(agent_cfg.cls, dim)
        # Decoder model
        self.dec = hydra.utils.instantiate(agent_cfg.decoder, dim,
                                           phyre.NUM_COLORS)
        # Other loss functions
        self.pix_loss = hydra.utils.instantiate(agent_cfg.loss_fn.pix)
        self.nce_loss = hydra.utils.instantiate(agent_cfg.loss_fn.nce, dim)

    @property
    def device(self):
        if hasattr(self, 'parameters') and next(self.parameters()).is_cuda:
            return 'cuda'
        else:
            return 'cpu'

    def _forward_dyn(self, feats, vids, n_fwd_times, need_intermediate=False):
        """
        Args:
            feats: (BxT_histxNobjxDxH'xW')
            vids: (BxT_histxCxHxW) The video corresponding to the feats, some
                dyn models might use them.
            n_fwd_times: Number of times to run the fwd model on the last frames
            need_intermediate: If true, give all the intermediate features
        Returns:
            all_preds: The predictions at each time step, in n_fwd_times
            all_pixs: The predictions in pixels. Note all dynamics models don't
                use pixels, so it might just give the last frame as output
            all_solved: The classification at each time step, for n_fwd_times
        """
        all_preds = []
        all_pixs = []
        all_addl_losses = []
        if n_fwd_times == 0:
            return [all_preds, all_pixs, all_addl_losses]

        def run_fwd_append(feats, pixs):
            pred, pred_pix, addl_losses = self.dyn(feats, pixs)
            all_preds.append(pred)
            all_pixs.append(pred_pix)
            all_addl_losses.append(addl_losses)

        run_fwd_append(feats, vids)
        n_fwd_times_copy = n_fwd_times
        while n_fwd_times - 1 > 0:
            feats = torch.cat(
                [feats[:, 1:, ...],
                 torch.unsqueeze(all_preds[-1], axis=1)],
                dim=1)
            vids = torch.cat(
                [vids[:, 1:, ...],
                 torch.unsqueeze(all_pixs[-1], axis=1)],
                dim=1)
            run_fwd_append(feats, vids)
            n_fwd_times -= 1
        assert len(all_preds) == n_fwd_times_copy, (
            '%d %d' % (len(all_preds), n_fwd_times_copy))
        if not need_intermediate:
            all_preds = [all_preds[-1]]
            all_pixs = [all_pixs[-1]]
            all_addl_losses = [all_addl_losses[-1]]
        # Will compute solved or not later, after decode, in case the classifier
        # needs that information
        return all_preds, all_pixs, all_addl_losses

    def _slice_for_dyn(self, features_batched, n_hist_frames, nslices=-1):
        """
        Args:
            features_batched: BxTx.... can deal with any following
                dimensions, typically it is (BxTxNobjxDxH'xW')
            n_hist_frames (int): Number of frames to use as history
            nslices (int): If -1, make as many slices of the training data
                as possible. If 1, keep only the first one. (1 used when
                training classifier on top, which should always see videos
                from the start)

        Returns:
            B'x n_hist_frames x ... (B'x n_hist_frames x Nobj x D x H' x W')
        """
        clip_hist = []
        assert features_batched.shape[1] >= n_hist_frames
        for i in range((features_batched.shape[1] - n_hist_frames + 1)):
            if nslices > 0 and i >= nslices:
                break
            clip_hist.append(features_batched[:, i:i + n_hist_frames, ...])
        clip_hist = torch.cat(clip_hist, dim=0)
        return clip_hist

    def _forward_dec(self, feats, pixels):
        """
        Args:
            feats: List of features (BxD) from the dynamics prediction stage,
                one for each time step predicted.
            pixels: List of corresponding pixels from the dynamics model. The
                dyn model may or may not actually generate new pixels.
        """
        return [self.dec(feat, pix) for feat, pix in zip(feats, pixels)]

    # Loss functions ###########################################################
    def cswm_loss(self, pred, gt, hinge=1.0):
        """
        The energy based contrastive loss.
        Args:
            pred (BxNobjxDxH'xW')
            gt (BxNobjxDxH'xW')
            From https://github.com/tkipf/c-swm/blob/master/modules.py#L94
        """
        pred = pred.view(pred.shape[:2] + (-1, ))
        gt = gt.view(gt.shape[:2] + (-1, ))
        batch_size = gt.size(0)
        perm = np.random.permutation(batch_size)
        neg = gt[perm]

        def energy(pred, gt, sigma=0.5):
            """Energy function based on normalized squared L2 norm.
            Args:
                pred (B, Nobj, D')
                gt (B, Nobj, D')
            """
            norm = 0.5 / (sigma**2)
            diff = pred - gt
            return norm * diff.pow(2).sum(2).mean(1)

        pos_loss = energy(pred, gt)
        zeros = torch.zeros_like(pos_loss)
        pos_loss = pos_loss.mean()
        neg_loss = torch.max(zeros, hinge - energy(pred, neg)).mean()
        return pos_loss + neg_loss

    def ce_loss(self, decisions, targets):
        targets = targets.to(dtype=torch.float, device=decisions.device)
        return torch.nn.functional.binary_cross_entropy_with_logits(
            decisions, targets)

    def autoencoder_loss(self, pix, latent, autoenc_loss_ratio):
        """
        Runs a random portion of the actual frames through decoder to incur a
        loss to encourage the intermediate representation to learn a good
        autoencoder as well. Random fraction only for compute reasons.
        Ideally would run every frame (ratio = 1)
        Args:
            pix (B, T, H, W): Actual pixels of the input frames
            latent (B, T, Nobj, D, H', W'): Latent representation of the input
                frames
            autoenc_loss_ratio (float): What percentage of the input frames to
                run it on. Only for compute reasons, ideally run it on all.
        Returns:
            loss {'autoenc': (1,) <float>} for the loss
        """
        # Flatten the Batch and time dimension to get all the frames
        pix_flat = torch.flatten(pix, 0, 1)
        latent_flat = torch.flatten(latent, 0, 1)
        # Select a subset of the frames to run the loss on
        assert pix_flat.shape[0] == latent_flat.shape[0]
        idx = np.arange(pix_flat.shape[0])
        np.random.shuffle(idx)
        sel_cnt = int(autoenc_loss_ratio * len(idx))
        idx_sel = np.sort(idx[:sel_cnt])
        pix_flat_sel = pix_flat[idx_sel, ...]
        latent_flat_sel = latent_flat[idx_sel, ...]
        # Generate the pixels for the latent, and incur loss
        pred_flat_sel = combine_obj_pixels(self.dec(latent_flat_sel, None), 1)
        loss = self.pix_loss(pred_flat_sel, pix_flat_sel).unsqueeze(0)
        return {'autoenc_pix': loss}

    def solved_or_not_loss(self, clip_preds_solved, vid_is_solved):
        """
        Repeat the is_solved to as many times the batch was repeated to get
        the class label at each forward prediction
        Args:
            clip_preds_solved (B',)
            vid_is_solved (B,)
            B and B' might be different but B' must be a multiple of B, since
                it happens when num_slices > 1
        Returns:
            loss {'ce': (1,) <float>} for the loss
        """
        assert clip_preds_solved.shape[0] % vid_is_solved.shape[0] == 0
        return {
            'ce':
            self.ce_loss(
                clip_preds_solved,
                vid_is_solved.repeat((clip_preds_solved.shape[0] //
                                      vid_is_solved.shape[0], ))).unsqueeze(0)
        }

    ############################################################################

    def _compute_losses(self, clip_pred, clip_pred_pix, vid_feat, vid,
                        n_hist_frames, n_fwd_times):
        """
        Compute all losses possible.
        """
        dummy_loss = torch.Tensor([-1]).to(clip_pred.device)
        losses = {}
        # NCE and pixel loss
        # find the GT for each clip, note that all predictions may not have a GT
        # since the last n_hist_frames for a video will make a prediction that
        # goes out of the list of frames that were extracted for that video.
        feat_preds = []
        feat_gt = []
        pix_preds = []
        pix_gt = []
        batch_size = vid_feat.shape[0]
        gt_max_time = vid_feat.shape[1]
        # Max slices that could have been made of the data, to use all of the
        # training clip
        max_slices_with_gt = gt_max_time - n_hist_frames - n_fwd_times + 1
        num_slices = clip_pred.shape[0] // batch_size
        for i in range(min(max_slices_with_gt, num_slices)):
            corr_pred = clip_pred[i * batch_size:(i + 1) * batch_size, ...]
            # Get the corresponding GT predictions for this pred
            corr_gt = vid_feat[:, i + n_hist_frames + n_fwd_times - 1]
            assert corr_gt.shape == corr_pred.shape
            feat_preds.append(corr_pred)
            feat_gt.append(corr_gt)
            # Same thing for pix
            if clip_pred_pix is not None:
                corr_pix_pred = clip_pred_pix[i * vid_feat.shape[0]:(i + 1) *
                                              vid_feat.shape[0], ...]
                corr_pix_gt = vid[:, i + n_hist_frames + n_fwd_times - 1]
                pix_preds.append(corr_pix_pred)
                pix_gt.append(corr_pix_gt)
        if len(feat_gt) > 0:
            # Keep a batch dimension to the loss, since it will be run over
            # multiple GPUs
            feat_preds = torch.cat(feat_preds)
            feat_gt = torch.cat(feat_gt)
            losses['nce'] = self.nce_loss(feat_preds, feat_gt).unsqueeze(0)
            losses['cswm'] = self.cswm_loss(feat_preds, feat_gt).unsqueeze(0)
        else:
            losses['nce'] = dummy_loss
            losses['cswm'] = dummy_loss

        # Reconstruction loss
        if len(pix_gt) > 0:
            losses['pix'] = self.pix_loss(torch.cat(pix_preds),
                                          torch.cat(pix_gt)).unsqueeze(0)
        else:
            losses['pix'] = dummy_loss
        return losses

    def _cls(self, feat_hist, pix_hist, feat_preds, pix_preds):
        """
        Wrapper around the classifier, collates all the input frames/features
            and predicted future frames/features.
            The images, features are already summed over the objects
        Args:
            feat_hist: (B, T, C, H', W')
            pix_hist: (B, T, 7, H, W)
            feat_preds [list of (B, C, H', W')] -- len = num predictions
            pix_preds [list of (B, 7, H, W)] -- len = num predictions
                The elements could be None, since not all models predict pixels
        Returns:
            (B,) predicted scores for the clips
        """
        feats_combined = feat_hist
        if feat_preds is not None and len(feat_preds) > 0:
            feats_combined = torch.cat([feat_hist] +
                                       [el.unsqueeze(1) for el in feat_preds],
                                       dim=1)
        pix_combined = pix_hist
        if (pix_preds is not None and len(pix_preds) > 0
                and pix_preds[0] is not None):
            pix_combined = torch.cat([pix_combined] +
                                     [el.unsqueeze(1) for el in pix_preds],
                                     dim=1)
        # Sum over objs -- we want the classifier model to see everything
        # at the same time
        # They are summed now, but need the dimension still
        pix_combined = pix_combined.unsqueeze(2)
        feats_combined = feats_combined.unsqueeze(2)
        # If need to keep only a subset of the frames
        if self.nframes_to_cls > 0:
            pix_combined = pix_combined[:, :self.nframes_to_cls, ...]
            feats_combined = feats_combined[:, :self.nframes_to_cls, ...]
        feats_combined = self.spat_att(feats_combined)
        # Keep the last prediction, as that should ideally be the best
        # prediction of whether it was solved or not
        # torch.max was hard to optimize through
        return self.cls(feats_combined, pix_combined)[:, -1]

    def forward(self,
                vid,
                vid_is_solved,
                n_hist_frames=3,
                n_fwd_times=1,
                n_fwd_times_incur_loss=999999,
                run_decode=False,
                compute_losses=False,
                need_intermediate=False,
                autoenc_loss_ratio=0.0,
                nslices=-1):
        """
        Args:
            vid: (BxTxNobjxHxW) The input video
            vid_is_solved: (Bx1) Whether the video is solved in the end of not.
                Could be None at test time.
            n_hist_frames: (int) Number of frames to use as history for
                prediction
            n_fwd_times: (int) How many times to run the forward dynamics model
            n_fwd_times_incur_loss (int): Upto how many of these forwards to
                incur loss on.
            run_decode: (bool) Decode the features into pixel output
            compute_losses: Should be set at train time. Will compute losses,
                whatever it can given the data (eg, if vid_is_solved is not
                passed to the function, it will not compute the CE loss).
            need_intermediate (bool): Set true if you want to run the dynamics
                model and need all the intermediate results. Else, will return
                a list with only 1 element, the final output.
            autoenc_loss_ratio (float btw 0-1): Set to 1 to run auto-encoder
                style loss on all frames when run_decode is set.
            num_slices (int): See in the _slice_for_dyn fn
        Returns:
            clip_feat: BxTxD
        """
        vid_preproc = self.preproc.preprocess_vid(vid)
        obj_feat = self.enc(vid_preproc)
        clip_hist = self._slice_for_dyn(obj_feat,
                                        n_hist_frames,
                                        nslices=nslices)
        vid_hist = self._slice_for_dyn(vid_preproc,
                                       n_hist_frames,
                                       nslices=nslices)
        assert clip_hist.shape[1] == n_hist_frames
        clip_hist = self.interactor(clip_hist)
        clip_preds, clip_preds_pix, clip_preds_addl_losses = self._forward_dyn(
            clip_hist, vid_hist, n_fwd_times, need_intermediate)
        if run_decode:
            clip_preds_pix = self._forward_dec(clip_preds, clip_preds_pix)
        else:
            clip_preds_pix = [None] * len(clip_preds)
        # Compute the solved or not, will only do for the ones asked for
        clip_preds_solved = self._cls(
            combine_obj_pixels(clip_hist, 2), combine_obj_pixels(vid_hist, 2),
            [combine_obj_pixels(el, 1) for el in clip_preds],
            [combine_obj_pixels(el, 1) for el in clip_preds_pix])
        all_losses = []
        clip_preds_pix_unpreproc_for_loss = [
            self.preproc.unpreprocess_frame_for_loss(el)
            for el in clip_preds_pix
        ]
        if compute_losses:
            for i in range(min(len(clip_preds), n_fwd_times_incur_loss)):
                # Compute losses at each prediction step, if need_intermediate
                # is set. Else, it will only return a single output
                # (at the last prediction), and then we can only incur loss at
                # that point.
                if not need_intermediate:
                    assert len(clip_preds) == 1
                    pred_id = -1
                    # Only loss on predicting the final rolled out obs
                    this_fwd_times = n_fwd_times
                else:
                    assert len(clip_preds) == n_fwd_times
                    pred_id = i
                    this_fwd_times = i + 1
                all_losses.append(
                    self._compute_losses(
                        # For the loss, using only the last prediction (for now)
                        clip_preds[pred_id],
                        combine_obj_pixels(
                            clip_preds_pix_unpreproc_for_loss[pred_id], 1),
                        obj_feat,
                        combine_obj_pixels(vid, 2),
                        n_hist_frames,
                        this_fwd_times))
            all_losses = average_losses(all_losses)
            all_losses.update(average_losses(clip_preds_addl_losses))
            all_losses.update(
                self.solved_or_not_loss(clip_preds_solved, vid_is_solved))
            # Add losses on the provided frames if requested
            if run_decode and autoenc_loss_ratio > 0:
                all_losses.update(
                    self.autoencoder_loss(combine_obj_pixels(vid, 2), obj_feat,
                                          autoenc_loss_ratio))
        clip_preds_pix_unpreproc = [
            combine_obj_pixels(self.preproc.unpreprocess_frame_after_loss(el),
                               1) for el in clip_preds_pix_unpreproc_for_loss
        ]
        all_preds = {
            'feats': clip_preds,
            'is_solved': clip_preds_solved,
            'pixels': clip_preds_pix_unpreproc,
        }
        return all_preds, all_losses
