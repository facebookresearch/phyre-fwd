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
"""Preprocessing functions."""

import torch
from torch import nn
import hydra
import phyre


class OneHotPreprocCore(nn.Module):
    """Preprocess video as 1-hot rep."""
    def __init__(self):
        super().__init__()
        self.register_buffer('embed_weights', torch.eye(phyre.NUM_COLORS))
        self.out_dim = phyre.NUM_COLORS

    def _image_colors_to_onehot(self, indices):
        onehot = torch.nn.functional.embedding(
            indices.to(dtype=torch.long, device=self.embed_weights.device),
            self.embed_weights)
        onehot = onehot.permute(0, 3, 1, 2).contiguous()
        return onehot

    def preproc(self, frame):
        """
        Args:
            frame (B, H, W)
        Returns:
            processed frame (B, C, H, W)
        """
        return self._image_colors_to_onehot(frame)

    @classmethod
    def unpreproc_for_loss(cls, frame):
        return frame

    @classmethod
    def unpreproc_after_loss(cls, proc_frame_for_loss):
        """
        Args:
            frame (B, C, H, W)
        Returns:
            processed frame (B, H, W)
        """
        return torch.argmax(proc_frame_for_loss, axis=1)


class OneHotPtNetPreprocCore(nn.Module):
    """Preprocess video as 1-hot rep."""
    def __init__(self):
        super().__init__()
        self.one_hot_preproc = OneHotPreprocCore()
        self.out_dim = phyre.NUM_COLORS * 3

    def preproc(self, frame):
        """
        Args:
            frame (B, H, W)
        Returns:
            processed frame (B, C, H, W)
        """
        # First make it 1-hot rep, then add the XY locations
        frame_onehot = self.one_hot_preproc.preproc(frame)
        # Make space for the X, Y
        frame_onehot_rep = frame_onehot.repeat_interleave(3, dim=1)
        # Compute the XY grid
        loc_x, loc_y = torch.meshgrid(torch.arange(frame.shape[-2]),
                                      torch.arange(frame.shape[-1]))
        frame_onehot_rep[:, 0::3, :, :] = loc_x
        frame_onehot_rep[:, 1::3, :, :] = loc_y
        return frame_onehot_rep

    @classmethod
    def unpreproc_for_loss(cls, proc_frame):
        """Generate a 1-hot pixel-level output to incur the loss."""
        proc_frame_ch = torch.chunk(proc_frame, phyre.NUM_COLORS, 1)
        all_feat = []
        for channel in proc_frame_ch:
            index = channel[:, :2, ...]
            index[:, 0, ...] = 2 * (index[:, 0, ...] / index.shape[-2]) - 1
            index[:, 1, ...] = 2 * (index[:, 1, ...] / index.shape[-1]) - 1
            feat = channel[:, 2:, ...]  # B, 1 (typically), H, W
            all_feat.append(
                nn.functional.grid_sample(feat,
                                          index.permute(0, 3, 2, 1),
                                          mode='bilinear',
                                          align_corners=True))
        return torch.cat(all_feat, dim=1)

    def unpreproc_after_loss(self, proc_frame_for_loss):
        """
        Args:
            proc_frame (B, C, H, W)
        Returns:
            frame (B, H, W)
        """
        return self.one_hot_preproc.unpreproc_after_loss(proc_frame_for_loss)


class VideoPreprocessor(nn.Module):
    """Basic video preprocessor."""
    def __init__(self, agent_cfg):
        super().__init__()
        self.preproc_core = hydra.utils.instantiate(agent_cfg.preproc_core)
        self.out_dim = self.preproc_core.out_dim

    @classmethod
    def _apply_each_obj(cls, func, frame):
        """Apply a function to each obj in the frame.
        Args:
            func
            frame: B, Nobj, ...
        """
        frame_flat = torch.flatten(frame, 0, 1)
        frame_flat_proc = func(frame_flat)
        frame_proc = frame_flat_proc.view(frame.shape[:2] +
                                          frame_flat_proc.shape[1:])
        return frame_proc

    def preproc_frame(self, frame):
        """Process a frame from the vid.
        Args:
            frame: (B,Nobj,H,W)
        Returns:
            Processed frame: (B,Nobj,C,H,W)
        """
        if frame is None:
            return None
        assert len(frame.shape) == 4
        return self._apply_each_obj(self.preproc_core.preproc, frame)

    def unpreprocess_frame_after_loss(self, proc_frame):
        """Unprocess a frame from the vid, that has already been unprocessed
            for loss using the unprocess_frame_for_loss function.
            Note that the decoder automatically handles objects, so no obj here
        Args:
            processed frame: (B,Nobj,C,H,W)
        Returns:
            frame: (B, Nobj, H, W)
        """
        if proc_frame is None:
            return None
        assert len(proc_frame.shape) == 5
        return self._apply_each_obj(self.preproc_core.unpreproc_after_loss,
                                    proc_frame)

    def unpreprocess_frame_for_loss(self, proc_frame):
        """Unprocess a frame from the vid, for loss.
        Args:
            processed frame: (B,Nobj,C,H,W)
        Returns:
            frame: (B, Nobj, C, H, W)
        """
        if proc_frame is None:
            return proc_frame
        assert len(proc_frame.shape) == 5
        return self._apply_each_obj(self.preproc_core.unpreproc_for_loss,
                                    proc_frame)

    def preprocess_vid(self, vid):
        """
        Args:
            vid (B, T, Nobj, H, W)
        Returns:
            res (B, T, Nobj, C, H, W): Basically the 1-hot representation
        """
        assert len(vid.shape) == 5
        vid_flat = torch.flatten(vid, 0, 1)
        vid_flat_onehot = self.preproc_frame(vid_flat)
        return torch.reshape(vid_flat_onehot,
                             vid.shape[:2] + vid_flat_onehot.shape[1:])
