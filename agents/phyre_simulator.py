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
""" Phyre simulator wrapper. """
import logging
import numpy as np
from scipy import ndimage
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import torch

import phyre
MAX_NUM_OBJECTS = 62


class PhyreSimulator:
    """Just a wrapper around the simulator to© be configurable."""
    def __init__(self, simulator, obj_fwd_model, stride, split_conn_comp, movable_ch,
                 prepend_empty_frames, stop_after_solved):
        """
        Args:
            simulator: The original phyre simulator
            stride (int): The stride to run the simulation at
            split_conn_comp (int): The number of objects to break the outupt
                into
            movable_ch (list): The indices of phyre channels that are movable.
                Will be used to club the channels together.
            prepend_empty_frames (int): Use this to prepend n_hist_frames-1
                empty frames, so both at test and train time, the model only
                looks at the 1 frame to make predictions, to be comparable to
                the standard evaluation on phyre. List the number of frames to
                be added. Note they will only be added to clips starting from
                0th frame. Any clips sampled from middle will not have the
                empty frames.
            stop_after_solved (bool): Set this to false to keep running the
                simulations if it's ended too. This is useful when training
                forward models, as otherwise we'll append the last frame
                repeated. Setting to true as default to repro old expts, though
                false should be ideal.
        """
        self.simulator = simulator
        self.stride = stride
        self.split_conn_comp = split_conn_comp
        self.movable_ch = movable_ch
        self.non_movable_ch = list(  # ignoring background
            set(list(range(1, phyre.NUM_COLORS))) - set(movable_ch))
        self.prepend_empty_frames = prepend_empty_frames
        self.stop_after_solved = stop_after_solved
        self.obj_fwd_model = obj_fwd_model

    def _roll_fwd_obs(self, task_indices, actions, roll_fwd_ratio, nframes):
        """
        Use the simulator to roll forward the simulation (for that action) and
        return the output. To get an upper bound on how fwd might work.
        Args:
            task_indices: This is the *integer* indexes into the IDs that the
                simulator is initialized with. So if the simulator was init
                with 3 tasks ['0001:224', '...' , '...'], then 0 will run the
                0th task -- 0001:224.
            actions: The actual actions you want to take in that task. Must a
                cpu numpy matrix.
            simulator: The initialized simulator with the task IDs.
            roll_fwd_ratio: 0-1 for where to pick the start point.
                -1 => pick randomly such that nframes is satisfied. In that case
                nframes can not be -1 as well.
            nframes (T): -1=> all the frames following that point
        Returns:
            new_observations: [TxHxW] -- the rolled out video for each batch
                elt (So total B elements in the list). We don't make a np.array
                of it since each might have diff number of frames, dep on how
                much it rolled out at test/train time etc.
        """
        assert 0.0 <= roll_fwd_ratio < 1.0 or roll_fwd_ratio == -1, 'Limits'
        new_observations = []
        new_obj_observations = []
        for i, task_idx in enumerate(task_indices):
            kwargs = dict(
                need_images=True,
                need_featurized_objects=True,
                stride=self.stride,
                perturb_step=-1,
                stop_after_solved=self.stop_after_solved,
            )
            # If the simulation has to start at the beginning and for fixed set
            # of frames, then might as well specify that and make it run faster.
            # Will specially help the test time performance.
            if roll_fwd_ratio == 0 and nframes != -1:
                frames_to_ask = nframes
                if self.prepend_empty_frames < 0:
                    # This many frames will be dropped, so ask for those many
                    # extra
                    frames_to_ask += (-self.prepend_empty_frames)
                kwargs.update({'nframes': frames_to_ask * self.stride})
            simulation = self.simulator.simulate_action(
                task_idx, actions[i], **kwargs)
            images = simulation.images
            objs = None
            if simulation.featurized_objects is not None:
                objs = simulation.featurized_objects.features
            if images is None:
                # This means the action was invalid. This should only happen at
                # test time when we don't filter for the valid actions only.
                # For now, returning the original obs as a single frame clip
                images = np.expand_dims(
                    self.simulator.initial_scenes[task_idx], 0)
                objs = self.simulator.initial_featurized_objects[
                    task_idx].features
                if 'nframes' in kwargs:
                    images = np.tile(images,
                                     (kwargs['nframes'] // self.stride, 1, 1))
                    objs = np.tile(objs,
                                   (kwargs['nframes'] // self.stride, 1, 1))
            if self.prepend_empty_frames > 0:
                empty_frames = np.zeros(
                    (self.prepend_empty_frames, ) + images.shape[1:],
                    dtype=images.dtype)
                images = np.concatenate([empty_frames, images], axis=0)
                empty_obj_frames = np.zeros(
                    (self.prepend_empty_frames, ) + objs.shape[1:],
                    dtype=images.dtype)
                objs = np.concatenate([empty_obj_frames, objs], axis=0)
            elif self.prepend_empty_frames < 0:
                # Drop these many frames from the beginning
                assert images.shape[0] > -self.prepend_empty_frames
                assert objs.shape[0] > -self.prepend_empty_frames
                images = images[-self.prepend_empty_frames:]
                objs = objs[-self.prepend_empty_frames:]
                # Remove this many frames from the beginning of the
            # To debug/visualize
            # if images is None:
            #     T = phyre.vis.observations_to_uint8_rgb(
            #         observations[i].cpu().numpy())
            #     import matplotlib.pyplot as plt
            #     plt.imsave('/private/home/rgirdhar/temp/prev.jpg', T)
            # phyre.vis.save_observation_series_to_gif(
            #     [images.tolist()], '/private/home/rgirdhar/temp/prev.gif')
            if roll_fwd_ratio == -1:
                assert nframes != -1, 'Cant pick start point randomly...'
                this_roll_fwd_ratio = max(
                    np.random.random() * (1 - (nframes / images.shape[0])), 0)
            else:
                this_roll_fwd_ratio = roll_fwd_ratio
            split_pt = int(images.shape[0] * this_roll_fwd_ratio)
            if nframes == -1:  # select all following images
                this_nframes = images.shape[0] - split_pt
            else:
                this_nframes = nframes
            clip = images[split_pt:split_pt + this_nframes, ...]
            obj_clip = objs[split_pt:split_pt + this_nframes, ...]
            # Pad with the last frame repeated if nframes too less
            if this_nframes not in [-1, clip.shape[0]]:
                assert self.stop_after_solved, (
                    f'If stop_after_solved is False, then it should always '
                    f'return enough frames! Returned clip of shape '
                    f'{clip.shape} while expected {this_nframes}')
                logging.debug('Have to pad with %d frames to meet %d nframes',
                              this_nframes - clip.shape[0], this_nframes)
                clip = np.concatenate([
                    clip,
                    np.tile(clip[-1:, ...],
                            [this_nframes - clip.shape[0], 1, 1])
                ], 0)
                obj_clip = np.concatenate([
                    obj_clip,
                    np.tile(obj_clip[-1:, ...],
                            [this_nframes - obj_clip.shape[0], 1, 1])
                ], 0)
            # Add the channel dimension
            new_observations.append(clip)
            new_obj_observations.append(obj_clip)
        return new_observations, new_obj_observations

    def _pad_simulation_objects(self, obj_clip):
        """Pad the clip so that it has max number of objects.
        Args:
            clip = TxNxF frame clips of objects
        Returns
             TxMAX_OBJxF padded frame clip
        """
        permuted_clip = torch.FloatTensor(obj_clip).permute(1, 0, 2)
        pad = torch.zeros(
            (MAX_NUM_OBJECTS, permuted_clip.shape[1], permuted_clip.shape[2]))
        padded = torch.nn.utils.rnn.pad_sequence((pad, permuted_clip),
                                                 batch_first=True,
                                                 padding_value=0)
        padded_clip = padded.permute(0, 2, 1, 3)[1:].squeeze(0).numpy()
        return padded_clip

    def _order_via_matching(self, cur_objs, prev_objs):
        """Order the cur_objs by hungarian matching the to prev_objs.
        Args:
            cur_objs: [(H, W)] List of images
            prev_objs: [(H, W)] List of images
        Returns
            cur_objs, re-sorted using the hungarian matching.
        """
        if len(cur_objs) != len(prev_objs):
            logging.debug('Mismatch in nobj found, not matching...')
            return cur_objs
        num_obj = len(cur_objs)
        if num_obj < 2:
            return cur_objs  # Nothing to sort in this case
        dists = cdist(
            np.stack(cur_objs).reshape((num_obj, -1)) > 0,
            np.stack(prev_objs).reshape((num_obj, -1)) > 0, 'cosine')
        _, prev_ind = linear_sum_assignment(dists)
        res = [None] * num_obj
        for i, p_i in enumerate(prev_ind):
            res[p_i] = cur_objs[i]
        return res

    def _split_conn_comps(self, clip):
        """Split the frame into connected components.
        Args:
            clip = TxHxW frame clips
        Returns
            TxNobjxHxW, where each channel contains a single object, with the
            orig channel color instead of 1s.
        """
        if self.split_conn_comp == 1:
            return np.expand_dims(clip, 1)
        res = []
        channels = [{} for _ in range(clip.shape[0])]
        for time_step in range(clip.shape[0]):
            # No point getting the background as obj, so start from 1
            for col in range(1, phyre.NUM_COLORS):
                ch_per_col = []
                mappings, nmap = ndimage.label(clip[time_step] == col)
                for map_id in range(1, nmap + 1):
                    ch_per_col.append(clip[time_step] * (mappings == map_id))
                # Try to make sure the objects are added in the same order as
                # in the previous time step
                if time_step > 0:
                    ch_per_col = self._order_via_matching(
                        ch_per_col, channels[time_step - 1][col])
                channels[time_step][col] = ch_per_col
            all_obj = sum(channels[time_step].values(), [])
            tot_obj = len(all_obj)
            if tot_obj > self.split_conn_comp:
                # Try to separate multiple objects of same movable colors, the
                # remaining all will be combined together back into a single obj
                # image (since there's nothing better I can do given the limit)
                # Keep the movable channel objects separate as much possible,
                # and club all the rest together
                all_obj = []
                # Traverse this list in descending order of colors with most
                # number of objects. Ideally want to split apart objects of
                # the same color
                for col in sorted(
                        self.movable_ch,
                        key=lambda x, t=time_step, c=channels: -len(c[t][x])):
                    all_obj += channels[time_step][col]
                # Then add the non movable colors
                for col in self.non_movable_ch:
                    all_obj += channels[time_step][col]
                all_obj = all_obj[:self.split_conn_comp - 1] + [
                    np.max(np.stack(all_obj[(self.split_conn_comp - 1):]),
                           axis=0)
                ]
            else:
                # Add channels with 0s in the end
                all_obj += [np.zeros_like(clip[time_step])
                            ] * (self.split_conn_comp - tot_obj)
            res.append(np.stack(all_obj))
        res = np.stack(res)
        # # To debug, store the different channels to disk and see if they match
        # phyre.vis.save_observation_series_to_gif(
        #     [[np.max(res, axis=1)]], '/private/home/$USER/temp/res.gif')
        # for i in range(self.split_conn_comp):
        #     phyre.vis.save_observation_series_to_gif(
        #         [[res[:, i, ...]]], '/private/home/$USER/temp/res%02d.gif' % i)
        # phyre.vis.save_observation_series_to_gif(
        #     [[clip]], '/private/home/$USER/temp/clip.gif')
        return res

    def _roll_fwd_model(self, objects, images, n_fwd_times):
        pix_rollout = images[:]
        assert n_fwd_times >= 0
        if n_fwd_times == 0:
            return images, objects
        full_obj_rollout, roll_pred = self.obj_fwd_model.fwd_only_dyn(
                torch.FloatTensor(objects),
                n_hist_frames=self.obj_fwd_model.n_hist_frames,
                n_fwd_times=n_fwd_times)
        for i in range(roll_pred.shape[0]):
            rendered_rollout = []
            for j in range(roll_pred.shape[1]):
                img = phyre.objects_util.featurized_objects_vector_to_raster(roll_pred[i][j].detach().numpy())
                rendered_rollout.append(img)

            pix_rollout[i] = np.concatenate((pix_rollout[i], rendered_rollout), axis = 0)
            #pix_rollout[i]np.array(rendered_rollout, dtype=np.long))
        return pix_rollout, full_obj_rollout.detach().numpy()

    def sim(self, task_indices, actions, roll_fwd_ratio, nframes):
        if self.obj_fwd_model is None:
            final_obs, final_objs = self._roll_fwd_obs(task_indices, actions, roll_fwd_ratio,
                                        nframes)
        else:
            sim_hist_frames = self.obj_fwd_model.n_hist_frames
            obs, objs = self._roll_fwd_obs(task_indices, actions, roll_fwd_ratio,
                            sim_hist_frames)
            objs = [self._pad_simulation_objects(clip) for clip in objs]
            final_obs, final_objs = self._roll_fwd_model(objs, obs, nframes-sim_hist_frames)
        final_obs = [self._split_conn_comps(clip) for clip in final_obs]
        final_objs = [self._pad_simulation_objects(clip) for clip in final_objs]
        # Check that batch size and number of frames is same for obj/vid
        assert len(final_obs) == len(final_objs)
        #logging.info(f'batch size {len(final_obs)}, shape {np.shape(final_obs[0])}')
        assert np.shape(final_obs[0])[0] == np.shape(final_objs[0])[0]
        return final_obs, final_objs
