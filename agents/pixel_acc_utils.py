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
"""Util functions for frame accuracy."""

import os
import os.path as osp
import logging
import numpy as np
from PIL import Image, ImageSequence
from tqdm import tqdm
from functools import partial
import multiprocessing as mp

NUM_FOLDS = 10
ALL_FOLDS = range(NUM_FOLDS)


def _hex_to_ints(hex_string):
    hex_string = hex_string.strip('#')
    return (
        int(hex_string[0:2], 16),
        int(hex_string[2:4], 16),
        int(hex_string[4:6], 16),
    )


RED = _hex_to_ints('f34f46')
GREEN = _hex_to_ints('6bcebb')
BLUE = _hex_to_ints('1877f2')
GRAY = _hex_to_ints('b9cad2')
L_RED = _hex_to_ints('fcdfe3')


def pixel_accuracy(prediction, gt):
    is_close_per_channel = np.isclose(prediction, gt)
    all_channels_close = is_close_per_channel.sum(
        axis=-1) == prediction.shape[-1]
    return np.sum(all_channels_close) / (prediction.shape[0] *
                                         prediction.shape[1])


def zero_out_non_moving_channels(img):
    is_red = np.isclose(img, RED)
    is_green = np.isclose(img, GREEN)
    is_blue = np.isclose(img, BLUE)
    is_gray = np.isclose(img, GRAY)
    is_l_red = np.isclose(img, L_RED)
    img[~(is_red | is_green | is_blue | is_gray | is_l_red)] = 0.0
    return img


def pixel_accuracy_moving_channels(prediction, gt):
    prediction = zero_out_non_moving_channels(prediction)
    gt = zero_out_non_moving_channels(gt)
    is_close_per_channel = np.isclose(prediction, gt)
    all_channels_close = is_close_per_channel.sum(
        axis=-1) == prediction.shape[-1]
    return np.sum(all_channels_close) / (prediction.shape[0] *
                                         prediction.shape[1])


def compute_per_ts_acc(gt_pred_gif_pair, start_frame=3, predicted_frames=10):
    res = np.full((predicted_frames, ), np.nan)
    for path in gt_pred_gif_pair:
        # If either doesnt exist, just return nans
        if not osp.exists(path):
            logging.debug('Rollout does not exist, returning nan: %s', path)
            return res
    gt_iter = ImageSequence.Iterator(Image.open(gt_pred_gif_pair[0]))
    prediction_iter = ImageSequence.Iterator(Image.open(gt_pred_gif_pair[1]))

    for _ in range(start_frame):
        next(gt_iter)
        next(prediction_iter)

    for i in range(predicted_frames):
        predicted_frame = np.array(next(prediction_iter).convert('RGB'))
        gt_frame = np.array(next(gt_iter).convert('RGB'))
        res[i] = pixel_accuracy_moving_channels(predicted_frame, gt_frame)
    return res


### Code to actually get the number


def get_vis_outdirs(conf_path_run_ids):
    conf_path, run_id = conf_path_run_ids.split(':')
    start, end = run_id.split('-')
    fold_run_ids = range(int(start), int(end) + 1)
    output_dir = osp.join('outputs/', conf_path)
    all_fold_output_dirs = []
    for run_id in fold_run_ids:
        this_output_dir = osp.join(output_dir, str(run_id), 'vis/eval_vis/')
        all_fold_output_dirs.append(this_output_dir)
    return all_fold_output_dirs


def get_gen_vis_list(conf_path_run_ids_gt,
                     conf_path_run_ids,
                     setting,
                     folds_to_use=ALL_FOLDS):
    if setting == 'within':
        expected_temp_ids = 25
        expected_tasks_actions = 20 * len(folds_to_use)
    elif setting == 'cross':
        expected_temp_ids = 5
        expected_tasks_actions = 100 * len(folds_to_use)
    else:
        raise NotImplementedError(f'Unknown setting {setting}')
    gt_output_dirs = get_vis_outdirs(conf_path_run_ids_gt)
    pred_output_dirs = get_vis_outdirs(conf_path_run_ids)
    assert len(gt_output_dirs) == len(pred_output_dirs) == len(folds_to_use)
    fpa_to_compute = []
    fpa_to_compute_sanity = []
    for fold_id in folds_to_use:
        # Get the task IDs
        temp_ids = os.listdir(gt_output_dirs[fold_id])
        if len(temp_ids) != expected_temp_ids:
            logging.warning('Only found %d temp_ids [%s], expected %d',
                            len(temp_ids), temp_ids, expected_temp_ids)
        for temp in temp_ids:
            gt_task_dir = osp.join(gt_output_dirs[fold_id], temp)
            pred_task_dir = osp.join(pred_output_dirs[fold_id], temp)
            # Get all the actions evaluated
            action_subdirs = os.listdir(gt_task_dir)
            if len(action_subdirs) != expected_tasks_actions:
                logging.debug('Found only %d subdirs in %s',
                              len(action_subdirs), gt_task_dir)
                # Add dummy actions for now, will be ignored but pointed out
                action_subdirs = (
                    action_subdirs + ['dummy'] *
                    (expected_tasks_actions - len(action_subdirs)))
            for action_subdir in action_subdirs:
                fpa_to_compute.append([
                    fold_id, temp, action_subdir,
                    osp.join(gt_task_dir, action_subdir, 'gt/combined.gif'),
                    osp.join(pred_task_dir, action_subdir,
                             'predictions/combined.gif')
                ])
                fpa_to_compute_sanity.append([
                    fold_id, temp, action_subdir,
                    osp.join(gt_task_dir, action_subdir, 'gt/combined.gif'),
                    osp.join(pred_task_dir, action_subdir, 'gt/combined.gif')
                ])
    return fpa_to_compute, fpa_to_compute_sanity


def compute_rollout_accuracy(conf_path_run_ids_gt, conf_path_run_ids,
                             start_frame, predicted_frames, setting,
                             folds_to_use):
    fpa_to_compute, fpa_to_compute_sanity = get_gen_vis_list(
        conf_path_run_ids_gt, conf_path_run_ids, setting, folds_to_use)
    logging.info('For %s', conf_path_run_ids)
    with mp.Pool(32) as pool:
        fpa = np.array(
            list(
                tqdm(
                    pool.imap(
                        partial(compute_per_ts_acc,
                                start_frame=start_frame,
                                predicted_frames=predicted_frames),
                        # Keep only the predicted and GT gifs, remove other params
                        [el[-2:] for el in fpa_to_compute]),
                    total=len(fpa_to_compute),
                    desc='Computing FPA')))
        fpa_sanity = np.array(
            list(
                tqdm(
                    pool.imap(
                        partial(
                            compute_per_ts_acc,
                            # No need to run this one for all the frames, just
                            # need enough to make sure the GT corresponds
                            start_frame=1,
                            predicted_frames=3),
                        # Keep only the predicted and GT gifs, remove others
                        [el[-2:] for el in fpa_to_compute_sanity]),
                    total=len(fpa_to_compute_sanity),
                    desc='Computing FPA Sanity')))
    not_computed = np.sum(np.isnan(fpa.sum(axis=1)))
    if not_computed > 0:
        # This likely happens because some tasks may not have enough positive
        # actions (that solve) to sample 5 positive actions.. hence it just
        # returns the same action multiple times and they all get saved on top
        # of each other into the same folder. just ignore them
        logging.error('Found %d/%d (fold, task, actions) not computed.',
                      not_computed,
                      np.shape(fpa)[0])
    assert np.isclose(np.nanmean(fpa_sanity), 1.0), 'GT not match!!'
    # Reshape, to get the fold axis: (N*fold, T) -> (fold, N, T)
    fpa = np.reshape(fpa, (len(folds_to_use), -1, np.shape(fpa)[-1]))
    fpa = np.nanmean(fpa, axis=1) * 100.0  # (fold, T)
    return fpa.transpose().tolist()  # (T, fold)


def main():
    fpa_values = compute_rollout_accuracy(
        # The path to a config followed by the run_ids that will be used to
        # get the ground truth rollouts. Ideally it should be the run_ids that
        # correspond to the 10 folds to get the final number, but like here
        # we can also just run it for a single fold -- in this case fold 0
        # for the joint model trained for n_fwd_times=10
        'expts/joint/000_joint_DEC_3f_win.txt:70-70',
        # The path to the config (+ run ids) that will be evaluated. It can be
        # the same as above since we always store the GT rollouts when producing
        # rollouts.
        'expts/joint/000_joint_DEC_3f_win.txt:70-70',
        start_frame=3,
        predicted_frames=10,
        setting='within',
        # Set this to the folds you are using for evaluation. In this case,
        # we are only using fold 0 (which corresponds to the run_id 70).
        folds_to_use=[0])
    print('FPA values over time', fpa_values)
    print('FPA avg over 10s rollout', np.mean(fpa_values))


if __name__ == '__main__':
    main()
