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
"""This library contains actual implementation of Fwd modeling."""
import json
import logging
import os
import time
from typing import Sequence, Tuple

import hydra
import numpy as np
import torch
from torch import nn

import phyre

import im_fwd_agent
import nets
from neural_agent import get_latest_checkpoint
import obj_nets

# For evaluation while training (small, to get a quick number)
AUCCESS_EVAL_TASKS = 200
XE_EVAL_SIZE = 10000

TaskIds = Sequence[str]
NeuralModel = nn.Module
TrainData = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, phyre.
                  ActionSimulator, torch.Tensor]


def gen_vis_vid_preds(orig_vid,
                      orig_objs,
                      model,
                      n_fwd_times=None,
                      run_decode=True,
                      n_hist_frames=3):
    """
    Generate a visualization of some training videos, along with model rollout
    (actual autoregressive rollout, so need to test again).
    Args:
        orig_vid: (B, T, Nobj, H, W) video batch
        model: the pytorch model for forward prediction
    Returns:
        RGB frames (B, T, 3, H, W) as torch tensor, in the standard format that
            can be used with tensorboard.
    """
    # Generate the predictions
    if n_fwd_times is None:
        n_fwd_times = orig_vid.shape[1] - n_hist_frames  # As many we've GT for
    # For vis, at least 1 frame would be needed for following code
    n_fwd_times = max(n_fwd_times, 1)
    vid = orig_vid[:, :n_hist_frames, ...]  # crop out the first part for pred
    objs = orig_objs[:, :n_hist_frames, ...]
    with torch.no_grad():
        model.eval()
        logging.info('gen vis preds')
        all_preds, _ = model.forward(objs,
                                     None,
                                     n_hist_frames=n_hist_frames,
                                     n_fwd_times=n_fwd_times,
                                     compute_losses=False,
                                     need_intermediate=True,
                                     run_decode=run_decode,
                                     nslices=1)
    stacked, _, _, _ = im_fwd_agent.ImgTrainer.vis_stacked_pred_gt(
        nets.combine_obj_pixels(orig_vid, 2).cpu().numpy(),
        nets.combine_obj_pixels(vid, 2),
        all_preds['pixels'] if run_decode else None)
    # For some reason need to flip the image in space and time for corr vis
    stacked_rgb = np.array(
        im_fwd_agent.convert_to_video_vis(stacked).transpose((0, 1, 4, 2, 3)))
    return torch.as_tensor(stacked_rgb)


class ObjTrainer(im_fwd_agent.ImgTrainer):
    """
    Trainer for object space forward modeling
    """
    @classmethod
    def load_agent_from_folder(cls,
                               model: NeuralModel,
                               agent_folder: str,
                               strict: bool = True) -> NeuralModel:
        """
        This loader is used in the offline_agents code, to load at test time.
        """
        last_checkpoint = get_latest_checkpoint(agent_folder)
        assert last_checkpoint is not None, agent_folder
        logging.info('Loading a model from: %s', last_checkpoint)
        last_checkpoint = torch.load(last_checkpoint)
        try:
            model.module.classification_model.pos_encoder.pe = model.module.classification_model.pos_encoder.pe.contiguous()
        except:
            pass
        missing_keys, unexp_keys = model.load_state_dict(
            last_checkpoint['model'], strict=strict)
        logging.warning('Could not init: %s', missing_keys)
        logging.warning('Unused keys in ckpt: %s', unexp_keys)
        model.to(nets.DEVICE)
        return model
    @classmethod
    def gen_model(cls, cfg, override_cfg=None, on_cpu=False):
        """Generate the random init model."""
        if override_cfg is not None:
            model = obj_nets.FwdObject(override_cfg)
        else:
            model = obj_nets.FwdObject(cfg)
        if on_cpu:
            return model.cpu()
        assert cfg.num_gpus <= torch.cuda.device_count()
        model = torch.nn.DataParallel(model,
                                      device_ids=list(range(cfg.num_gpus)))
        return model

    @classmethod
    def train(cls, model, dataset, output_dir, summary_writer,
              full_eval_from_model, cfg):
        """Main train function."""
        updates = cfg.train.num_iter
        report_every = cfg.train.report_every
        save_checkpoints_every = cfg.train.save_checkpoints_every
        full_eval_every = cfg.train.full_eval_every
        train_batch_size = cfg.train.batch_size
        max_frames_fwd = cfg.train.frames_per_clip
        n_hist_frames = cfg.train.n_hist_frames  # Frames used to predict the future
        loss_cfg = cfg.train.obj_loss
        opt_params = cfg.opt
        n_fwd_times = cfg.train.n_fwd_times


        # nslices (slice out the input for training)
        num_slices = cfg.train.num_slices

        if max_frames_fwd is not None and (max_frames_fwd <= n_hist_frames):
            logging.warning(
                'Cant train prediction model, max_frames_fwd too low')

        device = nets.DEVICE
        model.train()
        model.to(device)
        logging.info("%s", model)
        train_modules_subset = cfg.train.modules_to_train
        params_to_train = []
        if train_modules_subset is not None:
            mod_names = train_modules_subset.split('%')
            logging.warning(
                'Training only a few modules, listed below. NOTE: '
                'BNs/dropout will still be in train mode. Explicitly '
                'set those to eval mode if thats not desired.')
            for mod_name in mod_names:
                mod = getattr(model.module, mod_name)
                logging.warning('Training %s: %s', mod_name, mod)
                params_to_train.extend(mod.parameters())
        else:
            mod_names = []
            params_to_train = model.parameters()

        optimizer = hydra.utils.instantiate(opt_params, params_to_train)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               T_max=updates)
        logging.info('Starting actual training for %d updates', updates)

        last_checkpoint = get_latest_checkpoint(output_dir)
        batch_start = 0  # By default, starting from iteration 0, unles loading mdl
        if last_checkpoint is not None:
            logging.info('Going to load from %s', last_checkpoint)
            last_checkpoint = torch.load(last_checkpoint)
            model.load_state_dict(last_checkpoint['model'])
            optimizer.load_state_dict(last_checkpoint['optim'])
            # Subtracting 1 since we store batch_id + 1 when calling save_agent
            batch_start = last_checkpoint['done_batches'] - 1
            if scheduler is not None:
                scheduler.load_state_dict(last_checkpoint['scheduler'])

        def run_full_eval(batch_id):
            results = {}  # To store to a json
            eval_stats = full_eval_from_model(model)
            metric = eval_stats.compute_all_metrics()
            results['metrics'] = metric
            results[
                'metrics_rollout'] = eval_stats.compute_all_metrics_over_rollout(
                )
            results[
                'metrics_per_task'] = eval_stats.compute_all_metrics_per_task(
                )
            max_test_attempts_per_task = (cfg.max_test_attempts_per_task
                                          or phyre.MAX_TEST_ATTEMPTS)
            results['parsed_args'] = dict(
                # cfg=cfg,  # Not json serializable, anyway will be stored in dir
                main_kwargs=dict(
                    eval_setup_name=cfg.eval_setup_name,
                    fold_id=cfg.fold_id,
                    use_test_split=cfg.use_test_split,
                    agent_type=cfg.agent.type,
                    max_test_attempts_per_task=max_test_attempts_per_task,
                    output_dir=output_dir))
            results['target_metric'] = (
                results['metrics']['independent_solved_by_aucs']
                [max_test_attempts_per_task])
            results['target_metric_over_time'] = [
                el['independent_solved_by_aucs'][max_test_attempts_per_task]
                for el in results['metrics_rollout']
            ]
            logging.info('Iter %d: %s; Over rollout: %s', (batch_id + 1),
                         results['target_metric'],
                         results['target_metric_over_time'])

            score = metric['independent_solved_by_aucs'][-1]
            summary_writer.add_scalar('FullEval/AUCCESS', score, batch_id + 1)
            for solved_by_iter in metric['global_solved_by']:
                summary_writer.add_scalar(
                    'FullEval/solved_by_{}'.format(solved_by_iter),
                    metric['global_solved_by'][solved_by_iter], batch_id + 1)
            logging.info('Full eval perf @ %d: %s', batch_id + 1, score)
            for i, metric in enumerate(results['metrics_rollout']):
                summary_writer.add_scalar(
                    'FullEvalRollout/AUCCESS/{}'.format(i + 1),
                    metric['independent_solved_by_aucs'][-1], batch_id + 1)
                summary_writer.add_scalar(
                    'FullEvalRollout/solved_by_100/{}'.format(i),
                    metric['global_solved_by'][100], batch_id + 1)
            summary_writer.add_scalar('FullEvalRollout/prediction_accuracy/',
                                      metric['pred_acc'], batch_id + 1)
            respath = os.path.join(
                output_dir,
                'results_intermediate/{:08d}.json'.format(batch_id + 1))
            os.makedirs(os.path.dirname(respath), exist_ok=True)
            with open(respath, 'w') as fout:
                json.dump(results, fout)

        logging.info('Report every %d; full eval every %d', report_every,
                     full_eval_every)
        if save_checkpoints_every > full_eval_every:
            save_checkpoints_every -= save_checkpoints_every % full_eval_every

        losses_report = {}
        last_time = time.time()
        assert train_batch_size > 1 and train_batch_size % 2 == 0, (
            'Needs to get 2 elements at least to balance out')
        for batch_data_id, batch_data in enumerate(
                torch.utils.data.DataLoader(
                    dataset,
                    num_workers=im_fwd_agent.get_num_workers(
                        cfg.train.data_loader.num_workers,
                        dataset.frames_per_clip),
                    pin_memory=False,
                    # Asking for half the batch size since the dataloader is designed
                    # to give 2 elements per batch (for class balancing)
                    batch_size=train_batch_size // 2)):
            batch_id = batch_data_id + batch_start
            if (batch_id + 1) >= updates:
                im_fwd_agent.save_agent(output_dir, batch_id + 1, model,
                                        optimizer, scheduler)
                break
            model.train()
            batch_is_solved = batch_data['is_solved']
            batch_is_solved = batch_is_solved.to(device, non_blocking=True)
            batch_is_solved = batch_is_solved.reshape((-1, ))
            batch_obj_obs = batch_data['obj_obs']
            batch_obj_obs = batch_obj_obs.reshape(
                [-1] + list(batch_obj_obs.shape[2:]))
            batch_obj_obs = batch_obj_obs.to(device)

            # Run the forward classifcation model on the object frames
            train_noise_frac = 0.0
            if cfg.agent.train_with_noise:
                if (batch_id / updates) > cfg.agent.decay_noise_end:
                    train_noise_frac = 0.0
                elif (batch_id / updates) < cfg.agent.decay_noise_start:
                    train_noise_frac = cfg.agent.train_noise_percent
                else:
                    start_noise_decay = cfg.agent.decay_noise_start * updates
                    end_noise_decay = cfg.agent.decay_noise_end * updates
                    noise_decay_updates = end_noise_decay - start_noise_decay
                    train_noise_frac = cfg.agent.train_noise_percent * (
                        1 -
                        (batch_id - start_noise_decay) / noise_decay_updates)
            _, batch_losses = model.forward(
                batch_obj_obs,
                batch_is_solved,
                n_hist_frames=n_hist_frames,
                n_fwd_times=n_fwd_times,
                compute_losses=True,
                need_intermediate=False,  #loss_cfg.on_intermediate,
                nslices=num_slices,
                train_noise_frac=train_noise_frac,
                need_pixels=False)

            optimizer.zero_grad()
            total_loss = 0
            # Mean over each loss type from each replica
            for loss_type in batch_losses:
                loss_wt = getattr(loss_cfg, 'wt_' + loss_type)
                if loss_wt <= 0:
                    continue
                loss_val = loss_wt * torch.mean(batch_losses[loss_type], dim=0)
                if loss_type not in losses_report:
                    losses_report[loss_type] = []
                losses_report[loss_type].append(loss_val.item())
                total_loss += loss_val
            total_loss.backward()
            optimizer.step()
            if (save_checkpoints_every > 0
                    and (batch_id + 1) % save_checkpoints_every == 0):
                im_fwd_agent.save_agent(output_dir, batch_id + 1, model,
                                        optimizer, scheduler)

            if (batch_id + 1) % report_every == 0:
                speed = report_every / (time.time() - last_time)
                last_time = time.time()
                loss_stats = {
                    typ: np.mean(losses_report[typ][-report_every:])
                    for typ in losses_report if len(losses_report[typ]) > 0
                }
                logging.info(
                    'Iter: %s, examples: %d, mean loss: %s, speed: %.1f batch/sec,'
                    ' lr: %f', batch_id + 1, (batch_id + 1) * train_batch_size,
                    loss_stats, speed, im_fwd_agent.get_lr(optimizer))
                for typ in loss_stats:
                    summary_writer.add_scalar('Loss/{}'.format(typ),
                                              loss_stats[typ], batch_id + 1)
                summary_writer.add_scalar('Loss/Total',
                                          sum(loss_stats.values()),
                                          batch_id + 1)
                summary_writer.add_scalar('LR', im_fwd_agent.get_lr(optimizer),
                                          batch_id + 1)
                summary_writer.add_scalar('Speed', speed, batch_id + 1)
                # Add a histogram of the batch task IDs, to make sure it picks a
                # variety of task
                batch_templates = np.array(
                    dataset.task_ids)[batch_data['task_indices'].reshape(
                        (-1, ))].tolist()
                batch_templates = np.array(
                    [int(el.split(':')[0]) for el in batch_templates])
                gpu_mem_max = max([
                    torch.cuda.max_memory_allocated(device=i)
                    for i in range(torch.cuda.device_count())
                ])
                summary_writer.add_scalar('GPU/Mem/Max', gpu_mem_max,
                                          batch_id + 1)
                summary_writer.add_histogram('Templates',
                                             batch_templates,
                                             global_step=(batch_id + 1),
                                             bins=25)
            # Visualize a couple train videos, and actual rollouts if pix is
            # being trained
            # Just visualizing the first 256 videos in case the batch size is
            # larger; somehow the visualizations get corrupted (grey bg) for
            # more. Also no point filling up the memory.
            # Storing less frequently than the rest of the logs (takes lot of space)
            if n_fwd_times > 0 and (batch_id + 1) % (report_every * 10) == 0:
                batch_vid_obs = batch_data['vid_obs']
                batch_vid_obs = batch_vid_obs.reshape(
                    [-1] + list(batch_vid_obs.shape[2:]))
                batch_vid_obs = batch_vid_obs.to(device)
                vis_fwd_times = n_fwd_times if 'classification_model' in mod_names else None
                videos = gen_vis_vid_preds(batch_vid_obs[:256],
                                           batch_obj_obs[:256],
                                           model,
                                           n_fwd_times=vis_fwd_times,
                                           run_decode=True,
                                           n_hist_frames=n_hist_frames)
                summary_writer.add_video('InputAndRollout/train', videos,
                                         (batch_id + 1))
            if (batch_id + 1) % full_eval_every == 0:
                run_full_eval(batch_id)
            if scheduler is not None:
                scheduler.step()
        return model.cpu()
