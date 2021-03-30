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
from typing import Sequence, Tuple
import logging
import os
import time
import glob
import json
from tqdm import tqdm
from PIL import Image
from PIL import ImageDraw
import subprocess
import hydra
import imageio

import numpy as np
import torch
from torch import nn

import nets
import phyre

from neural_agent import get_latest_checkpoint

# For evaluation while training (small, to get a quick number)
AUCCESS_EVAL_TASKS = 200
XE_EVAL_SIZE = 10000

TaskIds = Sequence[str]
NeuralModel = nn.Module
TrainData = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, phyre.
                  ActionSimulator, torch.Tensor]


def save_agent(output_dir, batch_id, model, optimizer, scheduler, keep_last=3):
    """Store agent to disk."""
    fpath = os.path.join(output_dir, 'ckpt.%08d' % (batch_id))
    logging.info('Saving: %s', fpath)
    torch.save(
        dict(model=model.state_dict(),
             optim=optimizer.state_dict(),
             done_batches=batch_id,
             scheduler=scheduler and scheduler.state_dict()), fpath)
    # Delete the older ones, keep the last few
    all_ckpts = sorted(glob.glob(os.path.join(output_dir, 'ckpt.*')))
    to_del = all_ckpts[:-keep_last]
    for fpath in to_del:
        os.remove(fpath)


def convert_to_video_vis(vid, is_solved=None):
    """
    Generate a video visualization to go into tensorboard.
    Args:
        vid np.ndarray(BxTxHxW): Video in standard PHYRE style
        is_solved (int): Whether this video solves the task or not.
    Returns:
        vid_vis (BxTxHxWx3)
    """
    return np.stack([
        # The is_solved argument adds a little bar to the top for visualization
        # green if solved, red if not.
        np.stack([
            phyre.vis.observations_to_uint8_rgb(frame, is_solved=is_solved)
            for frame in clip
        ]) for clip in vid
    ])


def get_num_workers(num_workers, frames_per_clip):
    """Finetunes the nworkers if batch size/frames per clip too large, since
    otherwise jobs crash."""
    del frames_per_clip
    return num_workers


def gen_vis_vid_preds(orig_vid,
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

    with torch.no_grad():
        model.eval()
        all_preds, _ = model.forward(vid,
                                     None,
                                     n_hist_frames=n_hist_frames,
                                     n_fwd_times=n_fwd_times,
                                     compute_losses=False,
                                     need_intermediate=True,
                                     run_decode=run_decode,
                                     nslices=1)
    stacked, _, _, _ = ImgTrainer.vis_stacked_pred_gt(
        nets.combine_obj_pixels(orig_vid, 2).cpu().numpy(),
        nets.combine_obj_pixels(vid, 2),
        all_preds['pixels'] if run_decode else None)
    # For some reason need to flip the image in space and time for corr vis
    stacked_rgb = np.array(
        convert_to_video_vis(stacked).transpose((0, 1, 4, 2, 3)))
    return torch.as_tensor(stacked_rgb)


def get_lr(optimizer):
    """Read LR."""
    for param_group in optimizer.param_groups:
        return param_group['lr']


def phyre_batchvidresize(t, shape):
    """
    Args:
        t: Input video tensor batch, Long dtype, BxTxHxW
        shape: Output shape required, (H', W')
    """
    return nn.functional.interpolate(t.to(torch.float),
                                     size=list(shape),
                                     mode='nearest').to(torch.long)


def overlay_pred_scores(vid, scores, ch=1):
    """
    Args:
        vid (B, 1, H, W): PHYRE style video (torch.Tensor)
        scores (B,) Scores for each batch element to be overlayed in text on the
            frame
        ch: Which channel to overlay on.
    Returns:
        vid (B, H, W) with the score overlayed
    """
    overlays = []
    for batch_id in range(vid.shape[0]):
        img = Image.new('1', vid.shape[1:][::-1], 0)
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), '{:04f}'.format(scores[batch_id]), (1, ))
        overlays.append(np.array(img)[::-1, :])
    overlay = torch.LongTensor(np.stack(overlays)).to(vid.device)
    vid = vid * (overlay == 0) + (overlay > 0) * ch
    return vid


def compute_pixel_accuracy(gt, pred):
    """
    Args:
        gt torch.Tensor(B, T, H, W)
        pred torch.Tensor(B, T, H, W)
    Returns:
        acc torch.Tensor(B, phyre.NUM_COLORS)
    """
    match = (gt == pred)
    res = torch.zeros((
        gt.shape[0],
        phyre.NUM_COLORS,
    ))
    for col in range(phyre.NUM_COLORS):
        relevant = (gt == col)
        res[:, col] = torch.sum(match * relevant, dim=(1, 2, 3)) / torch.sum(
            relevant, dim=(1, 2, 3)).float()
    return res


def store_frames(frames, task_ids, outdir, subdir, actions):
    """
    Args:
        frames: (B, T, H, W)
        outdir (path where to store all frames)
        actions: (B, 3)
    """
    assert frames.shape[0] == len(task_ids)
    assert frames.shape[0] == actions.shape[0]
    for i, task_id in enumerate(task_ids):
        action = actions[i]
        action_str = '{:.5f}_{:.5f}_{:.5f}'.format(action[0], action[1],
                                                   action[2])
        template, _ = task_id.split(':')
        this_outdir = os.path.join(outdir, 'eval_vis', template,
                                   task_id + '_' + action_str, subdir)
        os.makedirs(this_outdir, exist_ok=True)
        all_rendered = []
        for time_step in range(frames[i].shape[0]):
            rendered = phyre.vis.observations_to_uint8_rgb(
                frames[i][time_step])
            # Storing individually was super slow!
            # Image.fromarray(rendered).save(
            #     os.path.join(this_outdir, '%d.png' % time_step))
            all_rendered.append(rendered)
        imageio.mimwrite(os.path.join(this_outdir, 'combined.gif'),
                         all_rendered,
                         fps=2)


class ImgTrainer(object):
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
        missing_keys, unexp_keys = model.load_state_dict(
            last_checkpoint['model'], strict=strict)
        logging.warning('Could not init: %s', missing_keys)
        logging.warning('Unused keys in ckpt: %s', unexp_keys)
        model.to(nets.DEVICE)
        return model

    @classmethod
    def gen_model(cls, cfg):
        """Generate the random init model."""
        model = nets.Fwd(agent_cfg=cfg.agent)
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
        loss_cfg = cfg.train.loss
        opt_params = cfg.opt
        # action_tier_name = cfg.tier
        n_fwd_times = cfg.train.n_fwd_times
        n_fwd_times_incur_loss = cfg.train.n_fwd_times_incur_loss
        run_decode = cfg.train.run_decode
        train_modules_subset = cfg.train.modules_to_train
        # nslices (slice out the input for training)
        num_slices = cfg.train.num_slices

        if max_frames_fwd is not None and (max_frames_fwd <= n_hist_frames):
            logging.warning(
                'Cant train prediction model, max_frames_fwd too low')
        assert loss_cfg.wt_pix == 0 or run_decode is True, (
            'If the loss is non zero, the decoder should be running')

        # logging.info('Creating eval subset from train')
        # eval_train = create_balanced_eval_set(cache, dataset.task_ids,
        #                                       XE_EVAL_SIZE, action_tier_name)
        # if dev_tasks_ids is not None:
        #     logging.info('Creating eval subset from dev')
        #     eval_dev = create_balanced_eval_set(cache, dev_tasks_ids, XE_EVAL_SIZE,
        #                                         action_tier_name)
        # else:
        #     eval_dev = None

        device = nets.DEVICE
        model.train()
        model.to(device)
        logging.info("%s", model)

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
            logging.info('Running full eval')
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
                    num_workers=get_num_workers(
                        cfg.train.data_loader.num_workers,
                        dataset.frames_per_clip),
                    pin_memory=False,
                    # Asking for half the batch size since the dataloader is designed
                    # to give 2 elements per batch (for class balancing)
                    batch_size=train_batch_size // 2)):
            # When the training restarts, it resets to the start of the data loader
            batch_id = batch_data_id + batch_start
            if (batch_id + 1) >= updates:
                save_agent(output_dir, batch_id + 1, model, optimizer,
                           scheduler)
                break
            model.train()
            batch_is_solved = batch_data['is_solved']
            batch_is_solved = batch_is_solved.to(device, non_blocking=True)
            batch_is_solved = batch_is_solved.reshape((-1, ))
            batch_vid_obs = batch_data['vid_obs']
            batch_vid_obs = batch_vid_obs.reshape(
                [-1] + list(batch_vid_obs.shape[2:]))
            batch_vid_obs = batch_vid_obs.to(device)

            # Run the forward image model on the video
            _, batch_losses = model.forward(
                batch_vid_obs,
                batch_is_solved,
                n_hist_frames=n_hist_frames,
                n_fwd_times=n_fwd_times,
                n_fwd_times_incur_loss=n_fwd_times_incur_loss,
                run_decode=run_decode,
                compute_losses=True,
                need_intermediate=loss_cfg.on_intermediate,
                autoenc_loss_ratio=loss_cfg.autoenc_loss_ratio,
                nslices=num_slices)

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
                save_agent(output_dir, batch_id + 1, model, optimizer,
                           scheduler)
            # Removing intermediate eval since it doesnt seem very useful, using the
            # full eval for now.
            # if (batch_id + 1) % eval_every == 0:
            #     print_eval_stats(batch_id)
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
                    loss_stats, speed, get_lr(optimizer))
                for typ in loss_stats:
                    summary_writer.add_scalar('Loss/{}'.format(typ),
                                              loss_stats[typ], batch_id + 1)
                summary_writer.add_scalar('Loss/Total',
                                          sum(loss_stats.values()),
                                          batch_id + 1)
                summary_writer.add_scalar('LR', get_lr(optimizer),
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
                summary_writer.add_video(
                    'InputAndRollout/train',
                    gen_vis_vid_preds(batch_vid_obs[:256],
                                      model,
                                      n_fwd_times=None,
                                      run_decode=run_decode,
                                      n_hist_frames=n_hist_frames),
                    (batch_id + 1))
            if (batch_id + 1) % full_eval_every == 0:
                run_full_eval(batch_id)
            if scheduler is not None:
                scheduler.step()
        return model.cpu()

    @classmethod
    def eval_actions(cls, model, dataset, nactionsXtasks, batch_size, cfg):
        """Evaluate likelihood of actions solving the task."""
        init_frames_to_sim = cfg.eval.init_frames_to_sim  # Run it for these many
        n_hist_frames = cfg.eval.n_hist_frames
        n_fwd_times = cfg.eval.n_fwd_times
        store_vis = cfg.eval.store_vis
        train_run_decode = cfg.train.run_decode

        assert init_frames_to_sim >= n_hist_frames, 'Need those many to start pred'

        def pad_tensor(tensor: torch.Tensor, sz: int):
            """
            Pad the tensor's bottom (along batch dim), using the last element,
            sz times.
            """
            bottom_tensor_rep = [tensor[-1:, ...]] * sz
            return torch.cat([tensor] + bottom_tensor_rep, dim=0)

        def unpad_tensor(tensor: torch.Tensor, sz: int):
            return tensor[:tensor.shape[0] - sz, ...]

        # Clear the directory of older vis if any
        out_dir = 'vis/'
        if store_vis:
            logging.warning('Removing older vis from %s/%s', os.getcwd(),
                            out_dir)
            subprocess.call(f'rm {out_dir}/*.gif', shell=True)
            subprocess.call(f'rm -r {out_dir}/eval_vis', shell=True)

        scores = []
        actions = []
        task_indices = []
        pixel_accs = []  # If generating output, how accurately did we do
        with torch.no_grad():
            model.eval()
            for batch_id, batch_data in enumerate(
                    tqdm(torch.utils.data.DataLoader(
                        dataset,
                        num_workers=get_num_workers(
                            cfg.eval.data_loader.num_workers,
                            dataset.frames_per_clip),
                        pin_memory=False,
                        batch_size=batch_size,
                        drop_last=False),
                         desc='All tasks X actions batches',
                         total=nactionsXtasks // batch_size)):
                batch_task_indices = batch_data['task_indices']
                batch_vid_obs = batch_data[
                    f'{cfg.agent.input_space}_obs'].squeeze(1)
                batch_vid_obs_orig = batch_data[
                    f'{cfg.agent.input_space}_obs_orig'].squeeze(1)
                batch_actions = batch_data['actions']
                # Since the code might be run with DataParallel, need to make sure
                # the batch size is divisible by the ngpus, so stick to the
                # requested batch size by padding actions.
                uniq_batch_size = batch_actions.shape[0]
                pad_len = max(batch_size - uniq_batch_size, 0)
                batch_vid_obs = pad_tensor(batch_vid_obs, pad_len)

                # Setting run_decode always true when true in training..
                # (earlier only when visualizing)
                # Sometimes evaluation might depend on the decoded frame, so might
                # as well...
                other_kwargs = {
                    'need_intermediate': True,
                    'run_decode': train_run_decode,
                    'nslices': 1
                }
                all_preds, batch_losses = model.forward(
                    batch_vid_obs,
                    None,
                    n_hist_frames=n_hist_frames,
                    n_fwd_times=n_fwd_times,
                    compute_losses=False,
                    **other_kwargs)

                # Unpad parts of all_preds that will be used further
                # Since the model is trained with BCELoss, normalize using sigmoid
                # On 2020/02/11, I changed it to return only one prediction for
                # any n_fwd_times (max-pool all to give 1 prediction), hence this
                # list will only contain a single element. To stay consistent with
                # prior code that expects a prediction at each time step, simply
                # repeating that prediction n_fwd_times.

                batch_scores = nn.Sigmoid()(unpad_tensor(
                    all_preds['is_solved'], pad_len))
                batch_vid_obs = unpad_tensor(batch_vid_obs, pad_len)

                if store_vis:
                    # Sum the vid obs over the channels, in case it was split into
                    # components
                    if cfg.agent.input_space == 'obj':
                        # update to videos, for storing vis
                        batch_vid_obs = batch_data['vid_obs'].squeeze(1)
                        batch_vid_obs_orig = batch_data[
                            'vid_obs_orig'].squeeze(1)
                        batch_vid_obs = pad_tensor(batch_vid_obs, pad_len)
                        batch_vid_obs = unpad_tensor(batch_vid_obs, pad_len)

                    task_ids = batch_data['task_ids']
                    _, pixel_acc, gt_frames, pred_frames = cls.vis_stacked_pred_gt(
                        torch.sum(batch_vid_obs_orig, axis=-3).cpu().numpy(),
                        torch.sum(batch_vid_obs, dim=-3), [
                            unpad_tensor(el.cpu(), pad_len)
                            for el in all_preds['pixels']
                        ])
                    '''
                        [batch_scores] * len(all_preds['pixels']),
                        # Could take any batch_task_indices, all are same
                        '{}/{:04d}_{:04d}.gif'.format(out_dir,
                                                      batch_task_indices[0],
                                                      batch_id))
                    '''
                    # Also store pure frames individually, will be used for rollout
                    # accuracy evaluation
                    store_frames(gt_frames, task_ids, out_dir, 'gt',
                                 batch_actions)
                    store_frames(pred_frames, task_ids, out_dir, 'predictions',
                                 batch_actions)
                else:
                    pixel_acc = torch.zeros(
                        (batch_scores.shape[0], phyre.NUM_COLORS))
                assert len(batch_scores) == len(batch_actions), (
                    batch_actions.shape, batch_scores.shape)
                # IMP: Don't convert to cpu() numpy() here.. it makes the function
                # much slower. Convert in one go at the end when returning
                scores.append(batch_scores)
                pixel_accs.append(pixel_acc)
                actions.append(batch_actions)
                task_indices.append(batch_task_indices)
        # There is only 1 element in scores, but unsqueezing so that
        # it's compatible with following code that expects a score prediction
        # over time. Here it will give 1 prediction, the final one.
        final_scores = torch.cat(scores, dim=0).unsqueeze(0).cpu().numpy()
        final_actions = torch.cat(actions, dim=0).cpu().numpy()
        final_task_indices = torch.cat(task_indices, dim=0).cpu().numpy()
        final_pixel_accs = torch.cat(pixel_accs, dim=0).cpu().numpy()
        if nactionsXtasks != len(final_actions):
            logging.warning('Only evaluated %d actions instead of full %d',
                            len(final_actions), nactionsXtasks)
            assert (nactionsXtasks - len(actions)) <= batch_size, (
                'Shouldnt miss more')
        return final_scores, final_actions, final_task_indices, final_pixel_accs

    @classmethod
    def vis_stacked_pred_gt(cls,
                            orig_vid_full,
                            orig_vid,
                            pred_vid_qnt,
                            pred_solved=None,
                            store_path=None):
        """
        Args:
            orig_vid_full: list of videos [T'x256x256] for each batch element in
                orig_vid, for even the frames that are going to be predicted
            orig_vid (BxTx256x256)
            pred_vid_qnt [(BxHxW)] (or None, if not available) (unprocessed; i.e.
                argmaxed from 1-hot if need be, done already)
            pred_solved: [(B,)] list of is_solved scores from the model. Or can be
                None
            store_path (str): Path to store the video. None if not store.
        Returns:
            (B, T, H, W) Combined output
            (B, phyre.NUM_COLORS) pixel accuracy of the generated video
        """
        if pred_vid_qnt is None:
            return (orig_vid.cpu().numpy(),
                    np.zeros(
                        (orig_vid.shape[0], phyre.NUM_COLORS)), None, None)
        assert len(orig_vid_full) == orig_vid.shape[0]
        # Prepare full GT predictions to go below each clip
        orig_vid_full_padded = []
        all_t = min(orig_vid_full[0].shape[0], len(pred_vid_qnt))
        for vid in orig_vid_full:
            if vid.shape[0] >= all_t:
                orig_vid_full_padded.append(vid.astype(np.long))
            else:
                # Pad the videos with white frames if that frame is not returned
                raise NotImplementedError('This should not happen')
        gt_video = phyre_batchvidresize(
            torch.stack([torch.as_tensor(el) for el in orig_vid_full_padded]),
            pred_vid_qnt[0].shape[1:]).cpu()
        gt_video = gt_video.numpy()
        # Convert the gt clip to same size as predictions, add temporal dim
        orig_vid = phyre_batchvidresize(orig_vid, pred_vid_qnt[0].shape[1:])

        frames_quantized = torch.cat([orig_vid] +
                                     [el.unsqueeze(1) for el in pred_vid_qnt],
                                     dim=1).cpu().numpy()
        # Pad the video with empty frames to match the size of predicted videos
        padder = np.tile(
            np.zeros_like(gt_video[:, -1:]),
            (1, abs(frames_quantized.shape[1] - gt_video.shape[1]), 1, 1))
        gt_video_padded = gt_video
        frames_quantized_padded = frames_quantized
        if gt_video.shape[1] > frames_quantized.shape[1]:
            frames_quantized_padded = np.concatenate(
                [frames_quantized, padder], axis=1)
        else:
            gt_video_padded = np.concatenate([gt_video, padder], axis=1)
        # Compute the accuracy between the generated frames, and the GT frames
        # Only do for generated frames (so taking the last few ones)
        # If few GT frames are given (eg, when just training classifier), it will
        # be comparing to empty frames and get low score but that is okay, we don't
        # care about the pixel numbers at that point anyway
        # Update April 2 2020: This is using frames with numbers overlayed etc... so
        # deprecating this eval.
        pix_acc = compute_pixel_accuracy(
            torch.as_tensor(gt_video_padded[:, -len(pred_vid_qnt):, ...]),
            torch.as_tensor(
                frames_quantized_padded[:, -len(pred_vid_qnt):, ...]))
        # Stack them on the height axis
        final_vid = np.concatenate([gt_video_padded, frames_quantized_padded],
                                   axis=-2)
        if store_path:
            os.makedirs(os.path.dirname(store_path), exist_ok=True)
            phyre.vis.save_observation_series_to_gif(
                [frames_quantized_padded, gt_video_padded],
                store_path,
                # Piggy-backing on the solved_state markers to show which parts are
                # GT and which parts are being predicted
                solved_states=([True] * orig_vid.shape[1] +
                               [False] * final_vid.shape[1]),
                solved_wrt_step=True,
                fps=2)
        return final_vid, pix_acc, gt_video, frames_quantized