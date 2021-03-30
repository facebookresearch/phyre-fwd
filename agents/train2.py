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
"""General train/eval loop for a single agent on a single train/test split.

The script:
  * Finds all knows agents - all subclasses for offline_agents.Agent in the
    included files.
  * Loads train/dev or train/test task split for the specified seed and tier.
    By default a dev split is used. Set --use-test-split=1 got get get the
    final, (train + dev)/test split.
  * Initializes the agent from the commandline flags.
  * Trains the agent on the train part.
  * Evaluates the agents on eval part.
  * Saves the evalution results to `output_dir`/results.json. The file will
    contain a dictionary with all evaluation metrics. The most important one,
    AUCCESS@100 is saved with key "target_metric".


See offline_agents for example agents.
Modified from train.py to work with hydra CLI. Only supports the fwd agent now.
"""
from typing import Tuple
import json
import logging
import os
import sys
from functools import partial
import subprocess
import hydra

from torch.utils.tensorboard import SummaryWriter
import phyre
import offline_agents

def get_train_test(eval_setup_name: str, fold_id: int, use_test_split: bool
                   ) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    train, dev, test = phyre.get_fold(eval_setup_name, fold_id)
    if use_test_split:
        return train + dev, test
    else:
        return train, dev

def find_all_agents():
    def yield_subclsses(base):
        for cls in base.__subclasses__():
            if not cls.__abstractmethods__:
                yield cls
            yield from yield_subclsses(cls)

    return {cls.name(): cls for cls in yield_subclsses(offline_agents.Agent)}

def subsample_for_vis(eval_task_ids, tasks_per_template):
    """Keep only 1 task per template."""
    templates_sel = {}
    res = []
    eval_task_ids = sorted(eval_task_ids)  # For repro
    for task_id in eval_task_ids:
        this_temp = task_id.split(':')[0]
        if this_temp not in templates_sel:
            templates_sel[this_temp] = 1
            res.append(task_id)
        elif templates_sel[this_temp] < tasks_per_template:
            templates_sel[this_temp] += 1
            res.append(task_id)
    return res


def get_subset_tasks(all_tasks, ratio):
    """Select a subset of tasks from all_tasks, keeping some % from each temp.
    Args:
        all_tasks: List of tasks like ['00001:001', ...]
        ratio: A number between 0-1, specifying how many of the tasks to keep.
    Returns:
        tasks: An updated list, with the ratio number of tasks.
    """
    if len(all_tasks) == 0:
        return all_tasks
    assert 0.0 <= ratio <= 1.0
    all_tasks = sorted(all_tasks)
    samples_to_keep = int(len(all_tasks) * ratio)
    return all_tasks[::(len(all_tasks) // samples_to_keep)][:samples_to_keep]


@hydra.main(config_path='conf/config.yaml')
def main(cfg):
    """Run the training and testing."""
    # Make a copy of overrides/etc files; so that if this code is run
    # again with a different override param (eg to generate vis etc), even if
    # it overwrites the config files and destroy that information, the original
    # info is stored and avlbl when making graphs etc
    if not os.path.exists('.hydra.orig'):
        subprocess.call('cp -r .hydra .hydra.orig', shell=True)
    templates_tasks = None
    if ':' in cfg.eval_setup_name:
        # Means that we only want template IDs defined after the ":"
        # The tasks itself would have "00001:<task_id>", hence splitting only 1
        cfg.eval_setup_name, templates_tasks = cfg.eval_setup_name.split(
            ':', 1)
    train_task_ids, eval_task_ids = get_train_test(cfg.eval_setup_name,
                                                   cfg.fold_id,
                                                   cfg.use_test_split)
    if templates_tasks is not None:
        # Subselect the train/eval task ids to only keep the ones in task_ids
        templates_tasks = templates_tasks.split(';')
        final_templates = []
        for temp_task in templates_tasks:
            if ':' in temp_task:
                temp, task = temp_task.split(':')
            else:
                temp = temp_task
                task = ''
            if '-' in temp_task:
                final_templates += [
                    '{:05d}:{}'.format(el, task)
                    for el in range(int(temp.split('-')[0]),
                                    int(temp.split('-')[1]) + 1)
                ]
            else:
                final_templates += ['{:05d}:{}'.format(int(temp), task)]
        templates_tasks = sorted(list(set(final_templates)))
        logging.info('Running on %s templates/tasks', templates_tasks)

        def fits_templates_tasks(task_id):
            for temp_task in templates_tasks:
                if task_id.startswith(temp_task):
                    return True
            return False

        train_task_ids = [
            el for el in train_task_ids if fits_templates_tasks(el)
        ]
        eval_task_ids = [
            el for el in eval_task_ids if fits_templates_tasks(el)
        ]
        assert len(train_task_ids) > 0 or len(eval_task_ids) > 0, (
            'At least one of train or test should have a task in it')
    train_task_ids = sorted(train_task_ids)
    eval_task_ids = sorted(eval_task_ids)
    logging.info('Final train task ids: %s', train_task_ids)
    logging.info('Final eval task ids: %s', eval_task_ids)
    assert 0.0 <= cfg.data_ratio_train <= 1.0, 'Should be within limits'
    assert 0.0 <= cfg.data_ratio_eval <= 1.0, 'Should be within limits'
    train_task_ids = get_subset_tasks(train_task_ids, cfg.data_ratio_train)
    eval_task_ids = get_subset_tasks(eval_task_ids, cfg.data_ratio_eval)
    assert cfg.tier is None, (
        'Do not set this beforehand; will figure from eval_setup')
    cfg.tier = phyre.eval_setup_to_action_tier(cfg.eval_setup_name)
    agent = find_all_agents()[cfg.agent.type]
    output_dir = os.getcwd()
    max_test_attempts_per_task = (cfg.max_test_attempts_per_task
                                  or phyre.MAX_TEST_ATTEMPTS)

    # Validate the config
    # If the following are not true, it gives weird errors, eg missing argument
    # in forward
    assert cfg.num_gpus == 0 or cfg.train.batch_size % cfg.num_gpus == 0
    if cfg.eval.batch_size is not None:
        assert cfg.num_gpus == 0 or cfg.eval.batch_size % cfg.num_gpus == 0

    # Scale the number of iters
    if cfg.train.scale_num_iter != 1.0:
        for param_name in [
                'num_iter', 'report_every', 'save_checkpoints_every',
                'full_eval_every'
        ]:
            logging.info(f'cfg.train.scale_num_iter {cfg.train.scale_num_iter}')
            logging.info(f'param_name {param_name}')
            old_val = getattr(cfg.train, param_name)
            logging.info(f'old_val {old_val}')
            new_val = type(old_val)(old_val * cfg.train.scale_num_iter)
            setattr(cfg.train, param_name, new_val)
            logging.warning('Setting cfg.train.%s to %s using scale %f',
                            param_name, new_val, cfg.train.scale_num_iter)

    # It's fine to use eval_task_ids iff it's dev.
    dev_tasks_ids = None if cfg.use_test_split else eval_task_ids
    summary_writer = SummaryWriter(log_dir=os.path.join(output_dir, 'logs'))

    full_eval_fn = partial(agent.eval,
                           task_ids=eval_task_ids,
                           max_attempts_per_task=max_test_attempts_per_task,
                           cfg=cfg)

    logging.info('Starting training')
    state = agent.train(train_task_ids,
                        dev_tasks_ids,
                        full_eval_fn,
                        output_dir=output_dir,
                        summary_writer=summary_writer,
                        cfg=cfg)
    ## Evaluation
    out_path = os.path.join(
        output_dir,
        'results-vis.json' if cfg.eval.store_vis else 'results.json')
    # Don't stop re-evaluations if doing vis
    if (os.path.exists(out_path) and not cfg.force_eval
            and not cfg.eval.store_vis):
        logging.warning('Eval out path exists (%s). Del or no eval.', out_path)
        return 0
    # Moved all of this to train, so the final prediction would be stored
    # in results_intermediate as well. However keeping the code here too since
    # it's used when only running testing.
    logging.info('Starting final eval')
    evaluation = full_eval_fn(state)

    num_tasks = len(eval_task_ids)
    results = {}
    results['num_eval_tasks'] = num_tasks
    results['metrics'] = evaluation.compute_all_metrics()
    results['metrics_rollout'] = evaluation.compute_all_metrics_over_rollout()
    results['metrics_per_task'] = evaluation.compute_all_metrics_per_task()
    results['args'] = sys.argv
    results['parsed_args'] = dict(
        # cfg=cfg,  # Not json serializable, anyway will be stored in dir
        main_kwargs=dict(eval_setup_name=cfg.eval_setup_name,
                         fold_id=cfg.fold_id,
                         use_test_split=cfg.use_test_split,
                         agent_type=cfg.agent.type,
                         max_test_attempts_per_task=max_test_attempts_per_task,
                         output_dir=output_dir))
    print(results['parsed_args'])
    results['target_metric'] = (
        results['metrics']['independent_solved_by_aucs']
        [max_test_attempts_per_task])
    results['target_metric_over_time'] = [
        el['independent_solved_by_aucs'][max_test_attempts_per_task]
        for el in results['metrics_rollout']
    ]
    logging.info('FINAL: %s; Over rollout: %s', results['target_metric'],
                 results['target_metric_over_time'])
    summary_writer.add_scalar('AUCCESS-full/eval', results['target_metric'])
    summary_writer.close()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(out_path, 'w') as stream:
        json.dump(results, stream)


if __name__ == '__main__':
    logging.basicConfig(format=('%(asctime)s %(levelname)-8s'
                                ' {%(module)s:%(lineno)d} %(message)s'),
                        level=logging.DEBUG,
                        datefmt='%Y-%m-%d %H:%M:%S')
    main()  # pylint: disable=no-value-for-parameter  # Uses hydra
