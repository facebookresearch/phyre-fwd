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
"""A wrapper around phyre.Evaluator, to deal with cases of multiple step
predictions."""
import logging
import torch
import phyre


class EvaluatorWrapper:
    """A wrapper over phyre.Evaluator, with multi-step predictions."""
    def __init__(self, simulator, task_ids, nsteps, max_attempts_per_task):
        self.simulator = simulator
        self.task_ids = task_ids
        self.evaluators = [phyre.Evaluator(task_ids) for _ in range(nsteps)]
        self.max_attempts_per_task = max_attempts_per_task

    def wrapper_add_scores(self, task_index, scores, actions):
        """
        Args:
            scores (nsteps, num_actions)
        """
        assert len(self.evaluators) == scores.shape[0]
        if torch.is_tensor(actions):
            actions = actions.cpu().numpy()
        for i, evaluator in enumerate(self.evaluators):
            scores_t = scores[i].tolist()
            sorted_scores_t, sorted_actions_t = zip(*sorted(
                zip(scores_t, actions), key=lambda x: (-x[0], tuple(x[1]))))
            for scr, act in zip(sorted_scores_t, sorted_actions_t):
                if evaluator.get_attempts_for_task(
                        task_index) >= self.max_attempts_per_task:
                    break
                status = self.simulator.simulate_action(
                    task_index, act, need_images=False).status
                # Also logging action (is optional), so can be stored
                evaluator.maybe_log_attempt(task_index,
                                            status,
                                            scr,
                                            action=act)
            if evaluator.get_attempts_for_task(task_index) == 0:
                logging.warning('Made 0 attempts for task %s',
                                self.task_ids[task_index])

    def compute_all_metrics_over_rollout(self):
        return [ev.compute_all_metrics() for ev in self.evaluators]

    def __getattr__(self, name):
        """To respond to any undefined function call, just use the last
        evaluator by default (corr to full rollout)."""
        # Limiting the function names to avoid accidental access to any other
        # functions that might modify the state of the evaluator. That must
        # be done through the wrapper_add_scores function.
        assert name in [
            'get_aucess', 'compute_all_metrics', 'compute_all_metrics_per_task'
        ]
        return getattr(self.evaluators[-1], name)
