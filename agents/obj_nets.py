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
"""This library contains object based forward and classification models."""
import math
import glob
import os
import logging
import functools
import torch.multiprocessing as multiprocessing
import hydra
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

import phyre
import phyre_simulator
import neural_agent
import nets
import preproc


def render_rollout(rollout, i):
    pix_rollout = []
    for j in range(rollout.shape[1]):
        row_sum = torch.sum(rollout[i][j], dim=-1)
        is_pad = row_sum == 0
        r_frame = rollout[i][j][~is_pad]
        img = phyre.objects_util.featurized_objects_vector_to_raster(r_frame)
        pix_rollout.append(img)
    return i, torch.LongTensor(pix_rollout)


class ForwardBaseModel(nn.Module):
    """
    Base class for forward models in object space, to hold shared logic.
    """
    @property
    def device(self):
        if hasattr(self, 'parameters') and next(self.parameters()).is_cuda:
            return 'cuda'
        else:
            return 'cpu'


class FwdPositionalEncoding(nn.Module):

    def __init__(self,
                 input_emb=phyre.FeaturizedObjects._NUM_FEATURES,
                 num_inp=phyre_simulator.MAX_NUM_OBJECTS,
                 dropout=0.1,
                 max_len=20):
        super(FwdPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, input_emb)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, input_emb, 2).float() *
            (-math.log(10000.0) / input_emb))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        #pe = pe.expand(-1, num_inp, -1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :, :]
        return self.dropout(x)


class TransformerForwardObjectNetwork(ForwardBaseModel):
    def __init__(self,
                 hist_frames,
                 only_pred_dynamic=False,
                 dont_predict_ball_theta=False,
                 predict_residuals=False,
                 clip_output=False):
        super(TransformerForwardObjectNetwork, self).__init__()
        self.dont_predict_ball_theta = dont_predict_ball_theta
        self.only_pred_dynamic = only_pred_dynamic
        self.clip_output = None if not clip_output else torch.nn.Hardtanh(0, 1)
        self.predict_residuals = predict_residuals

        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.hist_frames = hist_frames
        self.embedding_size = 128
        self.encoder = nn.Sequential(
            nn.Linear(phyre.FeaturizedObjects._NUM_FEATURES,
                      self.embedding_size),
            nn.ReLU(),
            nn.Linear(self.embedding_size, self.embedding_size),
        )
        # num_inp=phyre_simulator.MAX_NUM_OBJECTS,
        self.pos_encoder = FwdPositionalEncoding(
            input_emb=self.embedding_size,
            max_len=self.hist_frames)
        encoder_layers = TransformerEncoderLayer(self.embedding_size, nhead=8)
        self.transformer_encoder = TransformerEncoder(encoder_layers,
                                                      num_layers=6)
        self.aggregate_hidden_size = 100

        self.output_size = 3  # x, y, theta
        self.predict_states = nn.Sequential(
            nn.Linear(self.embedding_size, self.aggregate_hidden_size),
            nn.ReLU(),
            nn.Linear(self.aggregate_hidden_size, self.aggregate_hidden_size),
            nn.ReLU(),
            nn.Linear(self.aggregate_hidden_size, self.output_size),
        )

    def forward(self, tensor):
        tensor = tensor.clone()
        row_sum = torch.sum(tensor, dim=-1)
        is_pad = row_sum == 0

        # tensor is B x T x N x F
        num_objects = tensor.shape[2]
        # fixed feats is B x 1 x N x F-3
        fixed_feats = tensor.select(1, 0).narrow(
            2, 3, phyre.FeaturizedObjects._NUM_FEATURES - 3)
        # fixed pos is B x 1 x N x 3
        fixed_pos = tensor.select(1, -1).narrow(2, 0, 3)

        tensor = tensor.clone()
        # tensor is B x T x N x F
        tensor = self.encoder(tensor)
        # tensor is B x T x N x 128
        tensor = self.pos_encoder(tensor) * math.sqrt(self.embedding_size)
        # tensor is B x T x N x 128
        tensor = torch.flatten(tensor, start_dim=1, end_dim=2)
        mask = torch.flatten(is_pad, start_dim=1, end_dim=2)
        # tensor is B x (T x N) x 128
        tensor = tensor.permute(1, 0, 2)
        # tensor is (T x N) x B x 128
        tensor = self.transformer_encoder(tensor, src_key_padding_mask=mask)
        # tensor is (T x N) x B x 128
        tensor = tensor.permute(1, 0, 2)
        # tensor is  B x (T x N) x 128
        # only pass embedding of T-1 to mlp
        tensor = tensor[:, -1 * num_objects:]
        # tensor is B x N x 128

        states = self.predict_states(tensor)
        # states is B x N x 3
        if self.predict_residuals:
            states = states + fixed_pos
        if self.clip_output is not None:
            states = self.clip_output(states.clone()).clone()
        if self.only_pred_dynamic:
            black = fixed_feats[:, :, -1] == 1.
            purple = fixed_feats[:, :, -3] == 1.
            static_indicies = black + purple
            states[static_indicies] = fixed_pos[static_indicies]
        output = torch.cat((states, fixed_feats), 2)
        return output


class InteractionNetwork(ForwardBaseModel):
    DS = phyre.FeaturizedObjects._NUM_FEATURES + 3  # VECTOR_LENGTH + 3
    DR = 3
    D_DYN = 2 * DR  # x,y, theta, vx, vy, vtheta
    DX = 1

    def __init__(self,
                 hist_frames=2,
                 only_pred_dynamic=False,
                 clip_output=False,
                 dont_predict_ball_theta_v=False,
                 position_invariant=False):
        super().__init__()
        self.dont_predict_ball_theta_v = dont_predict_ball_theta_v
        self.only_pred_dynamic = only_pred_dynamic
        self.clip_output = None if not clip_output else torch.nn.Hardtanh(0, 1)

        self.position_invariant = position_invariant

        self.output_size = 3  #vx, vy, vtheta

        assert hist_frames == 2, 'IN only for two historical frames'
        self.hist_frames = hist_frames  #(to calculate vx, vy, vtheta)

        self.relation_matricies = {}
        self.external_effects_matricies = {}

        self.interactions_size = 50  #?
        self.interactions_hidden_size = 150
        self.states_hidden_size = 100
        self.states_output_size = 3

        interactions_input = self.D_DYN + (
            2 *
            (self.DS - self.D_DYN)) + self.DR if self.position_invariant else (
                2 * self.DS) + self.DR
        self.interactions_model = nn.Sequential(
            nn.Linear(interactions_input, self.interactions_hidden_size),
            nn.ReLU(),
            nn.Linear(self.interactions_hidden_size,
                      self.interactions_hidden_size), nn.ReLU(),
            nn.Linear(self.interactions_hidden_size,
                      self.interactions_hidden_size), nn.ReLU(),
            nn.Linear(self.interactions_hidden_size,
                      self.interactions_hidden_size), nn.ReLU(),
            nn.Linear(self.interactions_hidden_size, self.interactions_size))
        states_input = self.interactions_size + self.DS + self.DX - 3 if self.position_invariant else self.interactions_size + self.DS + self.DX
        self.states_model = nn.Sequential(
            nn.Linear(states_input, self.states_hidden_size),
            nn.ReLU(),
            nn.Linear(self.states_hidden_size, self.states_output_size),
        )

    def _intialize_external_effects_matrix(self, tensor):
        # External effect of gravity only effects dynamic objects
        black = tensor[:, -1] == 1.
        purple = tensor[:, -3] == 1.
        row_sum = torch.sum(tensor, dim=-1)
        pad = row_sum == 0
        static_indicies = black + purple + pad
        external_effects_matrix = torch.ones(
            (tensor.shape[0], 1)).to(tensor.device)
        external_effects_matrix[static_indicies] = 0.0

        external_effects_matrix = external_effects_matrix.permute(1, 0)
        return external_effects_matrix

    def _initialize_relations_matrix_for_relation(
            self, one_hot_objects, sender_indices, relation_index, pad_start,
            sender_matrix, receiver_matrix, relation_matrix):
        all_senders = None
        all_receivers = None
        for index in sender_indices:
            senders = torch.cat((one_hot_objects[:index],
                                 one_hot_objects[index + 1:pad_start]))
            receivers = one_hot_objects[index].clone().expand(
                senders.shape[0], -1)
            if all_senders is None:
                all_senders = senders
                all_receivers = receivers
            else:
                all_senders = torch.cat((all_senders, senders))
                all_receivers = torch.cat((all_receivers, receivers))
        if all_senders is not None:
            all_senders = all_senders.permute(1, 0)
            all_receivers = all_receivers.permute(1, 0)
            current_relation_matrix = torch.zeros(
                (self.DR, all_receivers.shape[1])).to(one_hot_objects.device)
            current_relation_matrix[relation_index, :] = 1
            if sender_matrix is None:
                sender_matrix = all_senders
            else:
                sender_matrix = torch.cat((sender_matrix, all_senders),
                                          axis=-1)
            if receiver_matrix is None:
                receiver_matrix = all_receivers
            else:
                receiver_matrix = torch.cat((receiver_matrix, all_receivers),
                                            axis=-1)
            if relation_matrix is None:
                relation_matrix = current_relation_matrix
            else:
                relation_matrix = torch.cat(
                    (relation_matrix, current_relation_matrix), axis=-1)
        return sender_matrix, receiver_matrix, relation_matrix

    def _initialize_relations_matricies(self, tensor):
        tensor = tensor.clone()

        one_hot_objects = torch.eye(tensor.shape[0]).to(tensor.device)
        black = tensor[:, -1] == 1.
        purple = tensor[:, -3] == 1.
        row_sum = torch.sum(tensor, dim=-1)
        pad = row_sum == 0

        pad_indicies = torch.nonzero(pad, as_tuple=True)[0]
        if len(pad_indicies) > 0:
            pad_start = min(pad_indicies)
        else:
            pad_start = tensor.shape[0]
        is_static = black + purple
        non_dyn_indicies = is_static + pad
        dyn_indicies = torch.nonzero(~non_dyn_indicies, as_tuple=True)[0]
        static_indicies = torch.nonzero(is_static, as_tuple=True)[0]

        all_senders = None
        all_receivers = None

        sender_matrix, receiver_matrix, relation_matrix = self._initialize_relations_matrix_for_relation(
            one_hot_objects, dyn_indicies, 0, pad_start, None, None, None)

        sender_matrix, receiver_matrix, relation_matrix = self._initialize_relations_matrix_for_relation(
            one_hot_objects, static_indicies, 1, pad_start, sender_matrix,
            receiver_matrix, relation_matrix)

        sender_matrix, receiver_matrix, relation_matrix = self._initialize_relations_matrix_for_relation(
            one_hot_objects, pad_indicies, 2, tensor.shape[0], sender_matrix,
            receiver_matrix, relation_matrix)

        allpad_senders = None
        allpad_recievers = None

        # add extra relations for sender dyn reciever pad
        if len(pad_indicies) > 0:
            for index in dyn_indicies:
                senders = one_hot_objects[pad_start:]
                receivers = one_hot_objects[index].clone().expand(
                    senders.shape[0], -1)
                if allpad_senders is None:
                    allpad_senders = senders
                    allpad_recievers = receivers
                else:
                    allpad_senders = torch.cat((allpad_senders, senders))
                    allpad_recievers = torch.cat((allpad_recievers, receivers))
            for index in static_indicies:
                senders = one_hot_objects[pad_start:]
                receivers = one_hot_objects[index].clone().expand(
                    senders.shape[0], -1)
                allpad_senders = torch.cat((allpad_senders, senders))
                allpad_recievers = torch.cat((allpad_recievers, receivers))
        if allpad_senders is not None:
            allpad_senders = allpad_senders.permute(1, 0)
            allpad_recievers = allpad_recievers.permute(1, 0)
            pad_relation_matrix = torch.zeros(
                (self.DR, allpad_recievers.shape[1])).to(tensor.device)
            pad_relation_matrix[2, :] = 1

            sender_matrix = torch.cat((sender_matrix, allpad_senders), axis=-1)
            receiver_matrix = torch.cat((receiver_matrix, allpad_recievers),
                                        axis=-1)
            relation_matrix = torch.cat((relation_matrix, pad_relation_matrix),
                                        axis=-1)
        assert sender_matrix.shape == receiver_matrix.shape
        return receiver_matrix, sender_matrix, relation_matrix

    def _marshall_interactions(self, objects, recievers, senders, relations):
        obj_recievers = torch.matmul(objects, recievers)
        obj_senders = torch.matmul(objects, senders)

        if self.position_invariant:
            send_receiver_dyn_diff = obj_recievers[:, :self.
                                                   D_DYN] - obj_senders[:, :self
                                                                        .D_DYN]
            marshalled = torch.cat(
                (send_receiver_dyn_diff, obj_recievers[:, self.D_DYN:],
                 obj_senders[:, self.D_DYN:], relations),
                dim=1)
        else:
            marshalled = torch.cat((obj_recievers, obj_senders, relations),
                                   dim=1)
        return marshalled

    def _compute_effects(self, interactions):
        # interactions is a B x (2Ds + Dr) x Nr
        interactions_permuted = interactions.permute(0, 2, 1)
        effects = self.interactions_model(interactions_permuted)
        tensor_effects = effects.permute(0, 2, 1)
        return tensor_effects

    def _aggregate_effects(self, effects, receiver, external_effects, objects):
        receiver_t = torch.transpose(receiver, 1, 2)
        # effects_reciever is B x self.interactions_size x # objects
        effects_reciever = torch.matmul(effects, receiver_t)

        if self.position_invariant:
            object_features = torch.cat((objects[:, :3], objects[:, 6:]),
                                        dim=1)
        else:
            object_features = objects
        aggregated = torch.cat(
            (object_features, external_effects, effects_reciever), dim=1)
        return aggregated

    def _compute_states(self, aggregated_effects):
        aggregated_effects_permuted = aggregated_effects.permute(0, 2, 1)
        states = self.states_model(aggregated_effects_permuted)
        tensor_states = states.permute(0, 2, 1)
        # tensor states should be B x 3 (x,y, theta diff) x # objects
        return tensor_states

    def forward(self, tensor):
        tensor = tensor.clone()

        receiver_matrix = []
        sender_matrix = []
        relation_matrix = []
        external_effects_matrix = []
        for i in range(tensor.shape[0]):
            black = tensor[i, 0, :, -1] == 1.
            purple = tensor[i, 0, :, -3] == 1.
            static_indicies = (black + purple).type(torch.IntTensor)
            row_sum = torch.sum(tensor[i, 0], dim=-1)
            pad = 2 * (row_sum == 0).type(torch.IntTensor)
            type_values = pad + static_indicies
            type_tuple = tuple(type_values.tolist())
            if type_tuple not in self.relation_matricies:
                rec, send, rel = self._initialize_relations_matricies(
                    tensor[i, 0])
                self.relation_matricies[type_tuple] = (rec, send, rel)
            else:
                (rec, send, rel) = self.relation_matricies[type_tuple]
                rec = rec.to(tensor.device)
                send = send.to(tensor.device)
                rel = rel.to(tensor.device)

            if type_tuple not in self.external_effects_matricies:
                ext = self._intialize_external_effects_matrix(tensor[i, 0])
                self.external_effects_matricies[type_tuple] = ext
            else:
                ext = self.external_effects_matricies[type_tuple]
                ext = ext.to(tensor.device)

            external_effects_matrix.append(ext)
            receiver_matrix.append(rec)
            sender_matrix.append(send)
            relation_matrix.append(rel)
        external_effects_matrix = torch.stack(external_effects_matrix, dim=0)
        receiver_matrix = torch.stack(receiver_matrix, dim=0)
        sender_matrix = torch.stack(sender_matrix, dim=0)
        relation_matrix = torch.stack(relation_matrix, dim=0)
        # input tensor is B x n-frames (2) x N-objects x VECTOR_LENGTH
        assert tensor.shape[1] == 2
        velocities = tensor.select(1, -1).narrow(2, 0, 3) - tensor.select(
            1, 0).narrow(2, 0, 3)
        # object_features is B x  N-objects x (3 (velocities) + VECTOR_LENGTH)
        # Ds = (3 (velocities) + VECTOR_LENGTH)
        object_features = torch.cat((velocities, tensor.select(1, -1)), dim=-1)
        # permute to B x state x N objects
        object_features = object_features.permute(0, 2, 1)

        # Dr is 1 (all 0's)
        # Nr = number of relations = # dynamic objects * (# objects - 1)
        # interactions is a B x (2Ds + Dr) x Nr
        interactions = self._marshall_interactions(object_features,
                                                   receiver_matrix,
                                                   sender_matrix,
                                                   relation_matrix)

        #effects should be B x self.interactions_size # Nr
        effects = self._compute_effects(interactions)

        #aggregated_effects should be B x (DS + DX + self.interactions_size) x #Number objects
        aggregated_effects = self._aggregate_effects(effects, receiver_matrix,
                                                     external_effects_matrix,
                                                     object_features)

        #aggregated_effects matrix is B x (DS ({DS}) + DX ({DX}) + self.#interactions_size  x N-objects

        # states is a B x 3 x N objects matrix of velocities

        states = self._compute_states(aggregated_effects)

        # transpose back to B x N x 3
        states = states.permute(0, 2, 1)

        # add to previous state to get predictions
        next_state = tensor.select(1, -1)
        if self.dont_predict_ball_theta_v:
            is_ball = next_state[:, :, 4] == 1.
            if len(states[is_ball]) > 0:
                # This doesn't actually state values, remove or update
                states[is_ball][-1] = 0.
        if self.only_pred_dynamic:
            black = next_state[:, :, -1] == 1.
            purple = next_state[:, :, -3] == 1.
            static_indicies = black + purple
            states[static_indicies] = 0.0
            row_sum = torch.sum(next_state, dim=-1)
            states[row_sum == 0] = 0.0

        prediction = next_state.clone()
        prediction[:, :, :3] += states
        if self.clip_output is not None:
            prediction[:, :, :3] = self.clip_output(
                prediction[:, :, :3].clone()).clone()
            if self.only_pred_dynamic:
                prediction[:, :, :3][static_indicies] = next_state[:, :, :3][
                    static_indicies]
        return prediction


class TemporalInteractionNetwork(nn.Module):
    DS = phyre.FeaturizedObjects._NUM_FEATURES + 3  # VECTOR_LENGTH + 3
    DR = 3
    D_DYN = 2 * DR  # x,y, theta, vx, vy, vtheta
    DX = 1

    def __init__(self,
                 hist_frames=3,
                 only_pred_dynamic=False,
                 clip_output=False,
                 dont_predict_ball_theta_v=False,
                 position_invariant=False):
        super(TemporalInteractionNetwork, self).__init__()
        self.dont_predict_ball_theta_v = dont_predict_ball_theta_v
        self.only_pred_dynamic = only_pred_dynamic
        self.clip_output = None if not clip_output else torch.nn.Hardtanh(0, 1)

        self.position_invariant = position_invariant

        self.output_size = 3  #vx, vy, vtheta
        logging.info(f'hist frames {hist_frames}')
        assert hist_frames == 3, 'Temporal IN only for three historical frames'
        self.hist_frames = hist_frames
        self.interaction_net_short = InteractionNetwork(
            hist_frames=2,
            only_pred_dynamic=only_pred_dynamic,
            clip_output=False,
            dont_predict_ball_theta_v=dont_predict_ball_theta_v,
            position_invariant=position_invariant,
        )
        self.interaction_net_long = InteractionNetwork(
            hist_frames=2,
            only_pred_dynamic=only_pred_dynamic,
            clip_output=False,
            dont_predict_ball_theta_v=dont_predict_ball_theta_v,
            position_invariant=position_invariant,
        )
        interaction_output = phyre.FeaturizedObjects._NUM_FEATURES
        hidden_size = 64  #int(math.sqrt((DR*2 + interaction_output)*DR)) + 1
        self.aggregator = nn.Sequential(
            nn.Linear(self.DR * 2 + interaction_output, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.DR),
        )

    def forward(self, tensor):
        tensor = tensor.clone()

        last_pos = tensor[:, -1, :, :].clone()
        short_t = tensor[:, -2:, :, :]
        long_t = tensor[:, 0:-1, :, :]
        short_output = self.interaction_net_short(
            short_t)[:, :, :3] - last_pos[:, :, :3]

        long_output = self.interaction_net_long(
            long_t)[:, :, :3] - last_pos[:, :, :3]
        temporal_cat = torch.cat((last_pos, short_output, long_output), dim=-1)
        states = self.aggregator(temporal_cat)

        if self.dont_predict_ball_theta_v:
            is_ball = last_pos[:, :, 4] == 1.
            if len(states[is_ball]) > 0:
                # This doesn't actually state values, remove or update
                states[is_ball][-1] = 0.
        if self.only_pred_dynamic:
            black = last_pos[:, :, -1] == 1.
            purple = last_pos[:, :, -3] == 1.
            static_indicies = black + purple
            states[static_indicies] = 0.0
            row_sum = torch.sum(last_pos, dim=2)
            states[row_sum == 0] = 0.0

        prediction = last_pos.clone()
        prediction[:, :, :3] += states
        if self.clip_output is not None:
            prediction[:, :, :3] = self.clip_output(
                prediction[:, :, :3].clone()).clone()
            if self.only_pred_dynamic:
                prediction[:, :, :3][static_indicies] = tensor[:, -2, :, :3][
                    static_indicies]
        return prediction


class DummyClassificationModel(ForwardBaseModel):
    def forward(self, tensor):
        return torch.zeros(tensor.shape[0]).to(tensor.device)


class PositionalEncoding(nn.Module):
    def __init__(self,
                 input_emb=phyre.FeaturizedObjects._NUM_FEATURES,
                 num_inp=phyre_simulator.MAX_NUM_OBJECTS,
                 dropout=0.1,
                 max_len=20):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, input_emb)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, input_emb, 2).float() *
            (-math.log(10000.0) / input_emb))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        pe = pe.expand(-1, num_inp, -1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :x.size(2), :]
        return self.dropout(x)


class TransformerPredictionNetwork(ForwardBaseModel):
    def __init__(self,
                 n_hist_frames=3,
                 n_fwd_times=0,
                 n_heads=8,
                 n_layers=6,
                 embedding_size=128,
                 aggregate_hidden_size=128,
                 score_hidden_first_layer_scale=1,
                 aggregate='mean',
                 n_inp=phyre_simulator.MAX_NUM_OBJECTS,
                 embed_tf=True,
                 shuffle_embed=False,
                 class_extra_layer=False,
                 tf_layer_norm=False):
        super().__init__()
        self.n_inp = n_inp
        self.score_hidden_first_layer_scale = score_hidden_first_layer_scale
        self.n_timesteps = n_hist_frames + n_fwd_times
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.embedding_size = embedding_size
        self.embed_tf = embed_tf
        self.shuffle_embed = shuffle_embed
        self.tf_layer_norm = tf_layer_norm
        self.encoder = nn.Sequential(
            nn.Linear(phyre.FeaturizedObjects._NUM_FEATURES,
                      self.embedding_size),
            nn.ReLU(),
            nn.Linear(self.embedding_size, self.embedding_size),
        )
        self.pos_encoder = PositionalEncoding(self.embedding_size,
                                              max_len=self.n_timesteps,
                                              num_inp=self.n_inp)
        encoder_layers = TransformerEncoderLayer(self.embedding_size,
                                                 nhead=self.n_heads)
        if self.tf_layer_norm:
            norm = nn.LayerNorm(self.embedding_size)
        else:
            norm = None
        self.transformer_encoder = TransformerEncoder(encoder_layers,
                                                      num_layers=self.n_layers,
                                                      norm=norm)
        self.aggregate_hidden_size = aggregate_hidden_size

        self.aggregation = aggregate
        self.class_extra_layer = class_extra_layer
        self.score_obj = nn.Sequential(
            nn.Linear(self.embedding_size, self.aggregate_hidden_size),
            nn.ReLU(),
            nn.Linear(self.aggregate_hidden_size, self.aggregate_hidden_size),
            nn.ReLU(),
            nn.Linear(self.aggregate_hidden_size, 1),
        )
        self.score = nn.Sequential(
            nn.Linear(
                phyre_simulator.MAX_NUM_OBJECTS * self.embedding_size *
                self.n_timesteps, self.aggregate_hidden_size),
            nn.ReLU(),
            nn.Linear(self.aggregate_hidden_size, self.aggregate_hidden_size),
            nn.ReLU(),
            nn.Linear(self.aggregate_hidden_size, 1),
        )
        if self.class_extra_layer:
            self.score_goal = nn.Sequential(
                nn.Linear(
                    2 * self.embedding_size * self.n_timesteps,
                    int(self.embedding_size * self.n_timesteps *
                        phyre_simulator.MAX_NUM_OBJECTS / 2.0)),
                nn.ReLU(),
                nn.Linear(
                    int(self.embedding_size * self.n_timesteps *
                        phyre_simulator.MAX_NUM_OBJECTS / 2.0),
                    self.aggregate_hidden_size),
                nn.ReLU(),
                nn.Linear(self.aggregate_hidden_size,
                          self.aggregate_hidden_size),
                nn.ReLU(),
                nn.Linear(self.aggregate_hidden_size, 1),
            )
            self.score_timestep_embeddings = nn.Sequential(
                nn.Linear(
                    self.embedding_size * self.n_timesteps,
                    self.embedding_size * self.n_timesteps *
                    phyre_simulator.MAX_NUM_OBJECTS),
                nn.ReLU(),
                nn.Linear(
                    self.embedding_size * self.n_timesteps *
                    phyre_simulator.MAX_NUM_OBJECTS,
                    self.aggregate_hidden_size),
                nn.ReLU(),
                nn.Linear(self.aggregate_hidden_size,
                          self.aggregate_hidden_size),
                nn.ReLU(),
                nn.Linear(self.aggregate_hidden_size, 1),
            )
        else:
            self.score_goal = nn.Sequential(
                nn.Linear(
                    2 * self.embedding_size * self.n_timesteps,
                    self.aggregate_hidden_size *
                    self.score_hidden_first_layer_scale),
                nn.ReLU(),
                nn.Linear(
                    self.aggregate_hidden_size *
                    self.score_hidden_first_layer_scale,
                    self.aggregate_hidden_size),
                nn.ReLU(),
                nn.Linear(self.aggregate_hidden_size, 1),
            )
            self.score_timestep_embeddings = nn.Sequential(
                nn.Linear(
                    self.embedding_size * self.n_timesteps,
                    self.aggregate_hidden_size *
                    self.score_hidden_first_layer_scale),
                nn.ReLU(),
                nn.Linear(
                    self.aggregate_hidden_size *
                    self.score_hidden_first_layer_scale,
                    self.aggregate_hidden_size),
                nn.ReLU(),
                nn.Linear(self.aggregate_hidden_size, 1),
            )

    def forward(self, t):
        # tensor is B x T x N x F
        # mask padding
        tensor = t.clone()
        if self.aggregation == 'mlp':
            # in case agent.strip_padding is true and < max num actions, pad
            num_padding = phyre_simulator.MAX_NUM_OBJECTS - tensor.shape[2]
            if num_padding > 0:
                pad_zeros = torch.zeros(tensor.shape[0], tensor.shape[1],
                                        num_padding,
                                        tensor.shape[3]).to(tensor.device)
                tensor = torch.cat((tensor, pad_zeros), dim=2)
        row_sum = torch.sum(tensor, dim=-1)
        is_pad = row_sum == 0
        #mask = is_pad
        tensor_enc = self.encoder(tensor)
        # tensor is B x T x N x 128
        tensor = self.pos_encoder(tensor_enc) * math.sqrt(self.n_inp)
        # tensor is B x T x N x 128
        tensor = torch.flatten(tensor, start_dim=1, end_dim=2)

        mask = torch.flatten(is_pad, start_dim=1, end_dim=2)
        # mask is B x (T x N)
        # tensor is B x (T x N) x 128
        tensor = tensor.permute(1, 0, 2)
        # tensor is (T x N) x B x 128
        if self.embed_tf:
            tensor = self.transformer_encoder(tensor,
                                              src_key_padding_mask=mask)
        # tensor is (T x N) x B x 128
        if self.shuffle_embed:
            indicies = torch.randperm(tensor.shape[0])
            tensor = tensor[indicies]

        if self.aggregation == 'mlp_copy_row':
            first_elem = tensor[0]
            tensor = first_elem.unsqueeze(0).expand(tensor.shape)
            tensor = tensor.permute(1, 0, 2)
            # tensor is  B x (T x N) x 128
            tensor = torch.flatten(tensor, start_dim=1, end_dim=-1)
            # tensor is  B x (T x N x 128)
            scores = self.score(tensor).squeeze(-1)

        if self.aggregation == 'mlp':
            tensor = tensor.permute(1, 0, 2)
            # tensor is  B x (T x N) x 128
            tensor = torch.flatten(tensor, start_dim=1, end_dim=-1)
            # tensor is  B x (T x N x 128)
            scores = self.score(tensor).squeeze(-1)
        elif self.aggregation == 'mean':
            # tensor is (T x N) x B x 128
            tensor = tensor.permute(1, 0, 2)
            # tensor is  B x (T x N) x 128
            tensor = torch.flatten(tensor, start_dim=1, end_dim=-1)
            # tensor is  B x (T x N x 128)
            scores = torch.mean(tensor, dim=-1)
        elif self.aggregation == 'mlp_mean':
            # tensor is (T x N) x B x 128
            tensor = tensor.permute(1, 0, 2)
            # tensor is  B x (T x N) x 128
            tensor = self.score_obj(tensor).squeeze(-1)
            # tensor is  B x (T x N)
            scores = torch.mean(tensor, dim=-1)
        elif self.aggregation == 'mean_pool_over_objects':
            tensor = tensor.permute(1, 0, 2)
            # tensor is B x (T x N) x E
            tensor = tensor.reshape(tensor_enc.shape)
            # tensor is B x T x N x E
            mean_pooled_obj = torch.mean(tensor, dim=2).squeeze(2)
            # mean_pooled_obj is B x T x E
            flattened_pooled = torch.flatten(mean_pooled_obj, start_dim=1)
            # flattened_pooled is B x (T x E)
            scores = self.score_timestep_embeddings(flattened_pooled).squeeze(
                -1)
            # scores is [B,]
        elif self.aggregation == 'max_pool_over_objects':
            tensor = tensor.permute(1, 0, 2)
            # tensor is B x (T x N) x E
            tensor = tensor.reshape(tensor_enc.shape)
            # tensor is B x T x N x E
            mean_pooled_obj = torch.max(tensor, dim=2).values.squeeze(2)
            # mean_pooled_obj is B x T x E
            flattened_pooled = torch.flatten(mean_pooled_obj, start_dim=1)
            # flattened_pooled is B x (T x E)
            scores = self.score_timestep_embeddings(flattened_pooled).squeeze(
                -1)
            # scores is [B,]
        elif self.aggregation == 'max':
            # tensor is (T x N) x B x 128
            tensor = tensor.permute(1, 0, 2)
            # tensor is  B x (T x N) x 128
            tensor = torch.flatten(tensor, start_dim=1, end_dim=-1)
            # tensor is  B x (T x N x 128)
            scores = torch.max(tensor, dim=-1).values.squeeze(-1)
        elif self.aggregation == 'mlp_max':
            # tensor is (T x N) x B x 128
            tensor = tensor.permute(1, 0, 2)
            # tensor is  B x (T x N) x 128
            tensor = self.score_obj(tensor).squeeze(-1)
            # tensor is  B x (T x N)
            scores = torch.max(tensor, dim=-1).values.squeeze(-1)
        elif self.aggregation == 'goal_mlp':
            tensor = tensor.permute(1, 0, 2)
            # tensor is B x (T x N) x E
            tensor = tensor.reshape(tensor_enc.shape)
            goal_tensor = t.clone()
            purple = goal_tensor[:, :, :, -3] == 1.
            blue = goal_tensor[:, :, :, -4] == 1.
            green = goal_tensor[:, :, :, -5] == 1.

            goal_indicies = purple + blue + green
            goal_objs = tensor[goal_indicies].reshape(
                (tensor.shape[0], tensor.shape[1], -1, tensor.shape[-1]))
            goal_objs = torch.flatten(goal_objs, start_dim=1)
            scores = self.score_goal(goal_objs).squeeze(-1)
        # scores is  B x 1
        return scores


class DummyFowardModel(ForwardBaseModel):
    def __init__(self):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(phyre.FeaturizedObjects._NUM_FEATURES * 1,
                      128), nn.Linear(128, 128),
            nn.Linear(128, phyre.FeaturizedObjects._NUM_FEATURES))

    def forward(self, tensor):
        # tensor is B x T x MAX_NUM_OBJECTS X Features
        return tensor[:,-1]


class PlaceholderFowardModel(ForwardBaseModel):
    def forward(self, tensor):
        raise NotImplementedError

class PlaceholderClassificationModel(ForwardBaseModel):
    def forward(self, tensor):
        raise NotImplementedError


class PixelClasssificationModel(ForwardBaseModel):
    def __init__(self, agent_cfg):
        super().__init__()
        # The image embedding model
        self.preproc = preproc.VideoPreprocessor(agent_cfg)
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

    def forward(self, vid):
        vid_preproc = self.preproc.preprocess_vid(vid)
        obj_feat = self.enc(vid_preproc)
        clip_hist = self.interactor(obj_feat)
        pix = nets.combine_obj_pixels(vid_preproc, 2)
        # Compute the solved or not, will only do for the ones asked for
        clip_preds_solved = self._cls(nets.combine_obj_pixels(clip_hist, 2),
                                      pix, [], [])
        return clip_preds_solved

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


class FwdObject(ForwardBaseModel):
    """
    Forward model for object space. Handles both rollout models in object
    space and object spaced classification models.
    """
    def __init__(self, config):
        super().__init__()
        self.forward_model = hydra.utils.instantiate(config.agent.obj_fwd)
        if config.agent.cls_pixel:
            self.classification_model = PixelClasssificationModel(config.agent)
        else:
            self.classification_model = hydra.utils.instantiate(
                config.agent.obj_cls)
        self.strip_padding = config.agent.strip_padding
        self.pad_for_classifier = False  #config.agent.pad_for_classifier
        self.cls_pixel = config.agent.cls_pixel
        self.n_hist_frames = config.agent.model_hist_frames

        self.dyn_dont_predict_ball_theta = config.agent.dyn_dont_predict_ball_theta
        self.dyn_use_cos_theta_loss = config.agent.dyn_use_cos_theta_loss
        self.train_noise_std = config.agent.train_noise_std

    def _forward_dyn(self,
                     frames,
                     n_fwd_times,
                     need_intermediate,
                     train_noise_frac=0):
        """
        Returns tuple of the full rollout, including GT frames, predicitons,
            and any losses incurred making the predicitons.
            If need_intermediate, predicitons is all preidictions made during
            n_fwd_times, otherwise only the last prediction and its
            corresponding losses are returned.
        """
        if n_fwd_times == 0:
            return frames, None, {}
        rollout = []
        previous_frames = frames.clone()
        if train_noise_frac > 0:
            # frames is B x T x N x F
            dataset_std = self.train_noise_std  # naive approximation
            noise_amt = torch.normal(
                0, dataset_std,
                (previous_frames.shape[0], previous_frames.shape[1],
                 previous_frames.shape[2], 3))
            bool_noise = torch.empty(noise_amt.shape).uniform_(
                0, 1) > train_noise_frac
            noise_amt *= bool_noise.type(torch.FloatTensor)

            # no noise on padding
            row_sum = torch.sum(previous_frames, dim=-1)
            pad = row_sum == 0
            pad_mask = pad
            noise_amt[pad_mask] = 0.0
            noise_amt = noise_amt.to(previous_frames.device)

            previous_frames[:, :, :, :3] += noise_amt

        for _ in range(n_fwd_times):
            pred_frame = self.forward_model(previous_frames)
            tmp = previous_frames[:, 1:].clone()
            previous_frames[:, -1] = pred_frame
            previous_frames[:, :-1] = tmp.clone()
            rollout.append(pred_frame.unsqueeze(1))
        full_rollout = torch.cat([frames] + rollout, dim=1)
        if need_intermediate:
            # Return all predicitons
            return full_rollout, torch.cat(rollout, dim=1), {}
        # Only return the last frame predicted
        return full_rollout, rollout[-1], {}

    def classification_losses(self, predictions, is_solved):
        assert predictions.shape[0] % is_solved.shape[0] == 0
        return {
            'ce':
            self.ce_loss(
                predictions,
                is_solved.repeat((predictions.shape[0] //
                                  is_solved.shape[0], ))).unsqueeze(0)
        }

    def ce_loss(self, predicted, solved):
        solved = solved.to(dtype=torch.float, device=predicted.device)
        return nn.functional.binary_cross_entropy_with_logits(
            predicted, solved)

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

    def l2_loss(self, decisions, targets):
        # decisions, targets tensor is B x N-objects x VECTOR_LENGTH
        # get only first three dimensions x,y, theta
        targets = targets.to(dtype=torch.float, device=decisions.device)
        decisions_pos = decisions.clone()
        targets_pos = targets.clone()
        decisions_pos = decisions_pos.narrow(2, 0, 3)
        targets_pos = targets_pos.narrow(2, 0, 3)
        if self.dyn_dont_predict_ball_theta:
            is_ball = decisions[:, :, 4] == 1.
            decisions_pos[:, :, 2][is_ball] = targets_pos[:, :, 2][is_ball]
        if self.dyn_use_cos_theta_loss:
            decisions_pos[:, :, -1] = torch.cos(2 * math.pi *
                                                decisions_pos[:, :, -1])
            targets_pos[:, :, -1] = torch.cos(2 * math.pi *
                                              targets_pos[:, :, -1])
        black = decisions[:, :, -1] == 1.
        purple = decisions[:, :, -3] == 1.
        row_sum = torch.sum(decisions, dim=2)
        pad = row_sum == 0
        dont_use_in_loss = black + purple + pad
        return nn.functional.mse_loss(decisions_pos[~dont_use_in_loss],
                                      targets_pos[~dont_use_in_loss])

    def _compute_losses(self, pred_obj_roll, gt_obj_roll, n_hist_frames,
                        n_fwd_times):
        """
        Compute all losses possible.
        """
        dummy_loss = torch.Tensor([-1]).to(pred_obj_roll.device)
        losses = {}
        # NCE and pixel loss
        # find the GT for each clip, note that all predictions may not have a GT
        # since the last n_hist_frames for a video will make a prediction that
        # goes out of the list of frames that were extracted for that video.

        obj_preds = []
        obj_gt = []

        batch_size = gt_obj_roll.shape[0]  #vid_feat.shape[0]
        gt_max_time = gt_obj_roll.shape[1]  #vid_feat.shape[1]
        # Max slices that could have been made of the data, to use all of the
        # training clip
        max_slices_with_gt = gt_max_time - n_hist_frames - n_fwd_times + 1
        num_slices = pred_obj_roll.shape[0] // batch_size  # clip_pred
        for i in range(min(max_slices_with_gt, num_slices)):
            corr_pred = pred_obj_roll[i * batch_size:(i + 1) * batch_size,
                                      ...]  # clip_pred
            # Get the corresponding GT predictions for this pred
            corr_gt = gt_obj_roll[:, i + n_hist_frames + n_fwd_times -
                                  1]  # vid_feat
            assert corr_gt.shape == corr_pred.shape
            obj_preds.append(corr_pred)  #feat_preds
            obj_gt.append(corr_gt)  #feat_gt
        if len(obj_gt) > 0:  # feat_gt
            # Keep a batch dimension to the loss, since it will be run over
            # multiple GPUs
            obj_preds = torch.cat(obj_preds)  # feat_preds
            obj_gt = torch.cat(obj_gt)  # feat_preds
            losses['l2'] = self.l2_loss(obj_preds, obj_gt).unsqueeze(0)
        return losses

    def render_frames(self, full_rollout):
        full_rollout = full_rollout.cpu().detach()
        results = [
            render_rollout(full_rollout, each)
            for each in range(full_rollout.shape[0])
        ]
        results = {r[0]: r[1] for r in results}
        pixel_rollout = torch.stack(list(results.values()), dim=0).unsqueeze(2)
        return pixel_rollout.to(self.device)

    def fwd_only_dyn(self, tensor, n_hist_frames=3, n_fwd_times=1):
        tensor = tensor.to(self.device).clone()
        if self.strip_padding:
            # strip off uncesssary padidng
            row_sum = torch.sum(tensor, dim=(0, 1, 3))
            pad = row_sum == 0
            pad_mask = pad.unsqueeze(0).unsqueeze(0).expand(tensor.shape[:-1])
            masked_tensor = tensor[~pad_mask]
            masked_tensor = masked_tensor.reshape(tensor.shape[0],tensor.shape[1],-1,tensor.shape[-1])
            tensor = masked_tensor
        previous_frames = self._slice_for_dyn(tensor,
                                              n_hist_frames,
                                              nslices=1)
        assert previous_frames.shape[1] == n_hist_frames
        rollout, roll_pred, roll_pred_addl_losses = self._forward_dyn(
            previous_frames, n_fwd_times, True)
        return rollout, roll_pred

    def forward(self,
                obj_tensor,
                tensor_is_solved,
                n_hist_frames=3,
                n_fwd_times=1,
                n_fwd_times_incur_loss=999999,
                run_decode=False,
                compute_losses=False,
                need_intermediate=False,
                autoenc_loss_ratio=0.0,
                nslices=-1,
                need_pixels=True,
                train_noise_frac=0.0):
        tensor = obj_tensor.to(self.device).clone()
        if self.strip_padding:
            # strip off uncesssary padidng
            row_sum = torch.sum(tensor, dim=(0, 1, 3))
            pad = row_sum == 0
            pad_mask = pad.unsqueeze(0).unsqueeze(0).expand(tensor.shape[:-1])
            masked_tensor = tensor[~pad_mask]
            masked_tensor = masked_tensor.reshape(tensor.shape[0],
                                                  tensor.shape[1], -1,
                                                  tensor.shape[-1])
            tensor = masked_tensor
        previous_frames = self._slice_for_dyn(tensor,
                                              n_hist_frames,
                                              nslices=nslices)
        assert previous_frames.shape[1] == n_hist_frames
        rollout, roll_pred, roll_pred_addl_losses = self._forward_dyn(
            previous_frames, n_fwd_times, need_intermediate, train_noise_frac)
        pix_rollout = None
        if self.cls_pixel:
            rollout = self.render_frames(rollout)
        if need_pixels and roll_pred is not None:
            pix_rollout = self.render_frames(roll_pred).squeeze(2)
            pix_rollout = [
                pix_rollout[:, i] for i in range(pix_rollout.shape[1])
            ]

        predictions = self.classification_model(rollout)
        losses = {}
        all_losses = []
        if compute_losses:
            if roll_pred is not None:
                for i in range(min(roll_pred.shape[1], n_fwd_times_incur_loss)):
                    # Compute losses at each prediction step, if need_intermediate
                    # is set. Else, it will only return a single output
                    # (at the last prediction), and then we can only incur loss at
                    # that point.
                    if not need_intermediate:
                        assert roll_pred.shape[1] == 1
                        pred_id = -1
                        # Only loss on predicting the final rolled out obs
                        this_fwd_times = n_fwd_times
                    else:
                        assert len(roll_pred.shape[1]) == n_fwd_times
                        pred_id = i
                        this_fwd_times = i + 1
                    all_losses.append(
                        self._compute_losses(roll_pred[:, pred_id], tensor,
                                            n_hist_frames, this_fwd_times))
                    # For the loss, using only the last prediction (for now)
            losses = nets.average_losses(all_losses)
            losses.update(nets.average_losses(roll_pred_addl_losses))
            classification_losses = self.classification_losses(
                predictions, tensor_is_solved)
            losses.update(classification_losses)
            # Add losses on the provided frames if requested
        all_preds = {
            'feats': None,  # Match object fwd model
            'is_solved': predictions,
            'pixels':
            pix_rollout,
        }
        return all_preds, losses
