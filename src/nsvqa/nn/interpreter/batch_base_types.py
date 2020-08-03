# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file
# in the project root for full license information.

"The basic types for the interpreter."

import torch
import torch.nn as nn

from nsvqa.nn.interpreter import util
from enum import IntEnum

######################################################################################################################################

class Quantifier(IntEnum):
    FOR_ALL = 0
    EXISTS = 1

class QuestionType(IntEnum):
    BINARY = 0
    QUERY = 1
    STATEMENT = 2
    OBJECT_STATEMENT = 3
    SCENE_GRAPH = 4

class TokenType(IntEnum):
    ATTRIBUTE = 0
    RELATION = 1
    NAME = 2
    CATEGORY = 3

######################################################################################################################################

class BatchVariableSet(object):
    
    def __init__(self, names, device, object_num, batch_size=1, quantifiers=Quantifier.EXISTS, log_attention=None, 
                batch_object_map=None, predicate_question_map=None, base_cumulative_loss=0, prev_variable_sets_num=0):
        # if batch_object_map is not None:
        #     assert batch_object_map.size()[0] == batch_size, "The batch size for Variable Set does not match."

        assert batch_object_map is not None or batch_size == 1, "Batch object map must be provided for batch_size > 1."
        
        self._name = names
        self._device = device
        self._object_num = object_num
        self._batch_size = batch_size
        self._base_cumulative_loss = base_cumulative_loss
        self._prev_variable_sets_num = prev_variable_sets_num

        if isinstance(quantifiers, (int, float, Quantifier)):
            self._quantifier = float(quantifiers) * torch.ones(self._batch_size, device=self._device)
        elif isinstance(quantifiers, (list, tuple)):
            self._quantifier = torch.tensor(quantifiers, dtype=torch.float32, device=self._device)
        else:
            self._quantifier = quantifiers

        if log_attention is None:
            self._log_attention = torch.zeros(self._batch_size, self._object_num, device=self._device)
        else:
            self._log_attention = log_attention

        if batch_object_map is None:
            batch_index = torch.zeros(object_num, dtype=torch.int64, device=self._device)

            all_ones = torch.ones(object_num, device=self._device)
            y_ind = torch.arange(object_num, dtype=torch.int64, device=self._device)
            ind = torch.stack([batch_index, y_ind])
            
            if self._device.type == 'cuda':
                self._batch_object_map = torch.cuda.sparse.FloatTensor(ind, all_ones, 
                                torch.Size([self._batch_size, object_num]), device=self._device)
            else:
                self._batch_object_map = torch.sparse.FloatTensor(ind, all_ones, 
                                torch.Size([self._batch_size, object_num]), device=self._device)
        else:
            self._batch_object_map = batch_object_map

        self._predicate_question_map = predicate_question_map

    def to(self, dtype):
        if dtype != self.dtype:
            self._quantifier = self._quantifier.to(dtype)
            self._log_attention = self._log_attention.to(dtype)
            self._batch_object_map = self._batch_object_map.to(dtype) if self._batch_object_map is not None else None
            self._predicate_question_map = self._predicate_question_map.to(dtype) if self._predicate_question_map is not None else None

        return self

    @property
    def dtype(self):
        return self._log_attention.dtype

    @property
    def device(self):
        return self._device

    def object_num(self):
        return self._object_num

    def batch_size(self):
        return self._batch_size

    def log_probability(self, hard_mode=False):
        if hard_mode:
            log_posterior = util.log_parametric_not(self._log_attention, self._quantifier.unsqueeze(1), 1)
            
            if self._predicate_question_map is not None:
                log_posterior = (util.mm(self._predicate_question_map, self._batch_object_map.to_dense()) * log_posterior).min(1)[0]
            else:
                log_posterior = (self._batch_object_map.to_dense() * log_posterior).min(1)[0]

            log_posterior = util.log_parametric_not(log_posterior, self._quantifier, 1)        

        else:
            log_posterior = self._log_attention.transpose(0, 1).contiguous()
            log_posterior = util.log_parametric_not(log_posterior, self._quantifier.unsqueeze(0), 1)

            if self._predicate_question_map is not None:
                log_posterior = util.mm(self._predicate_question_map, util.mm(self._batch_object_map, log_posterior))
            else:
                log_posterior = util.mm(self._batch_object_map, log_posterior)

            log_posterior = util.log_parametric_not(log_posterior.diag(), self._quantifier, 1)        
        
        return log_posterior

    def cumulative_loss(self):
        # Compute entropy loss
        # p = self._log_attention.exp()
        # return self._base_cumulative_loss - (p * self._log_attention + (1 - p) * util.log_not(self._log_attention)).mean()
        return 0

    def mean_cumulative_loss(self):
        return self.cumulative_loss() / (self._prev_variable_sets_num + 1)

    def _get_str_representation(self):
        return "Attention: " + str(util.safe_exp(self._log_attention)) + "\n" + \
                "Total likelihood: " + str(util.safe_exp(self.log_probability()))

    def __repr__(self):
        return "Object set of " + str(self._object_num) + " objects:\n" + self._get_str_representation()

    def __str__(self):
        return self._get_str_representation()

    def get_attention(self):
        return util.safe_exp(self._log_attention)

    def gate(self, variable_set, flag):
        if isinstance(flag, torch.Tensor):
            g = flag
        else:
            flag = [0 if f is None else f for f in flag]
            g = torch.tensor(flag, dtype=variable_set.dtype, device=self._device)
        
        quantifier = self._quantifier * g + variable_set._quantifier * (1.0 - g)
        
        g = g.unsqueeze(1)
        log_attention = self._log_attention * g + variable_set._log_attention * (1.0 - g)

        if isinstance(flag, torch.Tensor):
            foo = flag.cpu().numpy().tolist()
        else:
            foo = flag

        names = [x if f > 0 else y for x, y, f in zip(self._name, variable_set._name, foo)]
        return BatchVariableSet(names, self._device, self._object_num, self._batch_size, quantifiers=quantifier, 
                log_attention=log_attention, batch_object_map=self._batch_object_map, predicate_question_map=self._predicate_question_map).to(self.dtype)

    def apply_modulations(self, modulations, input_variable_set, predicate_question_map=None):
        if modulations is not None:
            max_activation = 10
            alpha = modulations[:, 0].unsqueeze(1) * max_activation
            beta = modulations[:, 1].unsqueeze(1) * max_activation
            c = modulations[:, 2].unsqueeze(1) * max_activation if modulations.size()[1] > 2 else torch.ones(1, device=self._device, dtype=modulations.dtype)
            d = modulations[:, 3].unsqueeze(1) if modulations.size()[1] > 3 else 0.5 * torch.ones(1, device=self._device, dtype=modulations.dtype)
            temp = alpha * self._log_attention + util.safe_log(c) + util.safe_log(d)
            self._log_attention = temp - util.safe_log((beta * util.log_not(self._log_attention) + util.safe_log(1.0 - d)).exp() + temp.exp())

            if modulations.size()[1] > 4:
                g = modulations[:, 4].unsqueeze(1)
                if predicate_question_map is None:
                    self._log_attention = util.safe_log(g * self._log_attention.exp() + (1.0 - g) * input_variable_set._log_attention.exp())
                else:
                    self._log_attention = util.safe_log(g * self._log_attention.exp() + (1.0 - g) * util.mm(predicate_question_map, input_variable_set._log_attention.exp()))

        return self

######################################################################################################################################

class BatchWorld(object):
    
    def __init__(self, device, object_num, attribute_features, relation_features, batch_index, meta_data=None, attention_transfer_state_dim=0):
        self._device = device
        self._attribute_features = attribute_features
        self._relation_features = relation_features
        self._object_num = object_num
        self._object_image_map = batch_index
        self._meta_data = meta_data
        self._attention_transfer_state_dim = attention_transfer_state_dim

        if isinstance(batch_index, (list, tuple)):
            self._batch_size = int(max(batch_index) + 1)
        else:
            self._batch_size = int(batch_index.max().cpu().numpy().tolist() + 1)

        if batch_index is None:
            bi = torch.zeros(self._object_num, dtype=torch.int64, device=self._device)
        else:
            bi = torch.tensor(batch_index, dtype=torch.int64, device=self._device)

        all_ones = torch.ones(self._object_num, device=self._device)
        y_ind = torch.arange(self._object_num, dtype=torch.int64, device=self._device)
        ind = torch.stack([bi, y_ind])
        
        if isinstance(self._device, int) or self._device.type == 'cuda':
            self._batch_object_map = torch.cuda.sparse.FloatTensor(ind, all_ones, 
                            torch.Size([self._batch_size, self._object_num]), device=self._device)
        else:
            self._batch_object_map = torch.sparse.FloatTensor(ind, all_ones, 
                            torch.Size([self._batch_size, self._object_num]), device=self._device)

    def to(self, dtype):
        if dtype != self.dtype:
            self._attribute_features = util.to(self._attribute_features, dtype)
            self._relation_features = util.to(self._relation_features, dtype)
            self._meta_data = util.to(self._meta_data, dtype)
            self._batch_object_map = self._batch_object_map.to(dtype)

        return self

    @property
    def dtype(self):
        return self._batch_object_map.dtype

    def batch_size(self):
        return self._batch_size
    
    def object_num(self):
        return self._object_num

    def word_embedding_dim(self):
        return self._meta_data['embedding'].size()[1]
        
    def variable_set(self, names, quantifier=Quantifier.EXISTS, log_attention=None):
        return BatchVariableSet(names, self._device, self._object_num, self._batch_size, quantifiers=quantifier, 
                log_attention=log_attention, batch_object_map=self._batch_object_map).to(self._batch_object_map.dtype)

    def attention_state(self, name, state=None):
        return BatchAttentionState(name, self._device, state if state is not None \
            else (torch.zeros(self.batch_size(), self._attention_transfer_state_dim, device=self._device), \
                torch.zeros(self.batch_size(), self._attention_transfer_state_dim, device=self._device))).to(self._batch_object_map.dtype)

######################################################################################################################################

class BatchAttentionState(object):

    def __init__(self, name, device, state, set_zeros=False):
        self._name = name
        self._device = device
        self._state = (torch.zeros_like(state[0]), torch.zeros_like(state[1])) if set_zeros else state

    def to(self, dtype):
        if dtype != self.dtype:
            self._state = util.to(self._state, dtype)
        
        return self
    
    @property
    def dtype(self):
        return self._state[0].dtype

    @property
    def device(self):
        return self._device

    def state_size(self):
        return self._state[0].size()[1]

    def gate(self, attention_state, flag):
        if isinstance(flag, torch.Tensor):
            g = flag
        else:
            flag = [0 if f is None else f for f in flag]
            g = torch.tensor(flag, dtype=attention_state.dtype, device=self._device)
        
        g = g.unsqueeze(1)
        # print(self._state[0].size(), g.size(), attention_state._state[0].size())
        state0 = self._state[0] * g + attention_state._state[0] * (1.0 - g)
        state1 = self._state[1] * g + attention_state._state[1] * (1.0 - g)

        if isinstance(flag, torch.Tensor):
            foo = flag.cpu().numpy().tolist()
        else:
            foo = flag

        names = [x if f > 0 else y for x, y, f in zip(self._name, attention_state._name, foo)]
        return BatchAttentionState(names, self._device, (state0, state1)).to(self.dtype)

    def expand(self, predicate_question_map):
        new_state0 = util.mm(predicate_question_map, self._state[0])
        new_state1 = util.mm(predicate_question_map, self._state[1])

        return BatchAttentionState(self._name, self._device, (new_state0, new_state1)).to(self.dtype)

    def squeeze(self, predicate_question_map):
        new_state0 = util.mm(predicate_question_map.transpose(0, 1), self._state[0])
        new_state1 = util.mm(predicate_question_map.transpose(0, 1), self._state[1])

        return BatchAttentionState(self._name, self._device, (new_state0, new_state1)).to(self.dtype)
