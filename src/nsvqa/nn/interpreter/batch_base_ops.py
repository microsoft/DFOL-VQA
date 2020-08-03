# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file
# in the project root for full license information.

"The basic diffrentiable operators for the interpreter."

import numpy as np
import torch
import torch.nn as nn
from itertools import compress
from operator import itemgetter

from nsvqa.nn.interpreter import util
from nsvqa.nn.interpreter.batch_base_types import BatchVariableSet, Quantifier, TokenType, BatchAttentionState


######################################################################################################################################

class NeuralLogicGate(nn.Module):

    def __init__(self):
        super(NeuralLogicGate, self).__init__()
        self._linear = nn.Linear(2, 6)
        self._sigmoid = nn.Sigmoid()

    def forward(self, log_p, log_q):
        max_size = max(log_p.size(), log_q.size())
        lp = log_p.expand(max_size).contiguous()
        lq = log_q.expand(max_size).contiguous()

        alpha = torch.stack([lp.view(-1), lq.view(-1)]).transpose(0, 1).contiguous()
        alpha = self._sigmoid(self._linear(alpha))

        nlp = util.log_parametric_not(lp, alpha[:, 0].view(max_size), alpha[:, 3].view(max_size))
        nlq = util.log_parametric_not(lq, alpha[:, 1].view(max_size), alpha[:, 4].view(max_size))
        res = util.log_parametric_not(nlp + nlq, alpha[:, 2].view(max_size), alpha[:, 5].view(max_size))
        
        return res

######################################################################################################################################

class BatchBayesianLogicCell(nn.Module):
    
    def __init__(self, arity, trainable_module_type=None, feature_dim=1, trainable_gate=False):        
        super(BatchBayesianLogicCell, self).__init__()
        self._arity = arity
        self._trainable_gate = trainable_gate
        self._feature_dim = feature_dim
        self._relu = nn.ReLU()

        if trainable_gate:
            self._nlg = nn.ModuleList([NeuralLogicGate() for _ in range(arity)])

        temp = 1 - 2*np.identity(self._arity, dtype=np.int32)
        self._reshape_dim = [t.tolist() for t in temp]

        if trainable_module_type is not None:
            self._trainable_module = trainable_module_type(self._feature_dim)
        else:
            self._trainable_module = None

    def _forward_core(self, log_prior, log_likelihood, quantifiers, dim_order, batch_object_map, predicate_question_map):
        
        # log_prior is (question_num x arity x object_num)
        # log_likelihood is (predicate_num x object_num x ... x object_num)
        # quantifiers is (predicate_num x arity)
        # In most cases question_num == predicate_num otherwise predicate_question_map must be provided
        
        object_num = log_prior.size()[2]
        # batch_size = batch_object_map.size()[0]
        predicate_num = log_likelihood.size()[0]
        question_num = log_prior.size()[0]

        if predicate_question_map is not None and predicate_num != question_num:
            log_p = util.mm(predicate_question_map, log_prior.view(question_num, -1)).view(predicate_num, self._arity, -1)
        else:
            log_p = log_prior

        result = torch.zeros(predicate_num, self._arity, object_num, device=self._device, dtype=log_prior.dtype)

        if question_num > 1 and self._arity > 1:
            # Compute the indices of the diagonals of (question_num x ... x question_num)
            coeff = (question_num ** (self._arity - 1) - question_num) / (question_num - 1)
            ind = torch.arange(question_num, dtype=torch.int64, device=self._device)
            ind += (ind * coeff).long()

        if object_num > 1 and self._arity > 1:
            obj_diag_ind = np.arange(object_num, dtype=np.int64).tolist()

        for a in range(self._arity):
            i = dim_order[a] + 1
            log_posterior = log_likelihood

            for b in range(self._arity):
                j = dim_order[b] + 1
                
                if i != j:
                    # Multiply the prior
                    if self._trainable_gate:
                        log_posterior = self._nlg[j - 1](log_posterior, log_p[:, j - 1, :].view([predicate_num] + self._reshape_dim[j - 1]))
                    else:
                        log_posterior = log_posterior + log_p[:, j - 1, :].view([predicate_num] + self._reshape_dim[j - 1])
                    
                    if quantifiers[:, j - 1].numel() == 1:
                        if quantifiers[:, j - 1] == Quantifier.EXISTS:
                            log_posterior = util.safe_log(1.0 - util.safe_exp(log_posterior))
                    else:
                        log_posterior = util.log_parametric_not(log_posterior, quantifiers[:, j - 1].view([-1] + self._arity*[1]), 1)

                    # Discounting the diagonal part (self-relations)
                    # REVIEW: Only works for arity <= 2
                    log_posterior[:, obj_diag_ind, obj_diag_ind] = 0

                    s1 = log_posterior.size()
                    log_posterior = log_posterior.transpose(0, j)
                    s2 = list(log_posterior.size())

                    # if quantifiers[:, j - 1].numel() == 1:
                    #     if quantifiers[:, j - 1] == Quantifier.EXISTS:
                    #         log_posterior = util.safe_log(1.0 - util.safe_exp(log_posterior))
                    # else:
                    #     log_posterior = util.log_parametric_not(log_posterior, quantifiers[:, j - 1].view([1] + self._reshape_dim[j - 1]), 1)
                    
                    log_posterior = log_posterior.contiguous().view(s1[j], -1)
                    log_posterior = util.mm(batch_object_map, log_posterior)
                    s2[0] = question_num
                    log_posterior = log_posterior.view(s2).transpose(0, j)
                    
                    if quantifiers[:, j - 1].numel() == 1:
                        if quantifiers[:, j - 1] == Quantifier.EXISTS:
                            log_posterior = util.safe_log(1.0 - util.safe_exp(log_posterior))
                    else:
                        log_posterior = util.log_parametric_not(log_posterior, quantifiers[:, j - 1].view([-1] + self._arity*[1]), 1)

            if self._trainable_gate:
                log_posterior = self._nlg[i - 1](log_posterior, log_p[:, i - 1, :].view([predicate_num] + self._reshape_dim[i - 1]))
            else:
                log_posterior = log_posterior + log_p[:, i - 1, :].view([predicate_num] + self._reshape_dim[i - 1])
            
            log_posterior = log_posterior.transpose(1, i).contiguous().view(predicate_num, object_num, -1)

            if question_num > 1 and self._arity > 1:
                log_posterior = log_posterior[:, :, ind]
                mask = batch_object_map.transpose(0, 1).to_dense().unsqueeze(0)
                log_posterior = (log_posterior * mask).sum(dim=2)
            else:
                log_posterior = log_posterior.squeeze(2)

            result[:, i - 1, :] = log_posterior

        return result # predicate_num x arity x object_num

    def forward(self, log_prior, log_likelihood, quantifiers, dim_order, batch_object_map=None, \
                predicate_question_map=None, is_negated=None, default_log_likelihood=-30):

        # log_prior is (question_num x arity x object_num)
        # log_likelihood is (predicate_num x object_num x ... x object_num x feature_num) or a dictionary

        assert quantifiers.size()[1] == self._arity, "The number of quantifiers must match the arity of the operator."
        assert len(dim_order) == self._arity, "The number of dimension order elements must match the arity of the operator."
        assert log_prior.size()[1] == self._arity, "The second dimension of log-prior must be equal to the arity of the operator."
        object_num = log_prior.size()[2]

        if isinstance(log_likelihood, torch.Tensor):
            assert log_likelihood.dim() == self._arity + 2, "The number of dimensions of log-likelihood must be equal to the arity of the operator + 2."
            assert log_prior.size()[0] == log_likelihood.size()[0] or (predicate_question_map is not None and \
                    log_likelihood.size()[0] == predicate_question_map.size()[0] and log_prior.size()[0] == predicate_question_map.size()[1]), \
                    "In case predicate_num != question_num, predicate_question_map of size (predicate_num, question_num) must be provided."

            assert (log_likelihood.dim() - 2) * [object_num] == list(log_likelihood.size())[1:-1], "The number of objects must be consistent between the log-prior and the log-likelihood."

            self._device = log_likelihood.device
        else:
            self._device = log_likelihood['feature'].device
        
        if batch_object_map is None:   
            batch_index = torch.zeros(object_num, dtype=torch.int64, device=self._device)
            all_ones = torch.ones(object_num, device=self._device, dtype=log_prior.dtype)
            y_ind = torch.arange(object_num, dtype=torch.int64, device=self._device)
            ind = torch.stack([batch_index, y_ind])
            
            if self._device.type == 'cuda':
                batch_object_map = torch.cuda.sparse.FloatTensor(ind, all_ones, 
                                torch.Size([1, object_num]), device=self._device)
            else:
                batch_object_map = torch.sparse.FloatTensor(ind, all_ones, 
                                torch.Size([1, object_num]), device=self._device)

        if isinstance(log_likelihood, torch.Tensor):
            if self._trainable_module is not None:
                ll = self._trainable_module(log_likelihood.view(-1, self._feature_dim))
                ll = ll.view(log_likelihood.size()[:-1])
            else:
                ll = -self._relu(-log_likelihood.mean(dim=log_likelihood.dim()-1))
        else: # Dictionary for sparse likelihood
            if self._trainable_module is not None:
                f = self._trainable_module(log_likelihood['feature']).flatten()
            else:
                f = -self._relu(-log_likelihood['feature'].mean(dim=1))

            temp = default_log_likelihood * torch.ones(log_likelihood['size'][0], log_likelihood['size'][1], device=self._device, dtype=log_prior.dtype)
            temp[log_likelihood['index'][0], log_likelihood['index'][1]] = f

            if self._arity == 1:
                ll = temp
            elif self._arity == 2:
                ll = default_log_likelihood * torch.ones(log_likelihood['size'][0], log_likelihood['size'][2], log_likelihood['size'][3], device=self._device, dtype=log_prior.dtype)
                ll[:, log_likelihood['index'][2], log_likelihood['index'][3]] = temp
            else:
                raise NotImplementedError("Likelihood in dictionary format is not implemented for arity > 2.")

        if is_negated is not None:
            ll = util.log_parametric_not(ll, is_negated.view([-1] + self._arity * [1]), 1)

        return self._forward_core(log_prior, ll, quantifiers, dim_order, batch_object_map, predicate_question_map)

    def compute_total_log_probability(self, log_posterior, quantifier, batch_object_map, predicate_question_map):
        if log_posterior.dim() < 3:
            log_posterior = log_posterior.unsqueeze(0)

        if quantifier.dim() < 2:
            quantifier = quantifier.unsqueeze(0)

        predicate_num, arity, object_num = log_posterior.size()
        log_posterior = util.log_parametric_not(log_posterior, quantifier.unsqueeze(2), 1)

        if predicate_question_map is not None:
            log_posterior = util.mm(predicate_question_map, util.mm(batch_object_map, log_posterior.transpose(0, 2).contiguous().view(object_num, -1)))
        else:
            log_posterior = util.mm(batch_object_map, log_posterior.transpose(0, 2).contiguous().view(object_num, -1))
        
        ind = torch.arange(predicate_num, dtype=torch.int64)
        log_posterior = log_posterior.view(predicate_num, arity, predicate_num)[ind, :, ind] # predicate_num x arity

        log_posterior = util.log_parametric_not(log_posterior, quantifier, 1)

        return  # predicate_num x arity

######################################################################################################################################

class BatchOperatorBase(nn.Module):
    
    def __init__(self, oracle, is_terminal, fan_in, fan_out, \
            forward_attention_network=None, backward_attention_network=None, attention_output_network=None):
        super(BatchOperatorBase, self).__init__()
        self._oracle = oracle
        self._is_terminal = is_terminal
        self._fan_in = fan_in
        self._fan_out = fan_out

        if forward_attention_network is not None and backward_attention_network is not None and attention_output_network is not None:
            self._forward_attention_network = forward_attention_network
            self._backward_attention_network = backward_attention_network
            self._attention_output_network = attention_output_network
    
    def is_terminal(self):
        return self._is_terminal

    def fan_in(self):
        return self._fan_in

    def fan_out(self):
        return self._fan_out

    def _get_features(self, world, token_list, op_features):
        result = self._oracle.get_embedding(token_list, world._meta_data, world._device)
        # ind = itemgetter(*token_list)(world._meta_data['index'])
        # result = world._meta_data['embedding'][ind, :]
        temp = op_features.repeat(len(token_list), 1) if len(token_list) > 1 else op_features.unsqueeze(0)
        if result.dim() < 2:
            result = result.unsqueeze(0)
        
        return torch.cat([temp, result], dim=1)

    def _compute_attention_modulations(self, forward_state, backward_state):
        if forward_state is None:
            fs = torch.zeros_like(backward_state[0])
        else:
            fs = forward_state[0]

        if backward_state is None:
            bs = torch.zeros_like(forward_state[0])
        else:
            bs = backward_state[0]

        return self._attention_output_network(torch.cat([fs, bs], dim=1))

######################################################################################################################################

class SelectBatch(BatchOperatorBase):

    def __init__(self, oracle, forward_attention_network=None, backward_attention_network=None, attention_output_network=None):
        super(SelectBatch, self).__init__(oracle, is_terminal=False, fan_in=0, fan_out=1, 
            forward_attention_network=forward_attention_network, backward_attention_network=backward_attention_network, attention_output_network=attention_output_network)

    def forward(self, id, world, name, quantifier=Quantifier.EXISTS):
        return world.variable_set(name, quantifier=quantifier)

######################################################################################################################################

class FilterBatch(BatchOperatorBase):

    def __init__(self, oracle, trainable_module_type=None, feature_dim=1, trainable_gate=False,\
            forward_attention_network=None, backward_attention_network=None, attention_output_network=None):
        super(FilterBatch, self).__init__(oracle, is_terminal=False, fan_in=1, fan_out=1, 
            forward_attention_network=forward_attention_network, backward_attention_network=backward_attention_network, attention_output_network=attention_output_network)
        self._blc = BatchBayesianLogicCell(arity=1, trainable_module_type=trainable_module_type, feature_dim=feature_dim, trainable_gate=trainable_gate)
        self._modulations = {}
        self._forward_state = {}
    
    def forward(self, op_id, world, variable_set, attribute_list, predicate_question_map=None, default_log_likelihood=-30, normalized_probability=True):        
        if not isinstance(attribute_list, list):
            attribute_list = [attribute_list]
        
        ind = [val is not None and val.strip() not in ['', '_'] for val in attribute_list]
        if not any(ind):
            return variable_set

        device = variable_set.device

        question_num = variable_set.batch_size()
        predicate_num = len(attribute_list)
        
        if predicate_question_map is not None and isinstance(predicate_question_map, (list, tuple)):
            all_ones = torch.ones(predicate_num, device=device, dtype=variable_set.dtype)
            x_ind = torch.arange(predicate_num, dtype=torch.int64, device=device)
            y_ind = torch.tensor(predicate_question_map, dtype=torch.int64, device=device)
            index = torch.stack([x_ind, y_ind])
            
            if util.is_cuda(device):
                predicate_question_map = torch.cuda.sparse.FloatTensor(index, all_ones, 
                                torch.Size([predicate_num, question_num]), device=device)
            else:
                predicate_question_map = torch.sparse.FloatTensor(index, all_ones, 
                                torch.Size([predicate_num, question_num]), device=device)

        assert (question_num == predicate_num) or \
            (predicate_question_map is not None and predicate_question_map.size() == torch.Size([predicate_num, question_num])),\
            "Batch size mismatch."

        quantifier = variable_set._quantifier.unsqueeze(1)
        if predicate_question_map is not None:
            quantifier = util.mm(predicate_question_map, quantifier)

        if not all(ind):
            attribute_list = list(compress(attribute_list, ind))

        is_any_negated, is_neg, a_list = util.detect_negations(attribute_list, device)
        is_neg = torch.tensor(is_neg, dtype=variable_set.dtype, device=device)

        if predicate_question_map is not None:
            attribute_image_map = predicate_question_map._indices()[1, ind]
        else:
            attribute_image_map = torch.arange(predicate_num, dtype=torch.int64, device=device)[ind]

        log_likelihood = self._oracle(TokenType.ATTRIBUTE, a_list, attribute_image_map, world, default_log_likelihood=default_log_likelihood, normalized_probability=normalized_probability)

        if isinstance(log_likelihood, torch.Tensor):
            if log_likelihood.dim() < 3:
                log_likelihood = log_likelihood.unsqueeze(0)

        if not all(ind):
            if isinstance(log_likelihood, torch.Tensor):
                ll = default_log_likelihood * torch.ones(predicate_num, log_likelihood.size()[1], log_likelihood.size()[2], device=device, dtype=variable_set.dtype)

                if any(ind):
                    ind = torch.tensor(ind, dtype=torch.bool, device=device)
                    ll[ind, :, :] = log_likelihood
            else:
                ll = log_likelihood
                if any(ind):
                    seq = torch.arange(predicate_num, dtype=torch.int64, device=device)
                    ll['index'][0] = seq[ind][ll['index'][0]]
                    ll['size'][0] = predicate_num

            if is_any_negated:
                is_negated = torch.zeros(predicate_num, device=device, dtype=variable_set.dtype)
                is_negated[ind] = is_neg
            else:
                is_negated = None

            log_attention = self._blc(variable_set._log_attention.unsqueeze(1), ll, quantifier, [0], 
                variable_set._batch_object_map, predicate_question_map, is_negated, default_log_likelihood=default_log_likelihood)

            log_attention[ind == False, 0, :] = variable_set._log_attention[ind == False, :]
        else:
            is_negated = is_neg if is_any_negated else None

            log_attention = self._blc(variable_set._log_attention.unsqueeze(1), log_likelihood, quantifier, [0], 
                variable_set._batch_object_map, predicate_question_map, is_negated, default_log_likelihood=default_log_likelihood)

        if predicate_question_map is None:
            quantifier = variable_set._quantifier
        else:
            quantifier = util.mm(predicate_question_map, variable_set._quantifier.unsqueeze(1)).squeeze(1)

        res = BatchVariableSet(variable_set._name, variable_set._device, variable_set.object_num(), predicate_num, quantifiers=quantifier, \
            log_attention=log_attention[:, 0, :], batch_object_map=variable_set._batch_object_map, predicate_question_map=predicate_question_map, \
            base_cumulative_loss=variable_set.cumulative_loss(), prev_variable_sets_num=variable_set._prev_variable_sets_num + 1).to(variable_set.dtype)

        if op_id in self._modulations:
            res = res.apply_modulations(self._modulations[op_id], variable_set, predicate_question_map).to(variable_set.dtype)
            self._modulations.pop(op_id, 'No Key found')
        
        return res

    def transform_attention(self, op_id, is_forward, world, attention_state, attribute_list, op_feature, predicate_question_map=None):
        if not isinstance(attribute_list, list):
            attribute_list = [attribute_list]
        
        ind = [val is not None and val.strip() not in ['', '_'] for val in attribute_list]
        if not any(ind):
            return attention_state

        device = attention_state.device

        question_num = world.batch_size()
        predicate_num = len(attribute_list)
        
        if predicate_question_map is not None and isinstance(predicate_question_map, (list, tuple)):
            all_ones = torch.ones(predicate_num, device=device, dtype=attention_state.dtype)
            x_ind = torch.arange(predicate_num, dtype=torch.int64, device=device)
            y_ind = torch.tensor(predicate_question_map, dtype=torch.int64, device=device)
            index = torch.stack([x_ind, y_ind])
            
            if util.is_cuda(device):
                predicate_question_map = torch.cuda.sparse.FloatTensor(index, all_ones, 
                                torch.Size([predicate_num, question_num]), device=device)
            else:
                predicate_question_map = torch.sparse.FloatTensor(index, all_ones, 
                                torch.Size([predicate_num, question_num]), device=device)

        assert (question_num == predicate_num) or \
            (predicate_question_map is not None and predicate_question_map.size() == torch.Size([predicate_num, question_num])),\
            "Batch size mismatch."

        if not all(ind):
            attribute_list = list(compress(attribute_list, ind))

        _, _, a_list = util.detect_negations(attribute_list, device)

        if not all(ind):
            features = torch.zeros(predicate_num, world.word_embedding_dim() + op_feature.size()[0] + 1, device=device, dtype=attention_state.dtype)
            features[ind, :] = self._get_features(world, a_list, torch.cat([op_feature, torch.zeros(1, device=device, dtype=attention_state.dtype)], dim=0))
        else:
            features = self._get_features(world, a_list, torch.cat([op_feature, torch.zeros(1, device=device, dtype=attention_state.dtype)], dim=0))

        if is_forward:
            if predicate_question_map is not None:
                old_attention_state = attention_state.expand(predicate_question_map)
            else:
                old_attention_state = attention_state

            new_state_tuple = self._forward_attention_network(features, old_attention_state._state)
            new_state = BatchAttentionState(attention_state._name, device, new_state_tuple).to(attention_state.dtype)
            self._forward_state[op_id] = new_state_tuple
        else:
            if op_id[-1] != 'n':
                self._modulations[op_id] = self._compute_attention_modulations(self._forward_state[op_id], attention_state._state)
            self._forward_state.pop(op_id, 'No Key found')
            new_state_tuple = self._backward_attention_network(features, attention_state._state)
            new_state = BatchAttentionState(attention_state._name, device, new_state_tuple).to(attention_state.dtype)
            
            if predicate_question_map is not None:
                new_state = new_state.squeeze(predicate_question_map)

        return new_state

######################################################################################################################################

class RelateBatch(BatchOperatorBase):

    def __init__(self, oracle, trainable_module_type=None, feature_dim=1, trainable_gate=False,\
            forward_attention_network=None, backward_attention_network=None, attention_output_network=None):
        super(RelateBatch, self).__init__(oracle, is_terminal=False, fan_in=2, fan_out=2, 
            forward_attention_network=forward_attention_network, backward_attention_network=backward_attention_network, attention_output_network=attention_output_network)
        self._blc = BatchBayesianLogicCell(arity=2, trainable_module_type=trainable_module_type, feature_dim=feature_dim, trainable_gate=trainable_gate)
        self._subject_modulations = {}
        self._object_modulations = {}
        self._forward_subject_state = {}
        self._forward_object_state = {}

    def forward(self, op_id, world, subject_variable_set, object_variable_set, relation_list, predicate_question_map=None, default_log_likelihood=-30, normalized_probability=True):
        assert subject_variable_set.batch_size() == object_variable_set.batch_size(), "The subject and object variable sets must have the same batch size."
        assert subject_variable_set.object_num() == object_variable_set.object_num(), "The subject and object variable sets must have the same number of objects."

        if not isinstance(relation_list, list):
            relation_list = [relation_list]
        
        ind = [val is not None and val.strip() not in ['', '_'] for val in relation_list]
        if not any(ind):
            return subject_variable_set, object_variable_set

        device = subject_variable_set.device

        question_num = world.batch_size()
        predicate_num = len(relation_list)

        if predicate_question_map is not None and isinstance(predicate_question_map, (list, tuple)):
            all_ones = torch.ones(predicate_num, device=device, dtype=subject_variable_set.dtype)
            x_ind = torch.arange(predicate_num, dtype=torch.int64, device=device)
            y_ind = torch.tensor(predicate_question_map, dtype=torch.int64, device=device)
            index = torch.stack([x_ind, y_ind])
            
            if util.is_cuda(device):
                predicate_question_map = torch.cuda.sparse.FloatTensor(index, all_ones, 
                                torch.Size([predicate_num, question_num]), device=device)
            else:
                predicate_question_map = torch.sparse.FloatTensor(index, all_ones, 
                                torch.Size([predicate_num, question_num]), device=device)

        assert (question_num == predicate_num) or \
            (predicate_question_map is not None and predicate_question_map.size() == torch.Size([predicate_num, question_num])),\
            "Batch size mismatch."

        log_attention = torch.stack([subject_variable_set._log_attention, object_variable_set._log_attention]).transpose(0, 1).contiguous()
        dim_order = [0, 1]
        quantifier = torch.stack([subject_variable_set._quantifier, object_variable_set._quantifier]).transpose(0, 1).contiguous()

        if predicate_question_map is not None:
            quantifier = util.mm(predicate_question_map, quantifier)

        if not all(ind):
            relation_list = list(compress(relation_list, ind))

        is_any_negated, is_neg, r_list = util.detect_negations(relation_list, device)
        is_neg = torch.tensor(is_neg, dtype=subject_variable_set.dtype, device=device)
        
        if predicate_question_map is not None:
            relation_image_map = predicate_question_map._indices()[1, ind]
        else:
            relation_image_map = torch.arange(predicate_num, dtype=torch.int64, device=device)[ind]

        log_likelihood = self._oracle(TokenType.RELATION, r_list, relation_image_map, world, default_log_likelihood=default_log_likelihood, normalized_probability=normalized_probability)

        if isinstance(log_likelihood, torch.Tensor):
            if log_likelihood.dim() < 4:
                log_likelihood = log_likelihood.unsqueeze(0)

        if not all(ind):
            if isinstance(log_likelihood, torch.Tensor):
                ll = default_log_likelihood * torch.ones(predicate_num, log_likelihood.size()[1], log_likelihood.size()[2], log_likelihood.size()[3], device=device, dtype=subject_variable_set.dtype)
                
                if any(ind):
                    ind = torch.tensor(ind, dtype=torch.bool, device=device)
                    ll[ind, :, :, :] = log_likelihood
            else:
                ll = log_likelihood
                if any(ind):
                    seq = torch.arange(predicate_num, dtype=torch.int64, device=device)
                    ll['index'][0] = seq[ind][ll['index'][0]]
                    ll['size'][0] = predicate_num

            if is_any_negated:
                is_negated = torch.zeros(predicate_num, device=device, dtype=subject_variable_set.dtype)
                is_negated[ind] = is_neg
            else:
                is_negated = None

            log_attention_posterior = self._blc(log_attention, ll, quantifier, dim_order, 
                        subject_variable_set._batch_object_map, predicate_question_map, is_negated, default_log_likelihood=default_log_likelihood)

            log_attention_posterior[ind == False, 0, :] = subject_variable_set._log_attention[ind == False, :]
            log_attention_posterior[ind == False, 1, :] = object_variable_set._log_attention[ind == False, :]
        else:        
            is_negated = is_neg if is_any_negated else None

            log_attention_posterior = self._blc(log_attention, log_likelihood, quantifier, dim_order, 
                        subject_variable_set._batch_object_map, predicate_question_map, is_negated, default_log_likelihood=default_log_likelihood)

        if predicate_question_map is None:
            quantifier = subject_variable_set._quantifier
        else:
            quantifier = util.mm(predicate_question_map, subject_variable_set._quantifier.unsqueeze(1)).squeeze(1)

        new_subject_set = BatchVariableSet(subject_variable_set._name, subject_variable_set._device, 
            subject_variable_set.object_num(), predicate_num, quantifiers=quantifier, 
            log_attention=log_attention_posterior[:, 0, :], batch_object_map=subject_variable_set._batch_object_map,
            predicate_question_map=predicate_question_map, base_cumulative_loss=subject_variable_set.cumulative_loss() + object_variable_set.cumulative_loss(),
            prev_variable_sets_num=subject_variable_set._prev_variable_sets_num + object_variable_set._prev_variable_sets_num + 1).to(subject_variable_set.dtype)

        new_object_set = BatchVariableSet(object_variable_set._name, object_variable_set._device, 
            object_variable_set.object_num(), predicate_num, quantifiers=quantifier, 
            log_attention=log_attention_posterior[:, 1, :], batch_object_map=object_variable_set._batch_object_map,
            predicate_question_map=predicate_question_map, base_cumulative_loss=subject_variable_set.cumulative_loss() + object_variable_set.cumulative_loss(),
            prev_variable_sets_num=subject_variable_set._prev_variable_sets_num + object_variable_set._prev_variable_sets_num + 1)
        
        if op_id in self._subject_modulations:
            new_subject_set = new_subject_set.apply_modulations(self._subject_modulations[op_id], subject_variable_set, predicate_question_map).to(subject_variable_set.dtype)
            self._subject_modulations.pop(op_id, 'No Key found')

        if op_id in self._object_modulations:
            new_object_set = new_object_set.apply_modulations(self._object_modulations[op_id], object_variable_set, predicate_question_map).to(object_variable_set.dtype)
            self._object_modulations.pop(op_id, 'No Key found')

        return new_subject_set, new_object_set

    def transform_attention(self, op_id, is_forward, world, subject_attention_state, object_attention_state, relation_list, op_feature, predicate_question_map=None):        
        if not isinstance(relation_list, list):
            relation_list = [relation_list]
        
        ind = [val is not None and val.strip() not in ['', '_'] for val in relation_list]
        if not any(ind):
            return subject_attention_state, object_attention_state

        device = subject_attention_state.device

        question_num = world.batch_size()
        predicate_num = len(relation_list)

        if predicate_question_map is not None and isinstance(predicate_question_map, (list, tuple)):
            all_ones = torch.ones(predicate_num, device=device, dtype=subject_attention_state.dtype)
            x_ind = torch.arange(predicate_num, dtype=torch.int64, device=device)
            y_ind = torch.tensor(predicate_question_map, dtype=torch.int64, device=device)
            index = torch.stack([x_ind, y_ind])
            
            if util.is_cuda(device):
                predicate_question_map = torch.cuda.sparse.FloatTensor(index, all_ones, 
                                torch.Size([predicate_num, question_num]), device=device)
            else:
                predicate_question_map = torch.sparse.FloatTensor(index, all_ones, 
                                torch.Size([predicate_num, question_num]), device=device)

        assert (question_num == predicate_num) or \
            (predicate_question_map is not None and predicate_question_map.size() == torch.Size([predicate_num, question_num])),\
            "Batch size mismatch."

        if not all(ind):
            relation_list = list(compress(relation_list, ind))

        _, _, r_list = util.detect_negations(relation_list, device)

        if not all(ind):
            features = torch.zeros(predicate_num, world.word_embedding_dim() + op_feature.size()[0] + 1, device=device, dtype=subject_attention_state.dtype)
            features[ind, :] = self._get_features(world, r_list, torch.cat([op_feature, torch.ones(1, device=device, dtype=subject_attention_state.dtype)], dim=0))
        else:
            features = self._get_features(world, r_list, torch.cat([op_feature, torch.ones(1, device=device, dtype=subject_attention_state.dtype)], dim=0))

        if is_forward:
            if predicate_question_map is not None:
                old_subject_attention_state = subject_attention_state.expand(predicate_question_map)
                old_object_attention_state = object_attention_state.expand(predicate_question_map)
            else:
                old_subject_attention_state = subject_attention_state
                old_object_attention_state = object_attention_state

            new_state_tuple = self._forward_attention_network(features, (old_subject_attention_state._state[0] + old_object_attention_state._state[0],\
                old_subject_attention_state._state[1] + old_object_attention_state._state[1]))
            new_subject_state = BatchAttentionState(subject_attention_state._name, device, new_state_tuple).to(subject_attention_state.dtype)

            h = torch.zeros_like(new_state_tuple[0])
            h.copy_(new_state_tuple[0])
            c = torch.zeros_like(new_state_tuple[1])
            c.copy_(new_state_tuple[1])

            new_object_state = BatchAttentionState(object_attention_state._name, device, (h, c)).to(object_attention_state.dtype)
            self._forward_subject_state[op_id] = new_state_tuple
            self._forward_object_state[op_id] = (h, c)
        else:
            if op_id[-1] != 'n':
                self._subject_modulations[op_id] = self._compute_attention_modulations(self._forward_subject_state[op_id], subject_attention_state._state)
                self._object_modulations[op_id] = self._compute_attention_modulations(self._forward_object_state[op_id], object_attention_state._state)
            
            self._forward_subject_state.pop(op_id, 'No Key found')
            self._forward_object_state.pop(op_id, 'No Key found')

            aggregate_state = (subject_attention_state._state[0] + object_attention_state._state[0],\
                subject_attention_state._state[1] + object_attention_state._state[1])
            
            new_state_tuple = self._backward_attention_network(features, aggregate_state)
            new_subject_state = BatchAttentionState(subject_attention_state._name, device, new_state_tuple).to(subject_attention_state.dtype)

            h = torch.zeros_like(new_state_tuple[0])
            h.copy_(new_state_tuple[0])
            c = torch.zeros_like(new_state_tuple[1])
            c.copy_(new_state_tuple[1])

            new_object_state = BatchAttentionState(object_attention_state._name, device, (h, c)).to(object_attention_state.dtype)
            
            if predicate_question_map is not None:
                new_subject_state = new_subject_state.squeeze(predicate_question_map)
                new_object_state = new_object_state.squeeze(predicate_question_map)

        return new_subject_state, new_object_state
