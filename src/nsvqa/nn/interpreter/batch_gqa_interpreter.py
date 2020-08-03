# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file
# in the project root for full license information.

import torch
import torch.nn as nn
import nsvqa.nn.interpreter.batch_gqa_ops as gqa

from nsvqa.nn.interpreter import util
from nsvqa.nn.interpreter.batch_base_types import BatchWorld
from nsvqa.nn.interpreter.batch_base_interpreter import BatchInterpreterBase

class BatchGQAInterpreter(BatchInterpreterBase):

    def __init__(self, name, oracle, ontology, featurizer=None, trainable_module_type=None, feature_dim=1, trainable_gate=False, likelihood_threshold=0, #attention_transfer_modulator=None, 
            hard_mode=False, attention_transfer_state_dim=0,\
            forward_attention_network=None, backward_attention_network=None, attention_output_network=None,\
            apply_modulation_everywhere=True, cached=False, visual_rule_learner=None, calibrator=None):
        super(BatchGQAInterpreter, self).__init__(name, oracle, featurizer, attention_transfer_state_dim=attention_transfer_state_dim, \
            apply_modulation_everywhere=apply_modulation_everywhere, cached=cached, visual_rule_learner=visual_rule_learner, calibrator=calibrator) #, attention_transfer_modulator=attention_transfer_modulator)
        self._ontology = ontology
        self._likelihood_threshold = likelihood_threshold
        self._hard_mode = hard_mode

        self._has_modulator = forward_attention_network is not None and backward_attention_network is not None and attention_output_network is not None

        self._ops = nn.ModuleDict({
            'select': gqa.GQASelectBatch(self._oracle, self._ontology, trainable_module_type=trainable_module_type, feature_dim=feature_dim, trainable_gate=trainable_gate, \
                forward_attention_network=forward_attention_network, backward_attention_network=backward_attention_network, attention_output_network=attention_output_network),
            'filter': gqa.GQAFilterBatch(self._oracle, self._ontology, trainable_module_type=trainable_module_type, feature_dim=feature_dim, trainable_gate=trainable_gate, \
                forward_attention_network=forward_attention_network, backward_attention_network=backward_attention_network, attention_output_network=attention_output_network),
            'relate': gqa.GQARelateBatch(self._oracle, self._ontology, trainable_module_type=trainable_module_type, feature_dim=feature_dim, trainable_gate=trainable_gate, \
                forward_attention_network=forward_attention_network, backward_attention_network=backward_attention_network, attention_output_network=attention_output_network),
            'query_attr': gqa.GQAQueryAttrBatch(self._oracle, self._ontology, trainable_module_type=trainable_module_type, feature_dim=feature_dim, trainable_gate=trainable_gate, \
                forward_attention_network=forward_attention_network, backward_attention_network=backward_attention_network, attention_output_network=attention_output_network),
            'choose_attr': gqa.GQAChooseAttrBatch(self._oracle, self._ontology, trainable_module_type=trainable_module_type, feature_dim=feature_dim, trainable_gate=trainable_gate, \
                forward_attention_network=forward_attention_network, backward_attention_network=backward_attention_network, attention_output_network=attention_output_network),
            # 'verify_attr': gqa.GQAVerifyAttrBatch(self._oracle, self._ontology, trainable_module_type=trainable_module_type, feature_dim=feature_dim, trainable_gate=trainable_gate, \
            # forward_attention_network=forward_attention_network, backward_attention_network=backward_attention_network, attention_output_network=attention_output_network),
            'verify_attrs': gqa.GQAVerifyAttrsBatch(self._oracle, self._ontology, trainable_module_type=trainable_module_type, feature_dim=feature_dim, trainable_gate=trainable_gate, \
                forward_attention_network=forward_attention_network, backward_attention_network=backward_attention_network, attention_output_network=attention_output_network),
            'choose_rel': gqa.GQAChooseRelBatch(self._oracle, self._ontology, trainable_module_type=trainable_module_type, feature_dim=feature_dim, trainable_gate=trainable_gate, \
                forward_attention_network=forward_attention_network, backward_attention_network=backward_attention_network, attention_output_network=attention_output_network),
            'verify_rel': gqa.GQAVerifyRelBatch(self._oracle, self._ontology, trainable_module_type=trainable_module_type, feature_dim=feature_dim, trainable_gate=trainable_gate, \
                forward_attention_network=forward_attention_network, backward_attention_network=backward_attention_network, attention_output_network=attention_output_network),
            'exist': gqa.GQAExistBatch(self._oracle, self._ontology),
            'and': gqa.GQAAndBatch(self._oracle, self._ontology),
            'or': gqa.GQAOrBatch(self._oracle, self._ontology),
            'all_same': gqa.GQAAllSameBatch(self._oracle, self._ontology, trainable_module_type=trainable_module_type, feature_dim=feature_dim, trainable_gate=trainable_gate, \
                forward_attention_network=forward_attention_network, backward_attention_network=backward_attention_network, attention_output_network=attention_output_network),
            'all_different': gqa.GQAAllDifferentBatch(self._oracle, self._ontology, trainable_module_type=trainable_module_type, feature_dim=feature_dim, trainable_gate=trainable_gate, \
                forward_attention_network=forward_attention_network, backward_attention_network=backward_attention_network, attention_output_network=attention_output_network),
            'two_same': gqa.GQATwoSameBatch(self._oracle, self._ontology, trainable_module_type=trainable_module_type, feature_dim=feature_dim, trainable_gate=trainable_gate, \
                forward_attention_network=forward_attention_network, backward_attention_network=backward_attention_network, attention_output_network=attention_output_network),
            'two_different': gqa.GQATwoDifferentBatch(self._oracle, self._ontology, trainable_module_type=trainable_module_type, feature_dim=feature_dim, trainable_gate=trainable_gate, \
                forward_attention_network=forward_attention_network, backward_attention_network=backward_attention_network, attention_output_network=attention_output_network),
            'compare': gqa.GQACompareBatch(self._oracle, self._ontology, trainable_module_type=trainable_module_type, feature_dim=feature_dim, trainable_gate=trainable_gate, \
                forward_attention_network=forward_attention_network, backward_attention_network=backward_attention_network, attention_output_network=attention_output_network),
            'end': gqa.GQAEndBatch(self._oracle, self._ontology),
            'object_attr': gqa.GQAObjectAttrBatch(self._oracle, self._ontology, trainable_module_type=trainable_module_type, feature_dim=feature_dim, trainable_gate=trainable_gate, \
                forward_attention_network=forward_attention_network, backward_attention_network=backward_attention_network, attention_output_network=attention_output_network),
            'object_rel': gqa.GQAObjectRelBatch(self._oracle, self._ontology, trainable_module_type=trainable_module_type, feature_dim=feature_dim, trainable_gate=trainable_gate, \
                forward_attention_network=forward_attention_network, backward_attention_network=backward_attention_network, attention_output_network=attention_output_network),
            'scene': gqa.GQASceneOpBatch(self._oracle, self._ontology)
        })

        self._ops_num = len(self._ops) - 3
        # self._ops_index = dict(zip(self._ops.keys(), [i for i in range(self._ops_num)]))
        self._ops_index = {'all_different': 0, 'all_same': 1, 'and': 2, 'choose_attr': 3, 'choose_rel': 4, 'compare': 5, 'end': 6, 'exist': 7, 'filter': 8, 'or': 9,\
            'query_attr': 10, 'relate': 11, 'select': 12, 'two_different': 13, 'two_same': 14, 'verify_attrs': 15, 'verify_rel': 16, 'object_attr': 3, 'object_rel': 4, 'scene': 6}

    def _execute(self, op_id, world, operator_batch, input_tuple, is_terminal, is_training):
        x = self._ops[operator_batch._op_name](*((op_id, world,) + input_tuple + tuple(operator_batch._arguments) + (not is_training, operator_batch._predicate_question_map, self._likelihood_threshold, self._hard_mode)))
        
        if is_terminal and not operator_batch._is_terminal: # Add the End op to calculate log-likelihood
            return self._ops['end'](op_id, world, x, not is_training, operator_batch._predicate_question_map), is_terminal

        return x, is_terminal

    def _transform_attention(self, op_id, is_forward, world, operator_batch, input_tuple, is_terminal, is_training):
        temp = torch.zeros(self._ops_num, device=world._device, dtype=world.dtype)
        temp[self._ops_index[operator_batch._op_name]] = 1.0
        # temp = torch.tensor([], device=world._device)
        
        x = self._ops[operator_batch._op_name].transform_attention(*((op_id, is_forward, world,) + input_tuple + tuple(operator_batch._arguments) + (temp, operator_batch._predicate_question_map)))
        return x, is_terminal
