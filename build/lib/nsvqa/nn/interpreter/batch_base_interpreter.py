# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file
# in the project root for full license information.

import torch
import torch.nn as nn
import os

from operator import itemgetter
from nsvqa.nn.interpreter import util
from nsvqa.nn.interpreter.batch_base_types import BatchWorld, BatchVariableSet, BatchAttentionState
from nsvqa.nn.interpreter.data_parallel import gather_results

class BatchInterpreterBase(nn.Module):

    def __init__(self, name, oracle, featurizer=None, attention_transfer_state_dim=0, apply_modulation_everywhere=True, cached=False, visual_rule_learner=None, calibrator=None): #, attention_transfer_modulator=None):
        super(BatchInterpreterBase, self).__init__()
        self._featurizer = featurizer
        self._oracle = oracle
        self._name = name
        self._global_step = nn.Parameter(torch.tensor([0], dtype=torch.float), requires_grad=False)
        # self._atm = attention_transfer_modulator
        self._has_modulator = False
        self._attention_transfer_state_dim = attention_transfer_state_dim
        self._apply_modulation_everywhere = apply_modulation_everywhere
        self._cached = cached
        self._visual_rule_learner = visual_rule_learner
        self._calibrator = calibrator

    def _execute(self, op_id, world, operator_batch, input_tuple, is_terminal, is_training):
        pass

    def _transform_attention(self, op_id, is_forward, world, operator_batch, input_tuple, is_terminal, is_training):
        pass

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def save(self, export_path_base):
        torch.save(self.state_dict(), os.path.join(export_path_base, self._name))

    def load(self, import_path_base):
        self.load_state_dict(torch.load(os.path.join(import_path_base, self._name)), strict=False)

    def build_scene(self, device, object_features, batch_index, meta_data):
        
        if self._featurizer is not None:
            features = self._featurizer.featurize_scene(device, object_features, batch_index, meta_data)
            attribute_features = features['attribute_features']
            relation_features = features['relation_features']
            object_num = features['object_num']

            if self._cached:
                attribute_features, relation_features['features'] = self._oracle.compute_all_log_likelihood_2(attribute_features, relation_features['features'])

                if self._calibrator is not None:
                    attribute_features[:, self._oracle._ontology._attribute_index], relation_features = self._calibrator(attribute_features[:, self._oracle._ontology._attribute_index], relation_features)

                if self._visual_rule_learner is not None:
                    relation_features['object_num'] = object_num
                    attribute_features[:, self._oracle._ontology._attribute_index], relation_features = self._visual_rule_learner(attribute_features[:, self._oracle._ontology._attribute_index], relation_features)
        else:
            object_num = object_features.size()[0]
            attribute_features = object_features.view(object_num, -1)
            arg1 = attribute_features.repeat(1, object_num).view(object_num**2, -1)
            arg2 = attribute_features.repeat(self._object_num, 1)
            relation_features = torch.cat([arg1, arg2], dim=1)

        return BatchWorld(device, object_num, attribute_features, relation_features, batch_index, meta_data, \
            attention_transfer_state_dim=self._attention_transfer_state_dim).to(object_features.dtype)

    def forward(self, program_batch_list, is_training, return_trace=False, modulator_switch=True):

        # Initialize the trace
        all_traces = []
        all_results = []
        device = program_batch_list[0].device

        # Main loop
        for program_batch in program_batch_list:
            
            # Set the objects features
            world = self.build_scene(program_batch.device, program_batch._object_features, program_batch._object_batch_index, program_batch._meta_data)
            # print('---------------------------------------------')

            # Modulator loops
            if self._has_modulator and modulator_switch:
                if not self._apply_modulation_everywhere:
                    for i in range(len(program_batch._op_batch_list) - 1):
                        program_batch._op_batch_list._op_id += 'n'
                
                # Forward loop
                trace = []
                for i, op_batch in enumerate(program_batch._op_batch_list):
                    if len(program_batch._dependencies[i]) > 1:
                        input_tuple = tuple(itemgetter(*program_batch._dependencies[i])(trace)) 
                    elif len(program_batch._dependencies[i]) == 1:
                        input_tuple = (trace[program_batch._dependencies[i][0]],)
                    else: 
                        input_tuple = (None,)

                    x, terminate = self._transform_attention(op_batch._op_id, True, world, op_batch, input_tuple, i == len(program_batch._op_batch_list) - 1, is_training)
                    
                    # Gate the unaffected questions
                    if i < len(program_batch._op_batch_list) - 1 and input_tuple[0] is not None and op_batch._mask is not None:
                        x = x.gate(input_tuple[0], op_batch._mask)

                    trace.append(x)

                    if terminate:
                        break

                # Backward loop
                reversed_dependencies = util.reverse_dependencies(program_batch._dependencies)
                first_attention_state = (BatchAttentionState(trace[-1]._name, device, trace[-1]._state, set_zeros=True).to(world.dtype), ) if not isinstance(trace[-1], (tuple, list)) else \
                    tuple([BatchAttentionState(att._name, device, att._state, set_zeros=True).to(world.dtype) for att in trace[-1]])
                
                trace = [None for _ in range(len(program_batch._op_batch_list))]
                for i, op_batch in reversed(list(enumerate(program_batch._op_batch_list))):
                    if len(reversed_dependencies[i]) == 1:
                        temp = trace[reversed_dependencies[i][0]]

                        if isinstance(temp, (tuple, list)):
                            input_tuple = (temp[1],) if i == len(program_batch._op_batch_list) - 2 else (temp[0],)
                        else:
                            input_tuple = (temp,) 
                    else: 
                        input_tuple = first_attention_state

                    x, terminate = self._transform_attention(op_batch._op_id, False, world, op_batch, input_tuple, i == 0, is_training)
                    
                    # Gate the unaffected questions
                    # print(op_batch._op_name)
                    if len(program_batch._dependencies[i]) > 0 and op_batch._mask is not None and isinstance(x, BatchAttentionState) and i != len(program_batch._op_batch_list) - 1:
                        x = x.gate(input_tuple[0], op_batch._mask)

                    trace[i] = x

                    if terminate:
                        break

            # if self._atm is not None:
            #     attention_transfer = self._atm(program_batch)

            # Execution loop
            trace = []
            for i, op_batch in enumerate(program_batch._op_batch_list):
                # print(op_batch._op_name)
                if len(program_batch._dependencies[i]) > 1:
                    input_tuple = tuple(itemgetter(*program_batch._dependencies[i])(trace)) 
                elif len(program_batch._dependencies[i]) == 1:
                    input_tuple = (trace[program_batch._dependencies[i][0]],)
                else: 
                    input_tuple = ()

                x, terminate = self._execute(op_batch._op_id, world, op_batch, input_tuple, i == len(program_batch._op_batch_list) - 1, is_training)

                # # Apply the transfer function if available
                # if self._atm is not None and isinstance(x, BatchVariableSet):
                #     alpha = attention_transfer[i, :, 0].unsqueeze(1)
                #     beta = attention_transfer[i, :, 1].unsqueeze(1)
                #     temp = alpha * x._log_attention
                #     x._log_attention = temp - util.safe_log((beta * util.log_not(x._log_attention)).exp() + temp.exp())
                
                # Gate the unaffected questions
                if isinstance(x, BatchVariableSet) and len(input_tuple) > 0 and op_batch._mask is not None:
                    x = x.gate(input_tuple[0], op_batch._mask)

                trace.append(x)

                if terminate:
                    break

            result = trace[-1] if len(trace) > 0 else None
            all_results.append(result)
            all_traces.append(trace)

        result = gather_results(all_results, device, util.is_cuda(device))
        
        if return_trace:
            return result, all_traces 
        
        return result
