# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file
# in the project root for full license information.

"The Batch GQA operators."

import torch
import torch.nn as nn
import json, re, math
import linecache
import numpy as np

from collections import defaultdict
from itertools import repeat, compress
from operator import itemgetter

from nsvqa.nn.interpreter import util
from nsvqa.nn.interpreter.batch_base_ops import BatchOperatorBase, FilterBatch, RelateBatch
from nsvqa.nn.interpreter.batch_base_types import Quantifier, BatchVariableSet, QuestionType

UNKNOWN = 'UNKNOWN'

######################################################################################################################################

class GQAOntology(object):

    def __init__(self, attribute_json_path, class_json_path, vocab_json_file, embedding_file=None, relation_json_path=None, frequency_json_path=None):
        self._attribute_dict = json.load(open(attribute_json_path))
        self._class_dict = json.load(open(class_json_path))
        self._nouns = list(set(sum(self._class_dict.values(), [])))
        self._adjectives = list(set(sum(self._attribute_dict.values(), [])))
        
        if frequency_json_path is not None:
            self._frequencies = json.load(open(frequency_json_path))
        
        temp = [list(zip(repeat(o), l)) for o, l in self._class_dict.items()]
        temp = [item for sublist in temp for item in sublist]
        self._inverted_class_dict = defaultdict(list)
        {self._inverted_class_dict[v].append(k) for k, v in temp}

        self._embedding_file = embedding_file
        self._word_index = dict()

        if embedding_file is not None:
            with open(embedding_file, 'r', encoding="utf8") as f:
                for i, line in enumerate(f):
                    self._word_index[line.split(' ')[0]] = i

            line = linecache.getline(self._embedding_file, 1)
            self._embedding_dim = len(line.split(' ')) - 1

        with open(vocab_json_file, 'r') as f:
            self._vocabulary = json.load(f)

        self._noun_index = sorted([self._vocabulary['arg_to_idx'][n] - 1 for n in self._nouns if n in self._vocabulary['arg_to_idx']])

        if relation_json_path is not None:
            self._relations = list(set(json.load(open(relation_json_path))))
            self._relation_index = sorted([self._vocabulary['arg_to_idx'][rel] - 1 for rel in self._relations if rel in self._vocabulary['arg_to_idx']])
            self._attribute_index = list(set(range(len(self._vocabulary['arg_to_idx']))) - set(self._relation_index))
            self._attributes = [self._vocabulary['idx_to_arg'][i] for i in self._attribute_index]
            self._relation_reveresed_index = {i: j for j, i in enumerate(self._relation_index)}
            self._attribute_reveresed_index = {i: j for j, i in enumerate(self._attribute_index)}

            self._noun_subindex = sorted([j for j, i in enumerate(self._attribute_index) if self._vocabulary['idx_to_arg'][i] in self._nouns])
            self._non_noun_subindex = list(set(range(len(self._attribute_index))) - set(self._noun_subindex))

    def get_family_subindex(self, attribute):
        if attribute not in self._inverted_class_dict:
            return []

        children = [self._class_dict[parent] for parent in self._inverted_class_dict[attribute]]
        children = list(set(list(sum(children, []))))
        return [j for j, a in enumerate(self._attributes) if a in children]

    def encode_token(self, token):
        t = str(token)
        is_negated = (re.match("not\((\w|\s)+\)", t.lower().strip()) is not None)
        
        if is_negated:
            t = t.lower().strip()[4:-1]
        else:
            t = t.lower().strip()

        return (-1 if is_negated else 1) * self._vocabulary['arg_to_idx'][t]

    def decode_token(self, idx):
        t = self._vocabulary['idx_to_arg'][np.abs(idx) - 1]
        if t == 'true':
            return True
        elif t == 'false':
            return False
        else:
            return t if idx >= 0 else 'not(' + t + ')' 

    def encode_op(self, op):
        return self._vocabulary['op_to_idx'][op.lower().strip()]

    def decode_op(self, idx):
        return self._vocabulary['idx_to_op'][idx - 1]

    def encode_img_id(self, img_id):
        return self._vocabulary['img_to_idx'][img_id.lower().strip()]

    def decode_img_id(self, idx):
        return self._vocabulary['idx_to_img'][idx - 1]

    def query_attribute(self, attr_name):
        return self._attribute_dict[attr_name] if attr_name in self._attribute_dict else UNKNOWN

    def query_class(self, class_name):
        return self._class_dict[class_name] if class_name in self._class_dict else UNKNOWN

    def query(self, name):
        if name in self._attribute_dict:
            return self._attribute_dict[name]
        elif name in self._class_dict:
            return self._class_dict[name]
        elif name is None:
            return [None]
        elif name == 'entity':
            return self._nouns
        else:
            return [name]

    def is_noun(self, name):
        return name in self._nouns

    def is_adjective(self, name):
        return name in self._adjectives

    def is_relation(self, name):
        return name in self._relations

    def get_embeddings(self, names):
        if self._embedding_file is None:
            return None
        
        res = np.zeros((len(names), self._embedding_dim), dtype=np.float32)
        for i, name in enumerate(names):
            tokens = name.split(' ')
            
            for t in tokens:
                if t in self._word_index:
                    line = linecache.getline(self._embedding_file, self._word_index[t] + 1)
                    res[i, :] += np.array([float(t) for t in line.split(' ')[1:]])

        return res

######################################################################################################################################

class GQABatchOperatorBase(BatchOperatorBase):
    
    def __init__(self, oracle, ontology, is_terminal, fan_in, fan_out):
        super(GQABatchOperatorBase, self).__init__(oracle, is_terminal, fan_in, fan_out)
        self._ontology = ontology

######################################################################################################################################

class GQASelectBatch(GQABatchOperatorBase):

    def __init__(self, oracle, ontology, trainable_module_type=None, feature_dim=1, trainable_gate=False, \
        forward_attention_network=None, backward_attention_network=None, attention_output_network=None):
        super(GQASelectBatch, self).__init__(oracle, ontology, is_terminal=False, fan_in=0, fan_out=1)
        self._filter = FilterBatch(oracle, trainable_module_type=trainable_module_type, feature_dim=feature_dim, trainable_gate=trainable_gate, 
            forward_attention_network=forward_attention_network, backward_attention_network=backward_attention_network, attention_output_network=attention_output_network)

    def forward(self, op_id, world, attribute_list=None, give_answer=True, predicate_question_map=None, likelihood_threshold=0, hard_mode=False):
        batch_size = world.batch_size()

        if attribute_list is None:
            name = ["entity" for _ in range(batch_size)]
            att = None
        else:
            name = ["entity" if a is None or a.lower() in ["_", "scene"] else a for a in attribute_list]
            name = name[:batch_size]
            att = [None if a is None or a.lower() in ["_", "scene"] else a for a in attribute_list]

            att = att[:batch_size]
            all_nones = (att == batch_size*[None])

        x = world.variable_set(name, quantifier=Quantifier.EXISTS)
        return x if att is None or all_nones else self._filter(op_id, world, x, att)

    def transform_attention(self, op_id, is_forward, world, attention_state, attribute_list, op_feature, predicate_question_map=None):
        batch_size = world.batch_size()

        if attribute_list is None:
            name = ["entity" for _ in range(batch_size)]
            att = None
        else:
            name = ["entity" if a is None or a.lower() in ["_", "scene"] else a for a in attribute_list]
            name = name[:batch_size]
            att = [None if a is None or a.lower() in ["_", "scene"] else a for a in attribute_list]

            att = att[:batch_size]
            all_nones = (att == batch_size*[None])

        if is_forward:
            x = world.attention_state(name)
            return x if att is None or all_nones else self._filter.transform_attention(op_id, is_forward, world, x, att, op_feature)
        else:
            return attention_state if att is None or all_nones else self._filter.transform_attention(op_id, is_forward, world, attention_state, att, op_feature)

######################################################################################################################################

class GQAChooseAttrBatch(GQABatchOperatorBase):
    
    def __init__(self, oracle, ontology, trainable_module_type=None, feature_dim=1, trainable_gate=False, \
        forward_attention_network=None, backward_attention_network=None, attention_output_network=None):
        super(GQAChooseAttrBatch, self).__init__(oracle, ontology, is_terminal=True, fan_in=1, fan_out=0)
        self._filter = FilterBatch(oracle, trainable_module_type=trainable_module_type, feature_dim=feature_dim, trainable_gate=trainable_gate, 
            forward_attention_network=forward_attention_network, backward_attention_network=backward_attention_network, attention_output_network=attention_output_network)

    def forward(self, op_id, world, variable_set, attribute_list_list, give_answer=True, predicate_question_map=None, likelihood_threshold=0, hard_mode=False):
        attribute_list, batch_index = util.flatten_list(attribute_list_list)
        x = self._filter(op_id, world, variable_set, attribute_list, batch_index if predicate_question_map is None else predicate_question_map)
        log_probability = x.log_probability(give_answer and hard_mode)
        answer = []
        answer_log_probability = []

        if give_answer:
            flags = util.find_max_ind(log_probability, x._predicate_question_map, likelihood_threshold).cpu().numpy().tolist()
            answer = util.unflatten_list(attribute_list, batch_index, flags)
            answer_log_probability = util.unflatten_list(log_probability.cpu().numpy().tolist(), batch_index, flags)
        
        return {'answer': answer, 'log_probability': log_probability, 'options': attribute_list_list, 'variable_set': x, 'type': QuestionType.QUERY,\
            'cumulative_loss': x.cumulative_loss(), 'variable_sets_num': x._prev_variable_sets_num + 1, 'answer_log_probability': answer_log_probability}

    def transform_attention(self, op_id, is_forward, world, attention_state, attribute_list_list, op_feature, predicate_question_map=None):
        attribute_list, batch_index = util.flatten_list(attribute_list_list)
        return self._filter.transform_attention(op_id, is_forward, world, attention_state, attribute_list, op_feature, batch_index if predicate_question_map is None else predicate_question_map)

######################################################################################################################################

class GQAChooseRelBatch(GQABatchOperatorBase):
    
    def __init__(self, oracle, ontology, trainable_module_type=None, feature_dim=1, trainable_gate=False, \
        forward_attention_network=None, backward_attention_network=None, attention_output_network=None):
        super(GQAChooseRelBatch, self).__init__(oracle, ontology, is_terminal=True, fan_in=1, fan_out=0)
        self._gqa_select = GQASelectBatch(oracle, ontology, trainable_module_type=trainable_module_type, feature_dim=feature_dim, trainable_gate=trainable_gate, 
            forward_attention_network=forward_attention_network, backward_attention_network=backward_attention_network, attention_output_network=attention_output_network)
        self._relate = RelateBatch(oracle, trainable_module_type=trainable_module_type, feature_dim=feature_dim, trainable_gate=trainable_gate, 
            forward_attention_network=forward_attention_network, backward_attention_network=backward_attention_network, attention_output_network=attention_output_network)

    def forward(self, op_id, world, variable_set, relation_list_list, is_subject, attribute_list=None, give_answer=True, predicate_question_map=None, likelihood_threshold=0, hard_mode=False):
        relation_list, batch_index = util.flatten_list(relation_list_list)
        x = self._gqa_select(op_id, world, attribute_list, give_answer, predicate_question_map, likelihood_threshold)

        subject_set = x.gate(variable_set, is_subject)
        object_set = variable_set.gate(x, is_subject)

        subject_set, object_set = self._relate(op_id, world, subject_set, object_set, relation_list, batch_index if predicate_question_map is None else predicate_question_map)
        is_subject_tensor = torch.tensor(is_subject, dtype=variable_set.dtype, device=variable_set.device).unsqueeze(1)
        is_subject_tensor = util.mm(subject_set._predicate_question_map, is_subject_tensor).squeeze(1)
        x = subject_set.gate(object_set, is_subject_tensor)
        log_probability = x.log_probability(give_answer and hard_mode)
        answer = []
        answer_log_probability = []
        
        if give_answer:
            flags = util.find_max_ind(log_probability, x._predicate_question_map, likelihood_threshold).cpu().numpy().tolist()
            answer = util.unflatten_list(relation_list, batch_index, flags)
            answer_log_probability = util.unflatten_list(log_probability.cpu().numpy().tolist(), batch_index, flags)
        
        return {'answer': answer, 'log_probability': log_probability, 'options': relation_list_list, 'variable_set': x, 'type': QuestionType.QUERY,\
            'cumulative_loss': x.cumulative_loss(), 'variable_sets_num': x._prev_variable_sets_num + 1, 'answer_log_probability': answer_log_probability}
    
    def transform_attention(self, op_id, is_forward, world, attention_state, relation_list_list, is_subject, attribute_list, op_feature, predicate_question_map=None):
        relation_list, batch_index = util.flatten_list(relation_list_list)
        is_subject_tensor = torch.tensor(is_subject, dtype=attention_state.dtype, device=attention_state.device).unsqueeze(1)
        is_subject_tensor = util.mm(predicate_question_map, is_subject_tensor).squeeze(1)

        if is_forward:
            x = self._gqa_select.transform_attention(op_id, is_forward, world, None, attribute_list, op_feature)
            
            subject_state = x.gate(attention_state, is_subject)
            object_state = attention_state.gate(x, is_subject)

            subject_state, object_state = self._relate.transform_attention(op_id, is_forward, world, subject_state, object_state, relation_list, op_feature,\
                batch_index if predicate_question_map is None else predicate_question_map)
            return subject_state.gate(object_state, is_subject_tensor)
        else:
            x = world.attention_state(attention_state._name, (torch.zeros_like(attention_state._state[0]), torch.zeros_like(attention_state._state[1])))

            object_set = x.gate(attention_state, is_subject_tensor)
            subject_set = attention_state.gate(x, is_subject_tensor)
            subject_set, object_set = self._relate.transform_attention(op_id, is_forward, world, subject_set, object_set, relation_list, op_feature,\
                batch_index if predicate_question_map is None else predicate_question_map)

            self._gqa_select.transform_attention(op_id, is_forward, world, subject_set.gate(object_set, is_subject), attribute_list, op_feature)
            return object_set.gate(subject_set, is_subject)

######################################################################################################################################

class GQAQueryAttrBatch(GQABatchOperatorBase):
    
    def __init__(self, oracle, ontology, trainable_module_type=None, feature_dim=1, trainable_gate=False, \
        forward_attention_network=None, backward_attention_network=None, attention_output_network=None):
        super(GQAQueryAttrBatch, self).__init__(oracle, ontology, is_terminal=True, fan_in=1, fan_out=0)
        self._gqa_choose_attr = GQAChooseAttrBatch(oracle, ontology, trainable_module_type=trainable_module_type, feature_dim=feature_dim, trainable_gate=trainable_gate, 
            forward_attention_network=forward_attention_network, backward_attention_network=backward_attention_network, attention_output_network=attention_output_network)

    def forward(self, op_id, world, variable_set, category_list, give_answer=True, predicate_question_map=None, likelihood_threshold=0, hard_mode=False):
        attribute_list_list = [self._ontology.query(category if category not in ['name', 'type'] else n) for category, n in zip(category_list, variable_set._name)]
        return self._gqa_choose_attr(op_id, world, variable_set, attribute_list_list, give_answer, predicate_question_map, likelihood_threshold)
    
    def transform_attention(self, op_id, is_forward, world, attention_state, category_list, op_feature, predicate_question_map=None):
        attribute_list_list = [self._ontology.query(category if category not in ['name', 'type'] else n) for category, n in zip(category_list, attention_state._name)]
        return self._gqa_choose_attr.transform_attention(op_id, is_forward, world, attention_state, attribute_list_list, op_feature, predicate_question_map)

######################################################################################################################################

class GQAFilterBatch(GQABatchOperatorBase):
    
    def __init__(self, oracle, ontology, trainable_module_type=None, feature_dim=1, trainable_gate=False, \
        forward_attention_network=None, backward_attention_network=None, attention_output_network=None):
        super(GQAFilterBatch, self).__init__(oracle, ontology, is_terminal=False, fan_in=1, fan_out=1)
        self._filter = FilterBatch(oracle, trainable_module_type=trainable_module_type, feature_dim=feature_dim, trainable_gate=trainable_gate, 
            forward_attention_network=forward_attention_network, backward_attention_network=backward_attention_network, attention_output_network=attention_output_network)

    def forward(self, op_id, world, variable_set, attribute_list, give_answer=True, predicate_question_map=None, likelihood_threshold=0, hard_mode=False):
        res = self._filter(op_id, world, variable_set, attribute_list)

        # name = []
        # for attribute, n in zip(attribute_list, res._name):
        #     # if self._ontology.is_adjective(attribute):
        #     #     name += [attribute + ' ' + n]
        #     if self._ontology.is_noun(attribute):
        #         name += [attribute]
        #     else:
        #         name += [n]
        
        # res._name = name
        return res

    def transform_attention(self, op_id, is_forward, world, attention_state, attribute_list, op_feature, predicate_question_map=None):
        res = self._filter.transform_attention(op_id, is_forward, world, attention_state, attribute_list, op_feature)

        # name = []
        # for attribute, n in zip(attribute_list, res._name):
        #     # if self._ontology.is_adjective(attribute):
        #     #     name += [attribute + ' ' + n]
        #     if self._ontology.is_noun(attribute):
        #         name += [attribute]
        #     else:
        #         name += [n]
        
        # res._name = name
        return res

######################################################################################################################################

class GQARelateBatch(GQABatchOperatorBase):
    
    def __init__(self, oracle, ontology, trainable_module_type=None, feature_dim=1, trainable_gate=False, \
        forward_attention_network=None, backward_attention_network=None, attention_output_network=None):
        super(GQARelateBatch, self).__init__(oracle, ontology, is_terminal=False, fan_in=1, fan_out=1)
        self._gqa_select = GQASelectBatch(oracle, ontology, trainable_module_type=trainable_module_type, feature_dim=feature_dim, trainable_gate=trainable_gate, 
            forward_attention_network=forward_attention_network, backward_attention_network=backward_attention_network, attention_output_network=attention_output_network)
        self._relate = RelateBatch(oracle, trainable_module_type=trainable_module_type, feature_dim=feature_dim, trainable_gate=trainable_gate, 
            forward_attention_network=forward_attention_network, backward_attention_network=backward_attention_network, attention_output_network=attention_output_network)

    def forward(self, op_id, world, variable_set, relation_list, is_subject, attribute_list=None, give_answer=True, predicate_question_map=None, likelihood_threshold=0, hard_mode=False):
        x = self._gqa_select(op_id, world, attribute_list, give_answer, predicate_question_map, likelihood_threshold)

        subject_set = x.gate(variable_set, is_subject)
        object_set = variable_set.gate(x, is_subject)
        subject_set, object_set = self._relate(op_id, world, subject_set, object_set, relation_list)

        return subject_set.gate(object_set, is_subject)

    def transform_attention(self, op_id, is_forward, world, attention_state, relation_list, is_subject, attribute_list, op_feature, predicate_question_map=None):
        if is_forward:
            x = self._gqa_select.transform_attention(op_id, is_forward, world, None, attribute_list, op_feature)

            subject_set = x.gate(attention_state, is_subject)
            object_set = attention_state.gate(x, is_subject)
            subject_set, object_set = self._relate.transform_attention(op_id, is_forward, world, subject_set, object_set, relation_list, op_feature)

            return subject_set.gate(object_set, is_subject)
        else:
            x = world.attention_state(attention_state._name, (torch.zeros_like(attention_state._state[0]), torch.zeros_like(attention_state._state[1])))

            object_set = x.gate(attention_state, is_subject)
            subject_set = attention_state.gate(x, is_subject)
            subject_set, object_set = self._relate.transform_attention(op_id, is_forward, world, subject_set, object_set, relation_list, op_feature)

            self._gqa_select.transform_attention(op_id, is_forward, world, subject_set.gate(object_set, is_subject), attribute_list, op_feature)
            return object_set.gate(subject_set, is_subject)

######################################################################################################################################

class GQAExistBatch(GQABatchOperatorBase):
    
    def __init__(self, oracle, ontology):
        super(GQAExistBatch, self).__init__(oracle, ontology, is_terminal=True, fan_in=1, fan_out=0)

    def forward(self, op_id, world, variable_set, give_answer=True, predicate_question_map=None, likelihood_threshold=0, hard_mode=False):
        log_probability = variable_set.log_probability(give_answer and hard_mode)
        answer = []
        answer_log_probability = []

        if give_answer:
            probability = util.safe_exp(log_probability).cpu().numpy().tolist()
            answer = [['yes'] if probability[i] > 0.5 else ['no'] for i in range(variable_set.batch_size())]
            answer_log_probability =  [[math.log(probability[i])] if probability[i] > 0.5 else [math.log(1 - probability[i])] for i in range(variable_set.batch_size())]
        
        return {'answer': answer, 'log_probability': log_probability, 'options': ['no', 'yes'], 'variable_set': variable_set, 'type': QuestionType.BINARY,\
            'cumulative_loss': variable_set.cumulative_loss(), 'variable_sets_num': variable_set._prev_variable_sets_num + 1, 'answer_log_probability': answer_log_probability}

    def transform_attention(self, op_id, is_forward, world, attention_state, op_feature, predicate_question_map=None):
        return attention_state

######################################################################################################################################

class GQAVerifyAttrBatch(GQABatchOperatorBase):
    
    def __init__(self, oracle, ontology, trainable_module_type=None, feature_dim=1, trainable_gate=False, \
        forward_attention_network=None, backward_attention_network=None, attention_output_network=None):
        super(GQAVerifyAttrBatch, self).__init__(oracle, ontology, is_terminal=True, fan_in=1, fan_out=0)
        self._filter = FilterBatch(oracle, trainable_module_type=trainable_module_type, feature_dim=feature_dim, trainable_gate=trainable_gate, 
            forward_attention_network=forward_attention_network, backward_attention_network=backward_attention_network, attention_output_network=attention_output_network)

    def forward(self, op_id, world, variable_set, attribute_list, give_answer=True, predicate_question_map=None, likelihood_threshold=0, hard_mode=False):
        x = self._filter(op_id, world, variable_set, attribute_list)
        log_probability = x.log_probability(give_answer and hard_mode)
        answer = []
        answer_log_probability = []

        if give_answer:
            probability = util.safe_exp(log_probability).cpu().numpy().tolist()
            answer = [['yes'] if probability[i] > 0.5 else ['no'] for i in range(variable_set.batch_size())]
            answer_log_probability =  [[math.log(probability[i])] if probability[i] > 0.5 else [math.log(1 - probability[i])] for i in range(variable_set.batch_size())]

        return {'answer': answer, 'log_probability': log_probability, 'options': ['no', 'yes'], 'variable_set': x, 'type': QuestionType.BINARY,\
            'cumulative_loss': x.cumulative_loss(), 'variable_sets_num': x._prev_variable_sets_num + 1, 'answer_log_probability': answer_log_probability}
    
    def transform_attention(self, op_id, is_forward, world, attention_state, attribute_list, op_feature, predicate_question_map=None):
        return self._filter.transform_attention(op_id, is_forward, world, attention_state, attribute_list, op_feature)

######################################################################################################################################

class GQAVerifyAttrsBatch(GQABatchOperatorBase):
    
    def __init__(self, oracle, ontology, trainable_module_type=None, feature_dim=1, trainable_gate=False, \
        forward_attention_network=None, backward_attention_network=None, attention_output_network=None):
        super(GQAVerifyAttrsBatch, self).__init__(oracle, ontology, is_terminal=True, fan_in=1, fan_out=0)
        self._filter = FilterBatch(oracle, trainable_module_type=trainable_module_type, feature_dim=feature_dim, trainable_gate=trainable_gate, 
            forward_attention_network=forward_attention_network, backward_attention_network=backward_attention_network, attention_output_network=attention_output_network)

    def forward(self, op_id, world, variable_set, attribute_list_list, give_answer=True, predicate_question_map=None, likelihood_threshold=0, hard_mode=False):
        attribute_list, batch_index = util.flatten_list(attribute_list_list)
        x = self._filter(op_id, world, variable_set, attribute_list, batch_index if predicate_question_map is None else predicate_question_map, normalized_probability=False)
        # log_probability = util.mm(x._predicate_question_map.transpose(0, 1), x.log_probability(give_answer and hard_mode).unsqueeze(1)).squeeze(1)
        
        log_attention = util.mm(x._predicate_question_map.transpose(0, 1), x._log_attention)
        y = BatchVariableSet(variable_set._name, variable_set._device, variable_set._object_num, batch_size=variable_set._batch_size, 
                quantifiers=variable_set._quantifier, log_attention=log_attention, batch_object_map=variable_set._batch_object_map, 
                predicate_question_map=None, base_cumulative_loss=x._base_cumulative_loss, 
                prev_variable_sets_num=x._prev_variable_sets_num).to(variable_set.dtype)

        log_probability = y.log_probability(give_answer and hard_mode)        
        answer = []
        answer_log_probability = []

        if give_answer:
            probability = util.safe_exp(log_probability).cpu().numpy().tolist()
            answer = [['yes'] if probability[i] > 0.5 else ['no'] for i in range(variable_set.batch_size())]
            answer_log_probability =  [[math.log(probability[i])] if probability[i] > 0.5 else [math.log(1 - probability[i])] for i in range(variable_set.batch_size())]
        
        return {'answer': answer, 'log_probability': log_probability, 'options': ['no', 'yes'], 'variable_set': y, 'type': QuestionType.BINARY,\
            'cumulative_loss': y.cumulative_loss(), 'variable_sets_num': y._prev_variable_sets_num + 1, 'answer_log_probability': answer_log_probability}
    
    def transform_attention(self, op_id, is_forward, world, attention_state, attribute_list_list, op_feature, predicate_question_map=None):
        attribute_list, batch_index = util.flatten_list(attribute_list_list)
        return self._filter.transform_attention(op_id, is_forward, world, attention_state, attribute_list, op_feature, batch_index if predicate_question_map is None else predicate_question_map)

######################################################################################################################################

class GQAVerifyRelBatch(GQABatchOperatorBase):

    def __init__(self, oracle, ontology, trainable_module_type=None, feature_dim=1, trainable_gate=False, \
        forward_attention_network=None, backward_attention_network=None, attention_output_network=None):
        super(GQAVerifyRelBatch, self).__init__(oracle, ontology, is_terminal=True, fan_in=1, fan_out=0)
        self._gqa_relate = GQARelateBatch(oracle, ontology, trainable_module_type=trainable_module_type, feature_dim=feature_dim, trainable_gate=trainable_gate, 
            forward_attention_network=forward_attention_network, backward_attention_network=backward_attention_network, attention_output_network=attention_output_network)

    def forward(self, op_id, world, variable_set, relation_list, is_subject, attribute_list=None, give_answer=True, predicate_question_map=None, likelihood_threshold=0, hard_mode=False):
        x = self._gqa_relate(op_id, world, variable_set, relation_list, is_subject, attribute_list, give_answer, predicate_question_map, likelihood_threshold)
        log_probability = x.log_probability(give_answer and hard_mode)
        answer = []
        answer_log_probability = []

        if give_answer:
            probability = util.safe_exp(log_probability).cpu().numpy().tolist()
            answer = [['yes'] if probability[i] > 0.5 else ['no'] for i in range(variable_set.batch_size())]
            answer_log_probability =  [[math.log(probability[i])] if probability[i] > 0.5 else [math.log(1 - probability[i])] for i in range(variable_set.batch_size())]
        
        return {'answer': answer, 'log_probability': log_probability, 'options': ['no', 'yes'], 'variable_set': x, 'type': QuestionType.BINARY,\
            'cumulative_loss': x.cumulative_loss(), 'variable_sets_num': x._prev_variable_sets_num + 1, 'answer_log_probability': answer_log_probability}
    
    def transform_attention(self, op_id, is_forward, world, attention_state, relation_list, is_subject, attribute_list, op_feature, predicate_question_map=None):
        return self._gqa_relate.transform_attention(op_id, is_forward, world, attention_state, relation_list, is_subject, attribute_list, op_feature)

######################################################################################################################################

class GQAAndBatch(GQABatchOperatorBase):

    def __init__(self, oracle, ontology):
        super(GQAAndBatch, self).__init__(oracle, ontology, is_terminal=True, fan_in=2, fan_out=0)

    def forward(self, op_id, world, variable_set1, variable_set2, give_answer=True, predicate_question_map=None, likelihood_threshold=0, hard_mode=False):
        v1 = variable_set1.log_probability(give_answer and hard_mode) if isinstance(variable_set1, BatchVariableSet) else variable_set1['log_probability']
        v2 = variable_set2.log_probability(give_answer and hard_mode) if isinstance(variable_set2, BatchVariableSet) else variable_set2['log_probability']
        
        log_probability = util.log_and(v1, v2)
        batch_size = variable_set1.batch_size() if isinstance(variable_set1, BatchVariableSet) else variable_set1['log_probability'].numel()
        answer = []
        answer_log_probability = []

        if give_answer:
            probability = util.safe_exp(log_probability).cpu().numpy().tolist()
            answer = [['yes'] if probability[i] > 0.5 else ['no'] for i in range(batch_size)]
            answer_log_probability =  [[math.log(probability[i])] if probability[i] > 0.5 else [math.log(1 - probability[i])] for i in range(batch_size)]

        cum_loss1 = variable_set1.cumulative_loss() if isinstance(variable_set1, BatchVariableSet) else variable_set1['cumulative_loss']
        vs_num1 = variable_set1._prev_variable_sets_num + 1 if isinstance(variable_set1, BatchVariableSet) else variable_set1['variable_sets_num']

        cum_loss2 = variable_set2.cumulative_loss() if isinstance(variable_set2, BatchVariableSet) else variable_set2['cumulative_loss']
        vs_num2 = variable_set2._prev_variable_sets_num + 1 if isinstance(variable_set2, BatchVariableSet) else variable_set2['variable_sets_num']

        return {'answer': answer, 'log_probability': log_probability, 'options': ['no', 'yes'], 'variable_set': None, 'type': QuestionType.BINARY,\
            'cumulative_loss': cum_loss1 + cum_loss2, 'variable_sets_num': vs_num1 + vs_num2, 'answer_log_probability': answer_log_probability}

    def transform_attention(self, op_id, is_forward, world, attention_state1, attention_state2, op_feature, predicate_question_map=None):
        return attention_state1, attention_state2

######################################################################################################################################

class GQAOrBatch(GQABatchOperatorBase):

    def __init__(self, oracle, ontology):
        super(GQAOrBatch, self).__init__(oracle, ontology, is_terminal=True, fan_in=2, fan_out=0)

    def forward(self, op_id, world, variable_set1, variable_set2, give_answer=True, predicate_question_map=None, likelihood_threshold=0, hard_mode=False):
        v1 = variable_set1.log_probability(give_answer and hard_mode) if isinstance(variable_set1, BatchVariableSet) else variable_set1['log_probability']
        v2 = variable_set2.log_probability(give_answer and hard_mode) if isinstance(variable_set2, BatchVariableSet) else variable_set2['log_probability']

        log_probability = util.log_or(v1, v2)
        batch_size = variable_set1.batch_size() if isinstance(variable_set1, BatchVariableSet) else variable_set1['log_probability'].numel()
        answer = []
        answer_log_probability = []

        if give_answer:
            probability = util.safe_exp(log_probability).cpu().numpy().tolist()
            answer = [['yes'] if probability[i] > 0.5 else ['no'] for i in range(batch_size)]
            answer_log_probability =  [[math.log(probability[i])] if probability[i] > 0.5 else [math.log(1 - probability[i])] for i in range(batch_size)]

        cum_loss1 = variable_set1.cumulative_loss() if isinstance(variable_set1, BatchVariableSet) else variable_set1['cumulative_loss']
        vs_num1 = variable_set1._prev_variable_sets_num + 1 if isinstance(variable_set1, BatchVariableSet) else variable_set1['variable_sets_num']

        cum_loss2 = variable_set2.cumulative_loss() if isinstance(variable_set2, BatchVariableSet) else variable_set2['cumulative_loss']
        vs_num2 = variable_set2._prev_variable_sets_num + 1 if isinstance(variable_set2, BatchVariableSet) else variable_set2['variable_sets_num']

        return {'answer': answer, 'log_probability': log_probability, 'options': ['no', 'yes'], 'variable_set': None, 'type': QuestionType.BINARY,\
            'cumulative_loss': cum_loss1 + cum_loss2, 'variable_sets_num': vs_num1 + vs_num2, 'answer_log_probability': answer_log_probability}

    def transform_attention(self, op_id, is_forward, world, attention_state1, attention_state2, op_feature, predicate_question_map=None):
        return attention_state1, attention_state2
    
######################################################################################################################################

class GQAAllSameBatch(GQABatchOperatorBase):
    
    def __init__(self, oracle, ontology, trainable_module_type=None, feature_dim=1, trainable_gate=False, \
        forward_attention_network=None, backward_attention_network=None, attention_output_network=None):
        super(GQAAllSameBatch, self).__init__(oracle, ontology, is_terminal=True, fan_in=1, fan_out=0)
        self._filter = FilterBatch(oracle, trainable_module_type=trainable_module_type, feature_dim=feature_dim, trainable_gate=trainable_gate, 
            forward_attention_network=forward_attention_network, backward_attention_network=backward_attention_network, attention_output_network=attention_output_network)

    def forward(self, op_id, world, variable_set, category_list, give_answer=True, predicate_question_map=None, likelihood_threshold=0, hard_mode=False):
        attribute_list_list = [self._ontology.query(category if category not in ['name', 'type'] else n) for category, n in zip(category_list, variable_set._name)]
        attribute_list, batch_index = util.flatten_list(attribute_list_list)
        x = self._filter(op_id, world, variable_set, attribute_list, batch_index if predicate_question_map is None else predicate_question_map)

        # Apply logical implication (Pre-condition ==> Being all the same) before aggregation.
        log_posterior = util.log_not(util.log_and(util.mm(x._predicate_question_map, variable_set._log_attention), \
            util.log_not(x._log_attention)))

        # Compute the aggregate log-likelihood
        temp = BatchVariableSet(x._name, variable_set.device, x.object_num(), batch_size=len(attribute_list), quantifiers=Quantifier.FOR_ALL, \
                log_attention=log_posterior, batch_object_map=x._batch_object_map, predicate_question_map=x._predicate_question_map).to(x.dtype)
        log_probability = temp.log_probability(give_answer and hard_mode)

        # Compute logical OR for each question
        log_probability = util.mm(x._predicate_question_map.transpose(0, 1), util.log_not(log_probability.unsqueeze(1)))
        log_probability = util.log_not(log_probability.squeeze(1))
        answer = []
        answer_log_probability = []

        if give_answer:
            probability = util.safe_exp(log_probability).cpu().numpy().tolist()
            answer = [['yes'] if probability[i] > 0.5 else ['no'] for i in range(variable_set.batch_size())]
            answer_log_probability =  [[math.log(probability[i])] if probability[i] > 0.5 else [math.log(1 - probability[i])] for i in range(variable_set.batch_size())]
        
        return {'answer': answer, 'log_probability': log_probability, 'options': ['no', 'yes'], 'variable_set': None, 'type': QuestionType.BINARY,\
            'cumulative_loss': x.cumulative_loss(), 'variable_sets_num': x._prev_variable_sets_num + 1, 'answer_log_probability': answer_log_probability}

    def transform_attention(self, op_id, is_forward, world, attention_state, category_list, op_feature, predicate_question_map=None):
        attribute_list_list = [self._ontology.query(category if category not in ['name', 'type'] else n) for category, n in zip(category_list, attention_state._name)]
        attribute_list, batch_index = util.flatten_list(attribute_list_list)
        return self._filter.transform_attention(op_id, is_forward, world, attention_state, attribute_list, op_feature, batch_index if predicate_question_map is None else predicate_question_map)
    
######################################################################################################################################

class GQAAllDifferentBatch(GQABatchOperatorBase):
    
    ## Review: AllDifferent has been implemented as the logical NOT of AllSame. This can be problematic if by 'All Different' we mean every two objects are different.
    
    def __init__(self, oracle, ontology, trainable_module_type=None, feature_dim=1, trainable_gate=False, \
        forward_attention_network=None, backward_attention_network=None, attention_output_network=None):
        super(GQAAllDifferentBatch, self).__init__(oracle, ontology, is_terminal=True, fan_in=1, fan_out=0)
        self._gqa_all_same = GQAAllSameBatch(oracle, ontology, trainable_module_type=trainable_module_type, feature_dim=feature_dim, trainable_gate=trainable_gate, 
            forward_attention_network=forward_attention_network, backward_attention_network=backward_attention_network, attention_output_network=attention_output_network)

    def forward(self, op_id, world, variable_set, category_list, give_answer=True, predicate_question_map=None, likelihood_threshold=0, hard_mode=False):
        all_same = self._gqa_all_same(op_id, world, variable_set, category_list, give_answer, predicate_question_map, likelihood_threshold)
        log_probability = util.log_not(all_same['log_probability'])
        answer = []
        answer_log_probability = []

        if give_answer:
            probability = util.safe_exp(log_probability).cpu().numpy().tolist()
            answer = [['yes'] if probability[i] > 0.5 else ['no'] for i in range(variable_set.batch_size())]
            answer_log_probability =  [[math.log(probability[i])] if probability[i] > 0.5 else [math.log(1 - probability[i])] for i in range(variable_set.batch_size())]
        
        return {'answer': answer, 'log_probability': log_probability, 'options': ['no', 'yes'], 'variable_set': all_same['variable_set'], 'type': QuestionType.BINARY,\
            'cumulative_loss': all_same['cumulative_loss'], 'variable_sets_num': all_same['variable_sets_num'], 'answer_log_probability': answer_log_probability}

    def transform_attention(self, op_id, is_forward, world, attention_state, category_list, op_feature, predicate_question_map=None):
        return self._gqa_all_same.transform_attention(op_id, is_forward, world, attention_state, category_list, op_feature, predicate_question_map)

######################################################################################################################################

class GQATwoSameBatch(GQABatchOperatorBase):

    def __init__(self, oracle, ontology, trainable_module_type=None, feature_dim=1, trainable_gate=False, \
        forward_attention_network=None, backward_attention_network=None, attention_output_network=None):
        super(GQATwoSameBatch, self).__init__(oracle, ontology, is_terminal=True, fan_in=2, fan_out=0)
        self._filter = FilterBatch(oracle, trainable_module_type=trainable_module_type, feature_dim=feature_dim, trainable_gate=trainable_gate, 
            forward_attention_network=forward_attention_network, backward_attention_network=backward_attention_network, attention_output_network=attention_output_network)

    def forward(self, op_id, world, variable_set1, variable_set2, category_list, give_answer=True, predicate_question_map=None, likelihood_threshold=0, hard_mode=False):
        attribute_list_list = [self._ontology.query(category if category not in ['name', 'type'] else n) for category, n in zip(category_list, variable_set1._name)]
        attribute_list, batch_index = util.flatten_list(attribute_list_list)
        
        x1 = self._filter(op_id + ':0', world, variable_set1, attribute_list, batch_index if predicate_question_map is None else predicate_question_map)
        x2 = self._filter(op_id + ':1', world, variable_set2, attribute_list, batch_index if predicate_question_map is None else predicate_question_map)

        log_probability = util.log_and(x1.log_probability(give_answer and hard_mode), x2.log_probability(give_answer and hard_mode))
        
        # Compute logical OR for each question
        log_probability = util.mm(x1._predicate_question_map.transpose(0, 1), util.log_not(log_probability.unsqueeze(1)))
        log_probability = util.log_not(log_probability.squeeze(1))
        answer = []
        answer_log_probability = []

        if give_answer:
            probability = util.safe_exp(log_probability).cpu().numpy().tolist()
            answer = [['yes'] if probability[i] > 0.5 else ['no'] for i in range(variable_set1.batch_size())]
            answer_log_probability =  [[math.log(probability[i])] if probability[i] > 0.5 else [math.log(1 - probability[i])] for i in range(variable_set1.batch_size())]
        
        cum_loss1 = x1.cumulative_loss()
        vs_num1 = x1._prev_variable_sets_num + 1

        cum_loss2 = x2.cumulative_loss()
        vs_num2 = x2._prev_variable_sets_num + 1

        return {'answer': answer, 'log_probability': log_probability, 'options': ['no', 'yes'], 'variable_set': None, 'type': QuestionType.BINARY,\
            'cumulative_loss': cum_loss1 + cum_loss2, 'variable_sets_num': vs_num1 + vs_num2, 'answer_log_probability': answer_log_probability}
    
    def transform_attention(self, op_id, is_forward, world, attention_state1, attention_state2, category_list, op_feature, predicate_question_map=None):
        attribute_list_list = [self._ontology.query(category if category not in ['name', 'type'] else n) for category, n in zip(category_list, attention_state1._name)]
        attribute_list, batch_index = util.flatten_list(attribute_list_list)
        
        x1 = self._filter.transform_attention(op_id + ':0', is_forward, world, attention_state1, attribute_list, op_feature, batch_index if predicate_question_map is None else predicate_question_map)
        x2 = self._filter.transform_attention(op_id + ':1', is_forward, world, attention_state2, attribute_list, op_feature, batch_index if predicate_question_map is None else predicate_question_map)

        return x1, x2

######################################################################################################################################

class GQATwoDifferentBatch(GQABatchOperatorBase):
    
    def __init__(self, oracle, ontology, trainable_module_type=None, feature_dim=1, trainable_gate=False, \
        forward_attention_network=None, backward_attention_network=None, attention_output_network=None):
        super(GQATwoDifferentBatch, self).__init__(oracle, ontology, is_terminal=True, fan_in=2, fan_out=0)
        self._gqa_two_same = GQATwoSameBatch(oracle, ontology, trainable_module_type=trainable_module_type, feature_dim=feature_dim, trainable_gate=trainable_gate, 
            forward_attention_network=forward_attention_network, backward_attention_network=backward_attention_network, attention_output_network=attention_output_network)
          
    def forward(self, op_id, world, variable_set1, variable_set2, category_list, give_answer=True, predicate_question_map=None, likelihood_threshold=0, hard_mode=False):
        two_same = self._gqa_two_same(op_id, world, variable_set1, variable_set2, category_list, give_answer, predicate_question_map, likelihood_threshold)
        log_probability = util.log_not(two_same['log_probability'])
        answer = []
        answer_log_probability = []

        if give_answer:
            probability = util.safe_exp(log_probability).cpu().numpy().tolist()
            answer = [['yes'] if probability[i] > 0.5 else ['no'] for i in range(variable_set1.batch_size())]
            answer_log_probability =  [[math.log(probability[i])] if probability[i] > 0.5 else [math.log(1 - probability[i])] for i in range(variable_set1.batch_size())]
        
        return {'answer': answer, 'log_probability': log_probability, 'options': ['no', 'yes'], 'variable_set': None, 'type': QuestionType.BINARY,\
            'cumulative_loss': two_same['cumulative_loss'], 'variable_sets_num': two_same['variable_sets_num'], 'answer_log_probability': answer_log_probability}
    
    def transform_attention(self, op_id, is_forward, world, attention_state1, attention_state2, category_list, op_feature, predicate_question_map=None):
        return self._gqa_two_same.transform_attention(op_id, is_forward, world, attention_state1, attention_state2, category_list, op_feature, predicate_question_map)

######################################################################################################################################

class GQACompareBatch(GQABatchOperatorBase):

    def __init__(self, oracle, ontology, trainable_module_type=None, feature_dim=1, trainable_gate=False, \
        forward_attention_network=None, backward_attention_network=None, attention_output_network=None):
        super(GQACompareBatch, self).__init__(oracle, ontology, is_terminal=True, fan_in=2, fan_out=0)
        self._filter = FilterBatch(oracle, trainable_module_type=trainable_module_type, feature_dim=feature_dim, trainable_gate=trainable_gate, 
            forward_attention_network=forward_attention_network, backward_attention_network=backward_attention_network, attention_output_network=attention_output_network)
        self._log_soft_max = nn.LogSoftmax(dim=1)

    def forward(self, op_id, world, variable_set1, variable_set2, attribute_list, is_less, give_answer=True, predicate_question_map=None, likelihood_threshold=0, hard_mode=False):
        x1 = self._filter(op_id + ':0', world, variable_set1, attribute_list)
        x2 = self._filter(op_id + ':1', world, variable_set2, attribute_list)

        log_probability = torch.stack([x1.log_probability(give_answer and hard_mode), x2.log_probability(give_answer and hard_mode)]).transpose(0, 1).contiguous()
        log_probability = self._log_soft_max(log_probability)

        alpha = torch.tensor(is_less, dtype=variable_set1.dtype, device=variable_set1.device).unsqueeze(1)
        log_probability = util.log_parametric_not(log_probability, alpha, 1)

        options = list(zip(variable_set1._name, variable_set2._name))
        answer = []
        answer_log_probability = []

        if give_answer:
            ind = log_probability.max(1)[1]
            for i in range(variable_set1.batch_size()):
                answer += [[options[i][ind[i]]]]
                answer_log_probability += [[log_probability[i, ind[i]].cpu().numpy().tolist()]]

        # options = list(sum(options, ()))        
        cum_loss1 = x1.cumulative_loss()
        vs_num1 = x1._prev_variable_sets_num + 1

        cum_loss2 = x2.cumulative_loss()
        vs_num2 = x2._prev_variable_sets_num + 1
        
        return {'answer': answer, 'log_probability': log_probability.view(-1), 'options': options, 'variable_set': None, 'type': QuestionType.QUERY,\
            'cumulative_loss': cum_loss1 + cum_loss2, 'variable_sets_num': vs_num1 + vs_num2, 'answer_log_probability': answer_log_probability}
    
    def transform_attention(self, op_id, is_forward, world, attention_state1, attention_state2, attribute_list, is_less, op_feature, predicate_question_map=None):
        x1 = self._filter.transform_attention(op_id + ':0', is_forward, world, attention_state1, attribute_list, op_feature)
        x2 = self._filter.transform_attention(op_id + ':1', is_forward, world, attention_state2, attribute_list, op_feature)

        return x1, x2

######################################################################################################################################

class GQAEndBatch(GQABatchOperatorBase):

    def __init__(self, oracle, ontology):
        super(GQAEndBatch, self).__init__(oracle, ontology, is_terminal=True, fan_in=1, fan_out=0)

    def forward(self, op_id, world, variable_set, give_answer=True, predicate_question_map=None, likelihood_threshold=0, hard_mode=False):
        answer = []
        answer_log_probability = []
        if give_answer:
            answer = [[name] for name in variable_set._name]

        return {'answer': answer, 'log_probability': variable_set.log_probability(give_answer and hard_mode), 'options': [], 'variable_set': variable_set, 'type': QuestionType.STATEMENT,\
            'cumulative_loss': variable_set.cumulative_loss(), 'variable_sets_num': variable_set._prev_variable_sets_num + 1, 'answer_log_probability': answer_log_probability}

    def transform_attention(self, op_id, is_forward, world, attention_state, op_feature, predicate_question_map=None):
        return attention_state

######################################################################################################################################

class GQAObjectAttrBatch(GQABatchOperatorBase):
    
    def __init__(self, oracle, ontology, trainable_module_type=None, feature_dim=1, trainable_gate=False, \
        forward_attention_network=None, backward_attention_network=None, attention_output_network=None):
        super(GQAObjectAttrBatch, self).__init__(oracle, ontology, is_terminal=True, fan_in=0, fan_out=0)
        self._filter = FilterBatch(oracle, trainable_module_type=trainable_module_type, feature_dim=feature_dim, trainable_gate=trainable_gate, 
            forward_attention_network=forward_attention_network, backward_attention_network=backward_attention_network, attention_output_network=attention_output_network)

    def forward(self, op_id, world, attribute_list_list_list, give_answer=True, predicate_question_map=None, likelihood_threshold=0, hard_mode=False):
        # [  [ ['red', 'table'], ['blue', 'ball'], ['green', 'apple'] ], [ ['red', 'table'], ['blue', 'ball'], ['green', 'apple'] ]  ]
        batch_size = world.batch_size()
        attribute_list_list, batch_index = util.flatten_list(attribute_list_list_list)
        attribute_list, object_index = util.flatten_list(attribute_list_list)
        object_batch_index = itemgetter(*object_index)(batch_index)

        name = ["entity" for _ in range(batch_size)]
        variable_set = world.variable_set(name, quantifier=Quantifier.EXISTS)
       
        x = self._filter(op_id, world, variable_set, attribute_list, object_batch_index)

        predicate_num = len(attribute_list)
        all_ones = torch.ones(predicate_num, device=world._device, dtype=world.dtype)
        x_ind = torch.arange(predicate_num, dtype=torch.int64, device=world._device)
        y_ind = torch.tensor(object_index, dtype=torch.int64, device=world._device)
        index = torch.stack([x_ind, y_ind])

        if util.is_cuda(world._device):
            object_map = torch.cuda.sparse.FloatTensor(index, all_ones, 
                            torch.Size([predicate_num, x.object_num()]), device=world._device)
        else:
            object_map = torch.sparse.FloatTensor(index, all_ones, 
                            torch.Size([predicate_num, x.object_num()]), device=world._device)

        log_probability = torch.sum(x._log_attention * object_map.to(world.dtype).to_dense(), dim=1)        
        answer = []
        answer_log_probability = []

        if give_answer:
            probability = util.safe_exp(log_probability).cpu().numpy().tolist()
            answer = [['yes'] if probability[i] > 0.5 else ['no'] for i in range(predicate_num)]
            answer_log_probability =  [[math.log(probability[i])] if probability[i] > 0.5 else [math.log(1 - probability[i])] for i in range(predicate_num)]
        
        return {'answer': answer, 'log_probability': log_probability, 'options': attribute_list_list_list, 'variable_set': x, 'type': QuestionType.OBJECT_STATEMENT,\
            'cumulative_loss': x.cumulative_loss(), 'variable_sets_num': x._prev_variable_sets_num + 1, 'answer_log_probability': answer_log_probability}

    def transform_attention(self, op_id, is_forward, world, attention_state, attribute_list_list_list, op_feature, predicate_question_map=None):
        batch_size = world.batch_size()
        attribute_list_list, batch_index = util.flatten_list(attribute_list_list_list)
        attribute_list, object_index = util.flatten_list(attribute_list_list)
        object_batch_index = itemgetter(*object_index)(batch_index)

        name = ["entity" for _ in range(batch_size)]
        return self._filter.transform_attention(op_id, is_forward, world, world.attention_state(name) if is_forward else attention_state, attribute_list, op_feature, object_batch_index)

######################################################################################################################################

class GQAObjectRelBatch(GQABatchOperatorBase):
    
    def __init__(self, oracle, ontology, trainable_module_type=None, feature_dim=1, trainable_gate=False, \
        forward_attention_network=None, backward_attention_network=None, attention_output_network=None):
        super(GQAObjectRelBatch, self).__init__(oracle, ontology, is_terminal=True, fan_in=0, fan_out=0)
        self._relate = RelateBatch(oracle, trainable_module_type=trainable_module_type, feature_dim=feature_dim, trainable_gate=trainable_gate, 
            forward_attention_network=forward_attention_network, backward_attention_network=backward_attention_network, attention_output_network=attention_output_network)

    def forward(self, op_id, world, relation_list_list, give_answer=True, predicate_question_map=None, likelihood_threshold=0, hard_mode=False):
        batch_size = world.batch_size()
        relation_list, object_index = util.flatten_list(relation_list_list)

        name = ["entity" for _ in range(batch_size)]
        subject_set = world.variable_set(name, quantifier=Quantifier.FOR_ALL)
        object_set = world.variable_set(name, quantifier=Quantifier.FOR_ALL)

        subject_set, object_set = self._relate(op_id, world, subject_set, object_set, relation_list, object_index, default_log_likelihood=0)
        log_probability = subject_set.log_probability(give_answer and hard_mode)
        answer = []
        answer_log_probability = []
        
        if give_answer:
            probability = util.safe_exp(log_probability).cpu().numpy().tolist()
            answer = [['yes'] if probability[i] > 0.5 else ['no'] for i in range(len(relation_list))]
            answer_log_probability =  [[math.log(probability[i])] if probability[i] > 0.5 else [math.log(1 - probability[i])] for i in range(relation_list)]
        
        return {'answer': answer, 'log_probability': log_probability, 'options': relation_list_list, 'variable_set': subject_set, 'type': QuestionType.OBJECT_STATEMENT,\
            'cumulative_loss': subject_set.cumulative_loss(), 'variable_sets_num': subject_set._prev_variable_sets_num + 1, 'answer_log_probability': answer_log_probability}
    
    def transform_attention(self, op_id, is_forward, world, attention_state, relation_list_list_list, is_subject, attribute_list, op_feature, predicate_question_map=None):
        batch_size = world.batch_size()
        relation_list_list, batch_index = util.flatten_list(relation_list_list_list)
        relation_list, object_index = util.flatten_list(relation_list_list)
        object_batch_index = itemgetter(*object_index)(batch_index)

        name = ["entity" for _ in range(batch_size)]
        return self._relate.transform_attention(op_id, is_forward, world, world.attention_state(name) if is_forward else attention_state, attribute_list, op_feature, object_batch_index)

######################################################################################################################################

class GQASceneOpBatch(GQABatchOperatorBase):

    def __init__(self, oracle, ontology):
        super(GQASceneOpBatch, self).__init__(oracle, ontology, is_terminal=True, fan_in=0, fan_out=0)

    def forward(self, op_id, world, give_answer=True, predicate_question_map=None, likelihood_threshold=0, hard_mode=False):
        attr_loglikelihood, rel_loglikelihood = self._oracle.compute_all_log_likelihood(world._attribute_features, world._relation_features)
        answer = []
        answer_log_probability = []

        if give_answer:
            attr_answer = (attr_loglikelihood > 0.5).float().cpu().numpy()
            rel_answer = (rel_loglikelihood > 0.5).float().cpu().numpy()
            answer = [attr_answer, rel_answer]
        
        return {'answer': answer, 'log_probability': [attr_loglikelihood, rel_loglikelihood], 'options': [], 'variable_set': None, 'type': QuestionType.SCENE_GRAPH,\
            'cumulative_loss': 0, 'variable_sets_num': 0, 'answer_log_probability': answer_log_probability}

    def transform_attention(self, op_id, is_forward, world, attention_state, attribute_list, op_feature, predicate_question_map=None):
        return attention_state
