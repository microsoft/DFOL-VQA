# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file
# in the project root for full license information.

"The data pipeline for nsvqa."

import torch
import torch.utils.data as data
from torch.utils.data.distributed import DistributedSampler

import linecache, json, re, math, time
import collections
import numpy as np
# import h5pickle as h5py
import h5py
from random import shuffle

from os import listdir
from os.path import isfile, join, splitext

from nsvqa.nn.interpreter.batch_base_types import BatchVariableSet, QuestionType
from nsvqa.nn.interpreter import util

def flatten_list(a_list_list):
    a_list = [a if a is not None else [None] for a in a_list_list]
    batch_index = [i for i, sublist in enumerate(a_list) for item in sublist]
    a_list = [item for sublist in a_list for item in sublist]

    return a_list, batch_index

class OperatorBatch(object):

    def __init__(self, op_name, arguments, question_num, is_terminal, mask=None, question_index=None, process_args=True):
        self._op_name = op_name
        self._is_terminal = is_terminal
        self._op_id = None

        if process_args:
            if 0 < len(arguments) and len(arguments) < question_num:
                self._arguments = arguments
                for _ in range(question_num - len(arguments)):
                    self._arguments.append(None)
            elif len(arguments) >= question_num:
                self._arguments = arguments[:question_num]
            else:
                self._arguments = []

            self._arguments = self._replace_none(self._arguments)
            self._arguments = self._transpose(self._arguments)
        else:
            self._arguments = arguments

        self._question_num = question_num        
        self._predicate_num = self._question_num
        self._predicate_question_map = None
        self._question_index = None

        if len(self._arguments) > 0:
            if any(len(el) > 1 if isinstance(el, list) else False for el in self._arguments[0]):
                args, batch_index = flatten_list(self._arguments[0])
                self._predicate_num = len(args)

                if question_index is not None:
                    self._question_index = question_index
                elif self._predicate_num != self._question_num:
                    self._question_index = torch.tensor(batch_index, dtype=torch.int64)

        if mask is not None:
            if isinstance(mask, np.ndarray):
                self._mask = torch.from_numpy(mask).float()
            else:
                self._mask = mask
        else:
            self._mask = None

    def create_sparse_map(self):
        if self._question_index is not None:
            all_ones = torch.ones(self._predicate_num)
            x_ind = torch.arange(self._predicate_num, dtype=torch.int64)
            index = torch.stack([x_ind, self._question_index])
            
            self._predicate_question_map = torch.sparse.FloatTensor(index, all_ones, 
                                torch.Size([self._predicate_num, self._question_num]))

    def _transpose(self, a):
        return list(map(list, zip(*a)))

    def _replace_none(self, a):
        l = 0
        for x in a:
            if isinstance(x, list):
                l = len(x)
                break

        none_val = [None for _ in range(l)] if l > 0 else []

        for i in range(len(a)):
            if a[i] is None:
                a[i] = none_val.copy()

        return a

    def to_cuda(self, device, non_blocking):
        question_index = self._question_index.cuda(device, non_blocking=non_blocking) if self._question_index is not None else None
        res = OperatorBatch(self._op_name, self._arguments.copy(), self._question_num, self._is_terminal, mask=self._mask.cuda(device, non_blocking=non_blocking), question_index=question_index, process_args=False)
        
        res._predicate_question_map = self._predicate_question_map.cuda(device, non_blocking=non_blocking) if self._question_index is not None else None
        res._op_id = self._op_id

        return res

    def to(self, dtype):
        self._predicate_question_map = self._predicate_question_map.to(dtype) if self._question_index is not None else None
        self._mask = self._mask.to(dtype)
        return self

    def pin_memory(self):
        if self._mask is not None:
            self._mask = self._mask.pin_memory()

        if self._question_index is not None:
            self._question_index = self._question_index.pin_memory()
        
        return self
    
    def _get_str_representation(self):
        res = "Operation: " + self._op_name + "\nTerminal: " + str(self._is_terminal) + "\n"
        
        if len(self._arguments) > 0:
            res += ("(Argument, Mask): " + str(list(zip(self._transpose(self._arguments), self._mask.cpu().numpy().tolist()))) + "\n")
        else:
            res += ("Mask: " + str(self._mask.cpu().numpy().tolist()) + "\n")

        if self._predicate_question_map is not None:
            res += ("Predicate to Question map: " + str(self._predicate_question_map._indices()[1, :].cpu().numpy().tolist()) + "\n")
        
        return res
    
    def __repr__(self):
        return self._get_str_representation()

    def __str__(self):
        return self._get_str_representation()  

######################################################################################################################################

class ProgramBatch(object):

    def __init__(self, device, op_batch_list, dependencies, answers, object_features, object_batch_index=None, original_dicts=None, meta_data=None):
        self._op_batch_list = op_batch_list
        self._object_features = object_features
        self._dependencies = dependencies
        self._answers = answers
        self._batch_size = self._op_batch_list[0]._question_num if self._op_batch_list is not None and len(self._op_batch_list) > 0 else 0
        self._original_dicts = original_dicts
        self._meta_data = meta_data
        self._device = device

        if object_batch_index is not None:
            if isinstance(object_batch_index, np.ndarray):
                self._object_batch_index = torch.from_numpy(object_batch_index)
            else:
                self._object_batch_index = object_batch_index
        else:
            self._object_batch_index = None

        batch_id = str(int(round(time.time() * 1000)))
        for i in range(len(self._op_batch_list)):
            self._op_batch_list[i]._op_id = batch_id + ':' + str(i)

        self._question_type = QuestionType.QUERY if self._op_batch_list[-1]._op_name in ['query_attr', 'choose_attr', 'choose_rel'] else QuestionType.BINARY

    @property
    def device(self):
        return self._device

    def batch_size(self):
        return self._batch_size
    
    def to_cuda(self, device, non_blocking):
        op_batch_list = [op_batch.to_cuda(device, non_blocking) for op_batch in self._op_batch_list]

        dependencies = util.to_cuda(self._dependencies, device, non_blocking)
        answers = util.to_cuda(self._answers, device, non_blocking)
        object_features = util.to_cuda(self._object_features, device, non_blocking)
        meta_data = util.to_cuda(self._meta_data, device, non_blocking)
        original_dicts = util.to_cuda(self._original_dicts, device, non_blocking)

        # dependencies = self._dependencies.copy() if isinstance(self._dependencies, (list, tuple)) else None 
        # answers = self._answers.copy() if isinstance(self._answers, (list, tuple)) else None
        
        # if isinstance(self._object_features, (list, tuple, dict)):
        #     object_features = self._object_features.copy()
        # elif isinstance(self._object_features, torch.Tensor):
        #     object_features = self._object_features.cuda(device, non_blocking=non_blocking)
        # else:
        #     object_features = None
        
        # if isinstance(self._meta_data, torch.Tensor):
        #     meta_data = self._meta_data.cuda(device, non_blocking=non_blocking)
        # elif isinstance(self._meta_data, dict):
        #     meta_data = self._meta_data.copy()
        #     for key, val in self._meta_data.items():
        #         if isinstance(val, torch.Tensor):
        #             meta_data[key] = val.cuda(device, non_blocking=non_blocking)
        # else:
        #     meta_data = self._meta_data
        
        # original_dicts = self._original_dicts.copy() if self._original_dicts is not None else None

        object_batch_index = self._object_batch_index.cuda(device, non_blocking=non_blocking) if self._object_batch_index is not None else None
        
        return ProgramBatch(device, op_batch_list, dependencies, answers, object_features, object_batch_index=object_batch_index, original_dicts=original_dicts, meta_data=meta_data)

    def to(self, dtype):
        self._op_batch_list = [op_batch.to(dtype) for op_batch in self._op_batch_list]
        self._object_features = util.to(self._object_features, dtype)
        self._meta_data = util.to(self._meta_data, dtype)
        self._object_batch_index = self._object_batch_index.to(dtype) if self._object_batch_index is not None else None
        
        return self

    def pin_memory(self):
        if isinstance(self._object_features, torch.Tensor):
            self._object_features = self._object_features.pin_memory()

        if isinstance(self._meta_data, torch.Tensor):
            self._meta_data = self._meta_data.pin_memory()
        elif isinstance(self._meta_data, dict):
            for key, val in self._meta_data.items():
                if isinstance(val, torch.Tensor):
                    self._meta_data[key] = val.pin_memory()

        if self._object_batch_index is not None:
            self._object_batch_index = self._object_batch_index.pin_memory()
        
        for i in range(len(self._op_batch_list)):
            self._op_batch_list[i] = self._op_batch_list[i].pin_memory()
        
        return self

    def create_sparse_tensors(self):
        for op_batch in self._op_batch_list:
            op_batch.create_sparse_map()

    def _get_str_representation(self):
        res = ""

        for i, op_batch in enumerate(self._op_batch_list):
            res += ("#" + str(i) + "\n") 
            res += op_batch._get_str_representation()

        res += ("Dependencies: " + str(self._dependencies) + "\n")
        res += ("Answers: " + str(self._answers) + "\n")

        if self._object_features is not None: #and isinstance(self._object_features, list):
            res += ("Object features: " + str(self._object_features) + "\n")

        if self._object_batch_index is not None:
            res += ("Batch index: " + str(self._object_batch_index) + "\n")

        if self._meta_data is not None:
            res += ("Meta data: " + str(self._meta_data) + "\n")

        return res
    
    def __repr__(self):
        return self._get_str_representation()

    def __str__(self):
        return self._get_str_representation()

    def retrieve_instance(self, index, trace=None):
        res = []
        if index >= self._batch_size:
            return res

        for i, op_batch in enumerate(self._op_batch_list):
            if op_batch._mask is not None:
                if op_batch._mask[index] == 0:
                    continue

            args = [a[index] for a in op_batch._arguments]

            if trace is not None and isinstance(trace[i], BatchVariableSet):
                res.append((op_batch._op_name, args, trace[i]._name[index], trace[i]._log_attention[index, :].exp().cpu().numpy()))
            else:
                res.append((op_batch._op_name, args))
        
        return res

######################################################################################################################################

class ProgramDataset(data.Dataset):

    def __init__(self, input_file, ontology, in_memory, max_cache_size=100000, keep_original_dict=False):
        super(ProgramDataset, self).__init__()

        self._input_file = input_file

        if isinstance(input_file, str):
            _, ext = splitext(input_file)       
            self._is_h5 = (ext == '.h5')
        else:
            self._is_h5 = False

        self._in_memory = in_memory or isinstance(self._input_file, (list, tuple)) #and self._is_h5

        if not self._in_memory:
            self._cache = collections.OrderedDict()
            self._max_cache_size = max_cache_size
        else:
            self._cache = {}

        if not self._is_h5:
            if isinstance(input_file, str):
                with open(self._input_file, 'r') as fh_input:
                    if self._in_memory:
                        self._data = fh_input.readlines()
                        self._row_num = len(self._data)
                    else:
                        self._row_num = len(fh_input.readlines())
            else:
                self._data = input_file
                self._row_num = len(input_file)
        else:
            self._h5_handle = None
            with h5py.File(self._input_file, 'r') as file:
                self._row_num = file['image_id'].shape[0]
        
        self._keep_original_dict = keep_original_dict
        self._ontology = ontology

    def __len__(self):
        return self._row_num

    def __getitem__(self, idx):
        if not self._in_memory:
            if idx in self._cache:
                return self._cache[idx]

        if self._is_h5:
            if self._h5_handle is None:
                self._h5_handle = h5py.File(self._input_file, 'r')
            
            json_obj = {'imageId': self._ontology.decode_img_id(self._load_datapoint('image_id', idx))}
            json_obj['answer'] = self._ontology.decode_token(self._load_datapoint('answer', idx))
            l_op, l_arg = self._decode_op_args(self._load_datapoint('last_op', idx), self._load_datapoint('last_args', idx))
            last_op = {'operator': l_op, 'arguments': l_arg}

            b_ops_array = self._load_datapoint('branch_ops', idx)
            b_args_array = self._load_datapoint('branch_args', idx)

            branch_num, branch_length = b_ops_array.shape
            branches = []

            for i in range(branch_num):
                branch = []
                for j in range(branch_length):
                    if b_ops_array[i, j] == 0:
                        break

                    b_op, b_arg = self._decode_op_args(b_ops_array[i, j], b_args_array[i, j, :])
                    branch.append({'operator': b_op, 'arguments': b_arg})
                branches.append(branch)

            json_obj['program'] = {'branches': branches, 'last_op': last_op}            
        else:
            line = self._data[idx] if self._in_memory else linecache.getline(self._input_file, idx + 1)
            json_obj = json.loads(line) if isinstance(line, str) else line

        result = self._transform_line(json_obj)

        if not self._in_memory:
            if len(self._cache) >= self._max_cache_size:
                self._cache.popitem(last=False)

            self._cache[idx] = result
        
        return result

    def _load_datapoint(self, section, idx):
        if self._in_memory:
            if section not in self._cache:
                self._cache[section] = self._h5_handle[section][...]
            
            return self._cache[section][idx]
        else:
            return self._h5_handle[section][idx]

    def _decode_op_args(self, op_code, arg_codes):
        op = self._ontology.decode_op(op_code)
        method = getattr(self, '_decode_' + op)
        args = method(arg_codes)

        return op, args
    
    def _decode_select(self, arg_codes):
        return [self._ontology.decode_token(arg_codes[0])]

    def _decode_filter(self, arg_codes):
        return [self._ontology.decode_token(arg_codes[0])]
    
    def _decode_relate(self, arg_codes):
        return [self._ontology.decode_token(arg_codes[i]) for i in range(3)]

    def _decode_query_attr(self, arg_codes):
        return [self._ontology.decode_token(arg_codes[0])]

    def _decode_choose_attr(self, arg_codes):
        return [[self._ontology.decode_token(arg_codes[i]) for i in range(2)]]

    def _decode_verify_attr(self, arg_codes):
        return [self._ontology.decode_token(arg_codes[0])]

    def _decode_verify_attrs(self, arg_codes):
        res = [self._ontology.decode_token(arg_codes[0])]
        if arg_codes[1] != 0:
            res.append(self._ontology.decode_token(arg_codes[1]))
        return [res]

    def _decode_choose_rel(self, arg_codes):
        return [[self._ontology.decode_token(arg_codes[i]) for i in range(2)], self._ontology.decode_token(arg_codes[2]), self._ontology.decode_token(arg_codes[3])]

    def _decode_verify_rel(self, arg_codes):
        return [self._ontology.decode_token(arg_codes[i]) for i in range(3)]

    def _decode_exist(self, arg_codes):
        return []
    
    def _decode_and(self, arg_codes):
        return []

    def _decode_or(self, arg_codes):
        return []

    def _decode_end(self, arg_codes):
        return []

    def _decode_all_same(self, arg_codes):
        return [self._ontology.decode_token(arg_codes[0])]

    def _decode_all_different(self, arg_codes):
        return [self._ontology.decode_token(arg_codes[0])]

    def _decode_two_same(self, arg_codes):
        return [self._ontology.decode_token(arg_codes[0])]

    def _decode_two_different(self, arg_codes):
        return [self._ontology.decode_token(arg_codes[0])]

    def _decode_compare(self, arg_codes):
        return [self._ontology.decode_token(arg_codes[i]) for i in range(2)]
    
    def _normalize(self, token_list):
        # Detect negations
        
        is_negated = [re.match("not\((\w|\s)+\)", a.strip()) is not None for a in token_list]
        any_negated = any(is_negated)
        
        if any_negated:
            b_list = []
            for i, a in enumerate(token_list):
                if is_negated[i]:
                    b_list.append(a.strip()[4:-1])
                else:
                    b_list.append(a.strip())
        else:
            b_list = token_list
        
        return b_list
    
    def _extract_select(self, op_arguments, name):
        args = self._normalize(op_arguments)
        return args, "entity" if args[0] is None or args[0].lower() in ["_", "scene"] else args[0]

    def _extract_filter(self, op_arguments, name):
        args = self._normalize(op_arguments)
        return args, args[0] if self._ontology.is_noun(args[0]) else name
    
    def _extract_relate(self, op_arguments, name):
        args0 = self._normalize([op_arguments[0]])
        args2 = self._normalize([op_arguments[2]])
        return args0 + args2, "entity" if args2[0] is None or args2[0].lower() in ["_", "scene"] else args2[0]

    def _extract_query_attr(self, op_arguments, name):
        attribute_list = self._ontology.query(op_arguments[0] if op_arguments[0] not in ['name', 'type'] else name)
        attribute_list.append(op_arguments[0])
        return self._normalize(attribute_list), name

    def _extract_choose_attr(self, op_arguments, name):
        return self._normalize(op_arguments[0]), name

    def _extract_verify_attr(self, op_arguments, name):
        return self._normalize(op_arguments), name

    def _extract_verify_attrs(self, op_arguments, name):
        return self._normalize(op_arguments[0]), name

    def _extract_choose_rel(self, op_arguments, name):
        args0 = self._normalize(op_arguments[0])
        args2 = self._normalize([op_arguments[2]])
        return args0 + args2, "entity" if args2[0] is None or args2[0].lower() in ["_", "scene"] else args2[0]

    def _extract_verify_rel(self, op_arguments, name):
        args0 = self._normalize([op_arguments[0]])
        args2 = self._normalize([op_arguments[2]])
        return args0 + args2, "entity" if args2[0] is None or args2[0].lower() in ["_", "scene"] else args2[0]

    def _extract_exist(self, op_arguments, name):
        return [], name
    
    def _extract_and(self, op_arguments, name):
        return [], name

    def _extract_or(self, op_arguments, name):
        return [], name

    def _extract_end(self, op_arguments, name):
        return [], name

    def _extract_all_same(self, op_arguments, name):
        attribute_list = self._ontology.query(op_arguments[0] if op_arguments[0] not in ['name', 'type'] else name)
        attribute_list.append(op_arguments[0])
        return self._normalize(attribute_list), name

    def _extract_all_different(self, op_arguments, name):
        attribute_list = self._ontology.query(op_arguments[0] if op_arguments[0] not in ['name', 'type'] else name)
        attribute_list.append(op_arguments[0])
        return self._normalize(attribute_list), name

    def _extract_two_same(self, op_arguments, name):
        attribute_list = self._ontology.query(op_arguments[0] if op_arguments[0] not in ['name', 'type'] else name)
        attribute_list.append(op_arguments[0])
        return self._normalize(attribute_list), name

    def _extract_two_different(self, op_arguments, name):
        attribute_list = self._ontology.query(op_arguments[0] if op_arguments[0] not in ['name', 'type'] else name)
        attribute_list.append(op_arguments[0])
        return self._normalize(attribute_list), name

    def _extract_compare(self, op_arguments, name):
        return self._normalize([op_arguments[0]]), name

    def _extract_object_attr(self, op_arguments, name):
        return self._normalize(list(sum(op_arguments[0], []))), name

    def _extract_object_rel(self, op_arguments, name):
        return self._normalize(op_arguments[0]), name

    def _extract_scene(self, op_arguments, name):
        return [], name

    def _collect_tokens(self, program):
        all_tokens = []
        name = ''
        
        for b in program['branches']:
            name = ''
            for op in b:
                method = getattr(self, '_extract_' + op['operator'])
                tokens, name = method(op['arguments'], name)
                all_tokens += tokens

        method = getattr(self, '_extract_' + program['last_op']['operator'])
        tokens, name = method(program['last_op']['arguments'], name)
        all_tokens += tokens

        return list(set(all_tokens))            

    def _transform_answer(self, op_name, answer):
        if answer is None:
            return None

        if isinstance(answer, (list, tuple)):
            if len(answer) == 0:
                res = []
            elif isinstance(answer[0], (list, tuple)):
                res = list(sum(answer, []))
                res = [a.lower().strip() for a in res]
            else:
                res = [a.lower().strip() for a in answer]
        else:
            res = answer.lower().strip()
            if op_name == 'choose_rel':
                if res == 'left':
                    res = 'to the left of'
                elif res == 'right':
                    res = 'to the right of'

        return res
    
    def _transform_line(self, q):
        op = q['program']['last_op']['operator']

        if op in ('choose_rel','choose_attr'):
            shuffle(q['program']['last_op']['arguments'][0])

        if 'answer' not in q:
            q['answer'] = ""
        
        all_tokens = self._collect_tokens(q['program'])
        result = {'program': q['program'], 'image_id': q['imageId'], 'answer': self._transform_answer(q['program']['last_op']['operator'], q['answer']), 'tokens': all_tokens,\
                'original_dict': q if self._keep_original_dict else None, 'question': q['question'] if not self._is_h5 and 'question' in q else None,
                'question_id': q['question_id'] if not self._is_h5 and 'question_id' in q else None}
        
        if 'object_pairs' in q:
            result['object_pairs'] = q['object_pairs']

        if 'weights' in q:
            if len(q['weights']) > 0 and isinstance(q['weights'][0], (list, tuple)):
                result['weights'] = list(sum(q['weights'], []))
            else:
                result['weights'] = q['weights']

        if 'attribute_dict' in q:
            result['attribute_dict'] = q['attribute_dict']

        if 'relation_list' in q:
            result['relation_list'] = q['relation_list']

        return result

######################################################################################################################################

class ProgramCollaterBase(object):

    def __init__(self, starter_op, sep_op, filler_op, split_num=1):
        self._sep_op = sep_op
        self._filler_op = filler_op
        self._starter_op = starter_op
        self._split_num = split_num

    def _create_op_dict(self, branch_list):
        result = {}
        index = 0
        
        for branch in branch_list:
            if branch is not None:
                for op in branch:
                    if op['operator'] not in result:
                        result[op['operator']] = {'index': index, 'op_batch_list': []}
                        index += 1

        return result

    def collate_programs(self, questions):
        batch_size = len(questions)
        final_batch_list = []
        final_dependencies = []
        dependency_offset = -1
        last_op_dependency = []

        max_branch_num = max([len(q['program']['branches']) for q in questions])

        for i in range(max_branch_num):
            # collate the starter op
            # print('a', i, [(len(q['program']['branches']), q['program']['last_op']['operator']) for q in questions])
            # print('b', [[len(b) for b in q['program']['branches']] for q in questions])
            args = [q['program']['branches'][i][0]['arguments'] if q['program']['branches'][i][0]['operator'] == self._starter_op \
                    else ['_'] for q in questions]
            mask = np.ones(batch_size, dtype=np.float32)
            final_batch_list += [OperatorBatch(self._starter_op, args, batch_size, False, mask=mask)]
            final_dependencies.append([])
            dependency_offset += 1

            # Collate the fillers and separators
            
            filler_list = []
            sep_list = []

            for k in range(batch_size):
                filler_ind, sep_ind = 0, 0

                length = len(questions[k]['program']['branches'][i])

                for j in range(1, length):
                    op = questions[k]['program']['branches'][i][j]

                    if op['operator'] == self._filler_op:
                        if sep_ind >= len(filler_list):
                            for _ in range(sep_ind - len(filler_list) + 1):
                                filler_list.append([])
                            
                            filler_ind = 0

                        if filler_ind >= len(filler_list[sep_ind]):
                            args = [None for _ in range(batch_size)]
                            mask = np.zeros(batch_size, dtype=np.float32)
                            filler_list[sep_ind].append({'arguments': args, 'mask': mask})

                        filler_list[sep_ind][filler_ind]['mask'][k] = 1.0
                        filler_list[sep_ind][filler_ind]['arguments'][k] = op['arguments']
                        filler_ind += 1
                    elif op['operator'] == self._sep_op:
                        if sep_ind >= len(sep_list):
                            args = [None for _ in range(batch_size)]
                            mask = np.zeros(batch_size, dtype=np.float32)
                            sep_list.append({'arguments': args, 'mask': mask})

                        sep_list[sep_ind]['mask'][k] = 1.0
                        sep_list[sep_ind]['arguments'][k] = op['arguments']
                        sep_ind += 1
                        filler_ind = 0

            t = max(len(sep_list), len(filler_list))
            for n in range(t):
                if len(filler_list) > n:
                    for d in filler_list[n]:
                        final_batch_list.append(OperatorBatch(self._filler_op, d['arguments'], batch_size, False, mask=d['mask']))
                        final_dependencies.append([dependency_offset])
                        dependency_offset += 1

                if len(sep_list) > n:
                    final_batch_list.append(OperatorBatch(self._sep_op, sep_list[n]['arguments'], batch_size, False, mask=sep_list[n]['mask']))
                    final_dependencies.append([dependency_offset])
                    dependency_offset += 1

            last_op_dependency.append(dependency_offset)

        # Collate the last operators
        op_dict = {}
        for k, q in enumerate(questions):
            op = q['program']['last_op']['operator']
            args = q['program']['last_op']['arguments']
            
            if op not in op_dict:
                # Create a new mask
                mask = np.zeros(batch_size, dtype=np.float32)
                mask[k] = 1.0

                # Create args list
                arg_list = [None for _ in range(batch_size)]
                arg_list[k] = args

                op_dict[op] = {'arguments': arg_list, 'mask': mask}
            else:
                op_dict[op]['arguments'][k] = args
                op_dict[op]['mask'][k] = 1.0

        for op_name, val in op_dict.items():
            final_batch_list.append(OperatorBatch(op_name, val['arguments'], batch_size,\
                        True, mask=val['mask']))
            final_dependencies.append(last_op_dependency)

        return final_batch_list, final_dependencies

    def collate_object_features(self, questions):
        return None, None
    
    def collate_meta_data(self, questions):
        return None

    def collate(self, questions):
        result = []
        n = len(questions)
        split_num = min(self._split_num, n)
        split_size = math.ceil(n / split_num)
        start = 0
        end = split_size
        device = torch.device('cpu')

        for i in range(split_num):
            if start >= end:
                break

            op_batch_list, dependencies = self.collate_programs(questions[start:end])
            object_features, object_batch_index = self.collate_object_features(questions[start:end])
            meta_data = self.collate_meta_data(questions[start:end])
            
            answers = [q['answer'] for q in questions[start:end]]
            original_dicts = [q['original_dict'] for q in questions[start:end]]

            result.append(ProgramBatch(device, op_batch_list, dependencies, answers, object_features, object_batch_index, original_dicts, meta_data=meta_data))
            le = len(result[-1]._op_batch_list)
            for j in range(le):
                result[-1]._op_batch_list[j]._op_id = str(i) + ':' + result[-1]._op_batch_list[j]._op_id
            
            start += split_size
            end += split_size
            end = min(end, n)

        return result

######################################################################################################################################

class MultiSetSampler(data.sampler.Sampler):
    
    def __init__(self, dataset_list, batch_size, drop_last, replacement=False, distributed=False):
        self._datasets = dataset_list
        self._distributed = distributed

        if distributed:
            samplers = [DistributedSampler(ds, shuffle=True) for ds in self._datasets]
            self._batch_samplers = [data.sampler.BatchSampler(s, batch_size, drop_last) for s in samplers]
            self._lengths = [len(ds) for ds in samplers]
            self._cumulative_lengths = data.ConcatDataset.cumsum(self._datasets)
        else:
            self._batch_samplers = [data.sampler.BatchSampler(data.sampler.RandomSampler(ds, replacement=replacement), batch_size, drop_last) for ds in self._datasets]
            self._lengths = [len(ds) for ds in self._datasets]
            self._cumulative_lengths = data.ConcatDataset.cumsum(self._datasets)
        
        self._num_samples = sum(self._lengths)

    def __len__(self):
        return self._num_samples

    def __iter__(self):
        lengths = torch.tensor(self._lengths, dtype=torch.float32)
        iterators = [iter(s) for s in self._batch_samplers]

        while lengths.sum() > 0:
            dataset_index = torch.multinomial(lengths, 1).cpu().numpy()[0]
            batch = next(iterators[dataset_index])
            lengths[dataset_index] = torch.max(lengths[dataset_index] - len(batch), torch.zeros_like(lengths[dataset_index]))

            if dataset_index > 0:
                batch = [self._cumulative_lengths[dataset_index - 1] + i for i in batch]

            yield batch

    def set_epoch(self, epoch):
        if self._distributed:
            for sampler in self._batch_samplers:
                sampler.sampler.set_epoch(epoch)

######################################################################################################################################

class MultiSetSequencialSampler(data.sampler.Sampler):
    
    def __init__(self, dataset_list, batch_size, drop_last, distributed=False):
        self._datasets = dataset_list
        self._dataset_num = len(dataset_list)
        self._distributed = distributed

        if distributed:
            samplers = [DistributedSampler(ds, shuffle=False) for ds in self._datasets]
            self._batch_samplers = [data.sampler.BatchSampler(s, batch_size, drop_last) for s in samplers]
            self._lengths = [len(ds) for ds in samplers]
            self._cumulative_lengths = data.ConcatDataset.cumsum(self._datasets)
        else:
            self._batch_samplers = [data.sampler.BatchSampler(data.sampler.SequentialSampler(ds), batch_size, drop_last) for ds in self._datasets]
            self._lengths = [len(ds) for ds in self._datasets]
            self._cumulative_lengths = data.ConcatDataset.cumsum(self._datasets)
        
        self._num_samples = sum(self._lengths)

    def __len__(self):
        return self._num_samples

    def __iter__(self):
        lengths = torch.tensor(self._lengths, dtype=torch.float32)
        iterators = [iter(s) for s in self._batch_samplers]
        dataset_index = 0

        while dataset_index < self._dataset_num:
            batch = next(iterators[dataset_index])
            lengths[dataset_index] = torch.max(lengths[dataset_index] - len(batch), torch.zeros_like(lengths[dataset_index]))

            if dataset_index > 0:
                batch = [self._cumulative_lengths[dataset_index - 1] + i for i in batch]

            if lengths[dataset_index] <= 0:
                dataset_index += 1

            yield batch

    def set_epoch(self, epoch):
        if self._distributed:
            for sampler in self._batch_samplers:
                sampler.sampler.set_epoch(epoch)

######################################################################################################################################

class GQADataManager(object):

    def __init__(self, data_path, ontology, in_memory, max_cache_size=100000, keep_original_dict=False):
        if isinstance(data_path, (list, tuple)):
            datasets = [ProgramDataset(data_path, ontology, in_memory, max_cache_size, keep_original_dict)]
        else:
            file_names = [join(data_path, f) for f in listdir(data_path) if isfile(join(data_path, f)) and (f.endswith('.json') or f.endswith('.h5'))]
            datasets = [ProgramDataset(f, ontology, in_memory, max_cache_size, keep_original_dict) for f in file_names]

        self._dataset = data.ConcatDataset(datasets)

    def get_loader(self, batch_size, collater, num_workers, use_cuda, drop_last=False, replacement=False, is_random=True, distributed=False):
        
        if is_random:
            sampler = MultiSetSampler(self._dataset.datasets, batch_size, drop_last, replacement, distributed=distributed)
        else:
            sampler = MultiSetSequencialSampler(self._dataset.datasets, batch_size, drop_last, distributed=distributed)

        data_loader = data.DataLoader(
            dataset=self._dataset,
            batch_sampler=sampler,
            num_workers=num_workers,
            collate_fn=collater.collate,
            pin_memory=use_cuda)

        return data_loader
        

if __name__ == '__main__':
    collater = ProgramCollaterBase('select', 'relate', 'filter')
    
    dataset = ProgramDataset(r'E:\datasets\GQA\p_questions1.2\ppp_val_balanced_questions_choose_attr.json')
    # dataset = ProgramDataset(r'/Users/saeed/Datasets/GQA/p_questions1.2/pp_val_balanced_questions_and.json')

    data_loader = data.DataLoader(
        dataset=dataset,
        batch_size=100,
        shuffle=False,
        num_workers=1,
        collate_fn=collater.collate)

    # # data_manager = GQADataManager(r'E:\datasets\GQA\p_questions1.2')
    # data_manager = GQADataManager('/Users/saeed/Datasets/GQA/p_questions1.2')
    # data_loader = data_manager.get_loader(5, collater, 0, False)
    # device = torch.device('cpu')

    for i, data in enumerate(data_loader):
        if i > 0:
            break
        
        data.create_sparse_tensors()
        # data = data.to_cuda(device, True)
        print(data)
