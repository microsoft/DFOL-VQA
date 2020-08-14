# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file
# in the project root for full license information.

import json, re, h5py, argparse
import numpy as np
from os import listdir, makedirs
from os.path import splitext, isdir, isfile, join, split, exists

from nsvqa.nn.interpreter.batch_gqa_ops import GQAOntology
from nsvqa.nn.parser.parse_utils import normalize

######################################################################################################################################

class GQAH5Encoder(object):

    def __init__(self, ontology):
        self._ontology = ontology
        self._max_branch_length = 10 #4

    def _find_sizes(self, input_file):
        with open(input_file) as f:
            for i, _ in enumerate(f):
                pass
        
            row_n = i + 1

        with open(input_file) as f:
            for line in f:
                q = json.loads(line)
                op = q['program']['last_op']['operator']

                if op in ['verify_attrs', 'choose_attr', 'compare']:
                    arg_n = 2
                elif op in ['verify_rel']:
                    arg_n = 3
                elif op in ['choose_rel']:
                    arg_n = 4
                else:
                    arg_n = 1

                if op in ['and', 'or', 'two_same', 'two_different', 'compare']:
                    branch_n = 2
                else:
                    branch_n = 1
                
                break

        return row_n, branch_n, arg_n

    def encode(self, input_path, output_path):
        file_names = [f for f in listdir(input_path) if isfile(join(input_path, f)) and f.endswith('.json')]

        for file in file_names:
            print(file)
            input_file = join(input_path, file)
            row_n, branch_n, arg_n = self._find_sizes(input_file)

            answer = np.zeros(row_n, dtype=np.int32)
            image_id = np.zeros(row_n, dtype=np.int32)
            branch_ops = np.zeros((row_n, branch_n, self._max_branch_length), dtype=np.int32)
            branch_args = np.zeros((row_n, branch_n, self._max_branch_length, 3), dtype=np.int32)
            last_op = np.zeros(row_n, dtype=np.int32)
            last_args = np.zeros((row_n, arg_n), dtype=np.int32)
            
            with open(input_file, 'r') as f:
                for i, line in enumerate(f):
                    q = json.loads(line)
                
                    image_id[i] = self._ontology.encode_img_id(q['imageId'])
                    answer[i] = self._ontology.encode_token(q['answer'])

                    for j, b in enumerate(q['program']['branches']):
                        for k, op in enumerate(b):
                            branch_ops[i, j, k] = self._ontology.encode_op(op['operator'])

                            flat_args = [item for sublist in op['arguments'] for item in (sublist if isinstance(sublist, list) else [sublist])]
                            for t, arg in enumerate(flat_args):
                                branch_args[i, j, k, t] = self._ontology.encode_token(arg)

                    last_op[i] = self._ontology.encode_op(q['program']['last_op']['operator'])
                    flat_args = [item for sublist in q['program']['last_op']['arguments'] for item in (sublist if isinstance(sublist, list) else [sublist])]
                    for t, arg in enumerate(flat_args):
                        last_args[i, t] = self._ontology.encode_token(arg)

            fname, _ = splitext(file)
            hf = h5py.File(join(output_path, fname + '.h5'), 'w')
            hf.create_dataset('answer', data=answer)
            hf.create_dataset('image_id', data=image_id)
            hf.create_dataset('branch_ops', data=branch_ops)
            hf.create_dataset('branch_args', data=branch_args)
            hf.create_dataset('last_op', data=last_op)
            hf.create_dataset('last_args', data=last_args)
            hf.close()
    
######################################################################################################################################

class GQAPreprocessor(object):

    STARTER_OPS = ['select']
    TRACE_CHANGER_OPS = ['relate']
    LOGICAL_OPS = ['and', 'or']

    def __init__(self, map_json_path, is_batch_format):
        self._op_map = json.load(open(map_json_path, 'r'))
        self._is_batch_format = is_batch_format

    def _dump_per_line(self, output, out_file):
        with open(out_file, 'a') as f:
            for _, value in output.items():
                f.write(json.dumps(value) + '\n')

    def preprocess(self, in_file, out_file, segregate, length_segregation, discard_global=False):
        if isdir(in_file):
            file_names = [join(in_file, f) for f in listdir(in_file) if isfile(join(in_file, f)) and (f.endswith('.json') or f.endswith('.txt'))]
        else:
            file_names = [in_file]
            
        fname, ext = splitext(out_file)

        for file in file_names:
            print(file)      
            output = {}
            
            with open(file, 'r') as f:
                data = json.load(f)

                for key, value in data.items():
                    if isinstance(value, dict):
                        q = self.parse_question(value, discard_global)
                        if q is None:
                            continue

                        q['question_id'] = key
                        
                        if segregate:
                            op = q['program']['last_op']['operator'] if self._is_batch_format else q['operators'][-1]

                            if length_segregation:
                                op = op + '_' + str(len(q['program']['branches'][0]))

                            if op not in output:
                                output[op] = {}

                            output[op][key] = q
                        else:
                            output[key] = q

            if segregate:
                # if self._is_batch_format:
                for op, value in output.items():
                    self._dump_per_line(value, fname + '_' + op + ext)
                # else:
                #     for op, value in output.items():
                #         with open(fname + '_' + op + ext, 'w') as f:
                #             json.dump(value, f)
            else:
                # if self._is_batch_format:
                self._dump_per_line(output, out_file)
                # else:
                #     with open(out_file, 'w') as f:
                #         json.dump(output, f)

            del output
        

    def parse_question(self, question, discard_global):
        # Filter out 'global' questions
        if discard_global and question['semantic'][0]['operation'] == 'select' and question['semantic'][0]['argument'] == 'scene':
            return None

        ops, args, deps = self.parse_program(question['semantic'])
        if None in ops or None in args:
            return None

        # Combine all 'verify_attr-and' to 'verify_attrs'
        trace, _ = self._compute_op_trace(ops, deps)
        ops, args, deps, trace = self._combine_verify(ops, args, deps, trace)

        if self._is_batch_format:
            question['program'] = self._de_branch_program(ops, args, deps)
            question['program'] = self._fix_logical_branches(question['program'])
        else:
            question['operators'] = ops
            question['arguments'] = args
            question['dependencies'] = deps

        question['answer'] = normalize(question['answer'])
        # print(question['semantic'], ' ---> ', ops, args, deps)
        return question

    def parse_program(self, program):
        op_arg = [self.parse_operation(p['operation'], p['argument']) for p in program]
        op_arg = list(zip(*op_arg))
        return list(op_arg[0]), list(op_arg[1]), [p['dependencies'] for p in program]

    def _fix_logical_branches(self, program):
        if program['last_op']['operator'] in GQAPreprocessor.LOGICAL_OPS:
            for i in range(len(program['branches'])):
                if program['branches'][i][-1]['operator'] == 'exist':
                    program['branches'][i] = program['branches'][i][:-1]
                elif program['branches'][i][-1]['operator'] == 'verify_rel':
                    program['branches'][i][-1]['operator'] = 'relate'
                elif program['branches'][i][-1]['operator'] == 'verify_attrs':
                    args = program['branches'][i][-1]['arguments']
                    program['branches'][i][-1]['operator'] = 'filter'
                    program['branches'][i][-1]['arguments'] = [args[0][0]]

                    l = len(args[0]) - 1
                    for j in range(l):
                        program['branches'][i].append({'operator': 'filter', 'arguments': [args[0][j + 1]]})

        return program

    def _compute_op_trace(self, operators, dependencies):
        trace_id = []
        trace_num = -1

        for op, dep in zip(operators, dependencies):
            if op in GQAPreprocessor.STARTER_OPS + GQAPreprocessor.TRACE_CHANGER_OPS:
                trace_num += 1
                trace_id.append(trace_num)
            else:
                trace_id.append(trace_id[dep[0]])

        return trace_id, trace_num

    def _combine_verify(self, operators, arguments, dependencies, trace):
        if operators[-1] == 'and' and all([operators[i] == 'verify_attrs' for i in dependencies[-1]]):
            if trace[dependencies[-1][0]] == trace[dependencies[-1][1]]:
                first_ind = min(dependencies[-1])
                second_ind = max(dependencies[-1])

                # Shift the dependency indices
                for i, dep in enumerate(dependencies):
                    for j, d in enumerate(dep):
                        if d > first_ind:
                            dependencies[i][j] = d - 1

                arguments[second_ind] = [[arguments[first_ind][0][0], arguments[second_ind][0][0]]]
                
                del operators[first_ind]
                del arguments[first_ind]
                del dependencies[first_ind]
                del trace[first_ind]

                return operators[:-1], arguments[:-1], dependencies[:-1], trace[:-1]

        return operators, arguments, dependencies, trace

    def _de_branch_program(self, operators, arguments, dependencies):
        branch_num = -1
        branch_id = []
        
        # Cutting two-branch programs        
        for i in range(len(operators) - 1):
            
            if operators[i] in GQAPreprocessor.STARTER_OPS:
                branch_num += 1
                branch_id.append(branch_num)
            elif dependencies[i] is not None and len(dependencies) > 0:
                branch_id.append(branch_id[dependencies[i][0]])
            elif i > 0:
                branch_id.append(branch_id[i - 1])
            else:
                raise ValueError('Operator not recognized.')

        branch_num += 1
        ops = [[] for _ in range(branch_num)]

        for i in range(len(operators) - 1):
            ops[branch_id[i]].append({'operator': operators[i], 'arguments': arguments[i]})

        return {'branches': ops, 'last_op': {'operator': operators[-1], 'arguments': arguments[-1]}}

    def parse_operation(self, operator, argument):
        if operator not in self._op_map:
            return None, None
        
        op = self._op_map[operator]

        if op is None:
            return None, None
        
        arg = re.sub('\((\d|,|\s)+\)|\((-|\s)*\)', '', argument).strip()
        op_tokens = re.split(' ', operator)
        arg_tokens = re.split(',', arg)
        method = getattr(self, '_parse_' + op)
        
        return op, method(op_tokens, arg_tokens)

    def _parse_select(self, op_tokens, arg_tokens):
        return (normalize(arg_tokens[0]),)

    def _parse_filter(self, op_tokens, arg_tokens):
        return (normalize(arg_tokens[0]),)
    
    def _parse_relate(self, op_tokens, arg_tokens):
        return (normalize(arg_tokens[1]), arg_tokens[2] == 's', normalize(arg_tokens[0]))

    def _parse_query_attr(self, op_tokens, arg_tokens):
        return (normalize(arg_tokens[0]),)

    def _parse_choose_attr(self, op_tokens, arg_tokens):
        arg_tokens = re.split('\|', arg_tokens[0])
        return ([normalize(t) for t in arg_tokens],)

    def _parse_verify_attr(self, op_tokens, arg_tokens):
        return (normalize(arg_tokens[0]),)

    def _parse_verify_attrs(self, op_tokens, arg_tokens):
        return ([normalize(t) for t in arg_tokens],)

    def _parse_choose_rel(self, op_tokens, arg_tokens):
        rels = re.split('\|', arg_tokens[1])
        rels = [normalize(r) for r in rels]
        return (rels, arg_tokens[2] == 's', normalize(arg_tokens[0]))

    def _parse_verify_rel(self, op_tokens, arg_tokens):
        return (normalize(arg_tokens[1]), arg_tokens[2] == 's', normalize(arg_tokens[0]))

    def _parse_exist(self, op_tokens, arg_tokens):
        return ()
    
    def _parse_and(self, op_tokens, arg_tokens):
        return ()

    def _parse_or(self, op_tokens, arg_tokens):
        return ()

    def _parse_end(self, op_tokens, arg_tokens):
        return ()

    def _parse_all_same(self, op_tokens, arg_tokens):
        return (normalize(arg_tokens[0]),)

    def _parse_all_different(self, op_tokens, arg_tokens):
        return (normalize(arg_tokens[0]),)

    def _parse_two_same(self, op_tokens, arg_tokens):
        op_tokens = [normalize(t) for t in op_tokens[1:]]
        return (' '.join(op_tokens),)

    def _parse_two_different(self, op_tokens, arg_tokens):
        op_tokens = [normalize(t) for t in op_tokens[1:]]
        return (' '.join(op_tokens),)

    def _parse_compare(self, op_tokens, arg_tokens):
        if len(op_tokens) >= 3:
            if normalize(op_tokens[1]) == 'more':
                return (normalize(op_tokens[2]), False)
            elif normalize(op_tokens[1]) == 'less':
                return (normalize(op_tokens[2]), True)

        token = normalize(op_tokens[1])
        if token.endswith('er'):
            token = token[:-2]
            if token.endswith('i'):
                token = token[:-1] + 'y'

        return (token, False)

######################################################################################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', help='The input file')
    parser.add_argument('output_path', help='The output path')
    parser.add_argument('-b', '--h5', help='Generate h5 format', action='store_true')
    parser.add_argument('-l', '--length_segregation', help='Segregate based on length', action='store_true')
    parser.add_argument('-g', '--discard_global', help='Discard global questions', action='store_true')
    args = vars(parser.parse_args())

    input_path, input_file = split(args['input_file'])
    gqap = GQAPreprocessor("./nsvqa/data/metadata/op_map.json", True)

    if isfile(args['input_file']):
        input_file, _ = splitext(input_file)

    output_path = join(args['output_path'], 'p_' + input_file)
    if not exists(output_path):
        makedirs(output_path)

    gqap.preprocess(args['input_file'], join(output_path, 'p_' + input_file + '.json'), True, args['length_segregation'], discard_global=args['discard_global'])

    if args['h5']:
        ATTRIBUTE_PATH = "./nsvqa/data/metadata/gqa_attribute.json"
        CLASS_PATH = "./nsvqa/data/metadata/gqa_class_clean.json"
        VOCAB_PATH = "./nsvqa/data/metadata/gqa_vocab.json"
        
        ontology = GQAOntology(ATTRIBUTE_PATH, CLASS_PATH, VOCAB_PATH)
        encoder = GQAH5Encoder(ontology)
    
        h5_output_path = join(args['output_path'], 'h5_' + input_file)
        if not exists(h5_output_path):
            makedirs(h5_output_path)
        
        encoder.encode(output_path, h5_output_path)
