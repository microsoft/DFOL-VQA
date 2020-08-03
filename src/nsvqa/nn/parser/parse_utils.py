# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file
# in the project root for full license information.

import re
from pattern.text.en import singularize
from nsvqa.nn.interpreter.batch_gqa_ops import GQAOntology

def normalize(string):
    plurale_tantum = ['this', 'yes', 'pants', 'shorts', 'glasses', 'scissors', 'panties', 'trousers', 'binoculars', 'pliers', 'tongs',\
        'tweezers', 'forceps', 'goggles', 'jeans', 'tights', 'leggings', 'chaps', 'boxers', 'indoors', 'outdoors', 'bus', 'octapus', 'waitress',\
        'pasta', 'pita', 'glass', 'asparagus', 'hummus', 'dress', 'cafeteria', 'grass', 'class']

    irregulars = {'shelves': 'shelf', 'bookshelves': 'bookshelf', 'olives': 'olive', 'brownies': 'brownie', 'cookies': 'cookie'}
    
    temp = string.strip().lower()
    if temp in irregulars:
        return irregulars[temp]
    
    return temp if temp.split(' ')[-1] in plurale_tantum or temp[-2:] == 'ss' else singularize(temp)


class ParserError(Exception):
    pass

class GQAProgramVerifier(object):

    def __init__(self, attribute_json_path, class_json_path, vocab_json_path, relation_json_path):
        self._ontology = GQAOntology(attribute_json_path, class_json_path, vocab_json_path, embedding_file=None, relation_json_path=relation_json_path, frequency_json_path=None)

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

    def _is_valid(self, arg):
        return arg in self._ontology._vocabulary['arg_to_idx']

    def _check_argument_num(self, op, arg_num, op_arguments):
        if len(op_arguments) != arg_num:
            raise ParserError("'{0}' must have {1} argument(s), but has {2} argument(s).".format(op, arg_num, len(op_arguments)))
    
    def _verify_select(self, op_arguments):
        self._check_argument_num('select', 1, op_arguments)
        args = self._normalize(op_arguments)
        if args[0].lower() not in ["_", "scene"] and not self._is_valid(args[0].lower()):
            raise ParserError("'select' argument must be a noun: " + args[0])

    def _verify_filter(self, op_arguments):
        self._check_argument_num('filter', 1, op_arguments)
        args = self._normalize(op_arguments)
        if not self._is_valid(args[0].lower()):
            raise ParserError("'filter' argument is not in the vocabulary: " + args[0])
    
    def _verify_relate(self, op_arguments):
        self._check_argument_num('relate', 3, op_arguments)
        args0 = self._normalize([op_arguments[0]])
        args2 = self._normalize([op_arguments[2]])
        
        if not self._ontology.is_relation(args0[0].lower()):
            raise ParserError("'relate' first argument must be a relation: " + args0[0])

        if not isinstance(op_arguments[1], bool):
            raise ParserError("'relate' second argument must be a boolean. Current type: " + str(type(op_arguments[1])))

        if not self._is_valid(args2[0].lower()) and args2[0].lower() not in ["_", "scene"]:
            raise ParserError("'relate' third argument is not in the vocabulary: " + args2[0])

    def _verify_query_attr(self, op_arguments):
        self._check_argument_num('query_attr', 1, op_arguments)
        if op_arguments[0] not in self._ontology._class_dict and op_arguments[0] not in self._ontology._attribute_dict and op_arguments[0] not in ['name', 'type']:
            raise ParserError("'query' has an unknown category argument: ", op_arguments[0])

    def _verify_choose_attr(self, op_arguments):
        self._check_argument_num('choose_attr', 2, op_arguments[0])
        args = self._normalize(op_arguments[0])

        for a in args:
            if not self._is_valid(a.lower()):
                raise ParserError("'choose_attr' argument is not in the vocabulary: " + a)

    # def _verify_verify_attr(self, op_arguments):
    #     args = self._normalize(op_arguments)
    #     if not self._ontology.is_adjective(args[0].lower()) and not self._ontology.is_noun(args[0].lower()):
    #         raise ParserError("'verify_attr' argument is not in the vocabulary: " + args[0])

    def _verify_verify_attrs(self, op_arguments):
        if len(op_arguments) != 1 or len(op_arguments[0]) == 0:
            raise ParserError("'verify_attrs' must have at least one argument.")
        args = self._normalize(op_arguments[0])

        for a in args:
            if not self._is_valid(a.lower()):
                raise ParserError("'verify_attrs' argument is not in the vocabulary: " + a)

    def _verify_choose_rel(self, op_arguments):
        self._check_argument_num('choose_rel', 3, op_arguments)
        if len(op_arguments[0]) == 0:
            raise ParserError("'choose_rel' must at least have one relation.")

        args0 = self._normalize(op_arguments[0])
        args2 = self._normalize([op_arguments[2]])
        
        for r in args0:
            if not self._ontology.is_relation(r.lower()):
                raise ParserError("'choose_rel' first argument must be a relation: " + r)

        if not isinstance(op_arguments[1], bool):
            raise ParserError("'choose_rel' second argument must be a boolean. Current type: " + str(type(op_arguments[1])))

        if not self._is_valid(args2[0].lower()) and args2[0].lower() not in ["_", "scene"]:
            raise ParserError("'choose_rel' third argument is not in the vocabulary: " + args2[0])

    def _verify_verify_rel(self, op_arguments):
        self._check_argument_num('verify_rel', 3, op_arguments)
        args0 = self._normalize([op_arguments[0]])
        args2 = self._normalize([op_arguments[2]])
        
        if not self._ontology.is_relation(args0[0].lower()):
            raise ParserError("'verify_rel' first argument must be a relation: " + args0[0])

        if not isinstance(op_arguments[1], bool):
            raise ParserError("'verify_rel' second argument must be a boolean. Current type: " + str(type(op_arguments[1])))

        if not self._is_valid(args2[0].lower()) and args2[0].lower() not in ["_", "scene"]:
            raise ParserError("'verify_rel' third argument is not in the vocabulary: " + args2[0])

    def _verify_exist(self, op_arguments):
        self._check_argument_num('exist', 0, op_arguments)
    
    def _verify_and(self, op_arguments):
        self._check_argument_num('and', 0, op_arguments)

    def _verify_or(self, op_arguments):
        self._check_argument_num('or', 0, op_arguments)

    # def _verify_end(self, op_arguments):
    #     self._check_argument_num('end', 0, op_arguments)

    def _verify_all_same(self, op_arguments):
        self._check_argument_num('all_same', 1, op_arguments)
        if op_arguments[0] not in self._ontology._class_dict and op_arguments[0] not in self._ontology._attribute_dict and op_arguments[0] not in ['name', 'type']:
            raise ParserError("'all_same' has an unknown category argument: ", op_arguments[0])

    def _verify_all_different(self, op_arguments):
        self._check_argument_num('all_different', 1, op_arguments)
        if op_arguments[0] not in self._ontology._class_dict and op_arguments[0] not in self._ontology._attribute_dict and op_arguments[0] not in ['name', 'type']:
            raise ParserError("'all_different' has an unknown category argument: ", op_arguments[0])

    def _verify_two_same(self, op_arguments):
        self._check_argument_num('two_same', 1, op_arguments)
        if op_arguments[0] not in self._ontology._class_dict and op_arguments[0] not in self._ontology._attribute_dict and op_arguments[0] not in ['name', 'type']:
            raise ParserError("'two_same' has an unknown category argument: ", op_arguments[0])

    def _verify_two_different(self, op_arguments):
        self._check_argument_num('two_different', 1, op_arguments)
        if op_arguments[0] not in self._ontology._class_dict and op_arguments[0] not in self._ontology._attribute_dict and op_arguments[0] not in ['name', 'type']:
            raise ParserError("'two_different' has an unknown category argument: ", op_arguments[0])

    def _verify_compare(self, op_arguments):
        self._check_argument_num('compare', 2, op_arguments)
        args = self._normalize([op_arguments[0]])
        
        if not self._is_valid(args[0].lower()):
            raise ParserError("'compare' first argument must be an adjective: " + args[0])
        
        if not isinstance(op_arguments[1], bool):
            raise ParserError("'compare' second argument must be a boolean. Current type: " + str(type(op_arguments[1])))

    # def _verify_object_attr(self, op_arguments):
    #     self._check_argument_num('select', 1, op_arguments)
    #     return self._normalize(list(sum(op_arguments[0], [])))

    # def _verify_object_rel(self, op_arguments):
    #     self._check_argument_num('select', 1, op_arguments)
    #     return self._normalize(op_arguments[0])

    # def _verify_scene(self, op_arguments):
    #     self._check_argument_num('select', 1, op_arguments)
    #     return []

    def verify(self, program):
        if 'last_op' not in program:
            raise ParserError("The 'last_op' field is missing: " + str(program))

        if 'operator' not in program['last_op']:
            raise ParserError("The 'operator' field is missing: " + str(program['last_op'])) 

        if program['last_op']['operator'] in ['select', 'filter', 'relate']:
            raise ParserError("'{0}' is not a terminal operator: ".format(program['last_op']['operator']) + str(program['last_op']))
        
        try:
            method = getattr(self, '_verify_' + program['last_op']['operator'])
        except AttributeError:
            raise ParserError("Invalid operator: " + program['last_op']['operator'])
        
        method(program['last_op']['arguments'])

        if 'branches' not in program:
            raise ParserError("The 'branches' field is missing: " + str(program))

        branch_count = len(program['branches'])
        if program['last_op']['operator'] in ['and', 'or', 'two_same', 'two_different', 'compare'] and branch_count != 2:
            raise ParserError("'{0}' must have exactly two branches.".format(program['last_op']['operator']))
        elif program['last_op']['operator'] not in ['and', 'or', 'two_same', 'two_different', 'compare'] and branch_count != 1:
            raise ParserError("'{0}' must have exactly one branch.".format(program['last_op']['operator']))

        for b in program['branches']:
            for i, op in enumerate(b):
                if 'operator' not in op:
                    raise ParserError("The 'operator' field is missing: " + str(op)) 
                
                if i == 0 and op['operator'] != 'select':
                    raise ParserError("The first operator of a branch must be 'select': " + str(b))
                elif i > 0 and op['operator'] not in ['filter', 'relate']:
                    raise ParserError("All operators in a branch (except the first operator) must be either 'filter' or 'relate': " + op['operator'])
                try:
                    method = getattr(self, '_verify_' + op['operator'])
                except AttributeError:
                    raise ParserError("Invalid operator: " + op['operator'])
                
                if 'arguments' not in op:
                    raise ParserError("The 'arguments' field is missing: " + str(op)) 
                
                method(op['arguments'])

        return True            
