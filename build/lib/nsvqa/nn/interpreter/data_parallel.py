# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file
# in the project root for full license information.

"The data parallel mechanism for Program Batches."

import math
import torch
import torch.nn as nn
import numpy as np

from torch.nn.parallel.scatter_gather import gather
from nsvqa.nn.interpreter.batch_base_types import QuestionType

def gather_results(outputs, target_device, is_cuda):
    if outputs[0]['type'] == QuestionType.SCENE_GRAPH:
        log_probability = []
        for i in range(len(outputs[0]['log_probability'])):
            log_probability_list = [o['log_probability'][i] for o in outputs]
            if is_cuda:
                log_probability.append(gather(log_probability_list, target_device, dim=0))
            else:
                log_probability.append(torch.stack(log_probability_list))

        answer = []
        for i in range(len(outputs[0]['answer'])):
            answer_list = [o['answer'][i] for o in outputs]
            answer.append(np.concatenate(answer_list, axis=0))

        answer_log_probability = []         
    else:
        log_probability_list = [o['log_probability'] for o in outputs]
        if is_cuda:
            log_probability = gather(log_probability_list, target_device, dim=0).flatten()
        else:
            log_probability = torch.stack(log_probability_list).flatten()
    
        answer = list(sum([o['answer'] for o in outputs], []))
        answer_log_probability = list(sum([o['answer_log_probability'] for o in outputs], []))

    if outputs[0]['type'] == QuestionType.QUERY:
        options = list(sum([o['options'] for o in outputs], []))
    else:
        options = outputs[0]['options']

    cumulative_loss = sum([o['cumulative_loss'] for o in outputs])
    variable_sets_num = sum([o['variable_sets_num'] for o in outputs])

    return {'answer': answer, 'log_probability': log_probability, 'options': options, 'variable_set': None, 'type': outputs[0]['type'],\
        'cumulative_loss': cumulative_loss, 'variable_sets_num': variable_sets_num, 'answer_log_probability': answer_log_probability}

######################################################################################################################################

class ProgramDataParallel(nn.DataParallel):

    def __init__(self, module, device_ids=None, output_device=None, dim=0):
        super(ProgramDataParallel, self).__init__(module, device_ids, output_device, dim)

    def scatter(self, inputs, kwargs, device_ids):
        device_num = len(device_ids)
        input_num = len(inputs[0])
        out_inputs = []
        out_kwargs = []
        input_per_device = math.ceil(input_num / device_num)
        
        for i, device in enumerate(device_ids):
            end = min(input_num, (i+1) * input_per_device) if i < device_num - 1 else input_num
            program_batches = [pb.to_cuda(device, True) for pb in inputs[0][(i * input_per_device):end]]

            for k, pb in enumerate(program_batches):
                for j in range(len(pb._op_batch_list)):
                    pb._op_batch_list[j]._op_id = ':'.join([str(i), str(k), pb._op_batch_list[j]._op_id])

            out_inputs.append((program_batches, inputs[1]))
            out_kwargs.append(kwargs.copy())

            if end >= input_num:
                break

        return tuple(out_inputs), tuple(out_kwargs)

    def gather(self, outputs, output_device):
        return gather_results(outputs, output_device, True)
