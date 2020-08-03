# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file
# in the project root for full license information.

import torch
import re
import numpy as np

type_map = {torch.float64: np.float64, torch.float32: np.float32, torch.float16: np.float16}

def is_cuda(device):
    if isinstance(device, int):
        return True

    return device.type == 'cuda'

def safe_exp(x):
    "Implements safe exp operation."
    return x.exp()
    # return torch.min(x, torch.tensor([30.0], device=device)).exp()

def safe_log(x):
    "Implements safe exp operation."

    return x.clamp(min=1e-6 if x.dtype == torch.float16 else 1e-20).log()
    # return x.log()
    # return torch.max(x, torch.tensor([1e-20], device=device)).log()

def log_and(op1, op2):
    return op1 + op2

def log_or(op1, op2):
    return safe_log(1.0 - (1.0 - safe_exp(op1)) * (1.0 - safe_exp(op2)))

def log_not(op):
    return safe_log(1.0 - safe_exp(op))

def log_and_tensor(op, dim=None):
    return op.sum() if dim is None else op.sum(dim)

def log_or_tensor(op, dim=None):
    t = log_not(op)
    t = t.sum() if dim is None else t.sum(dim)
    return log_not(t)

def log_parametric_not(op, alpha, beta):
    return safe_log(alpha + beta * (1 - 2*alpha) * safe_exp(op))

def almost_equal(a, b, eps=0):
    return (a - b).abs() <= eps

def flatten_list(a_list_list):
    a_list = [a if a is not None else [None] for a in a_list_list]
    batch_index = [i for i, sublist in enumerate(a_list) for item in sublist]
    a_list = [item for sublist in a_list for item in sublist]

    return a_list, batch_index

def unflatten_list(a_list, batch_index, flags):
    d = {i:[] for i in set(batch_index)}
    {d.setdefault(x, []).append(y) for x, y, z in zip(batch_index, a_list, flags) if z > 0}
    return list(d.values())

def find_max_ind(log_likelihood, predicate_question_map, likelihood_threshold=0):
    temp = log_likelihood.unsqueeze(1).exp() * predicate_question_map.to_dense()
    return (almost_equal(temp, (temp.max(0)[0]).unsqueeze(0)) * (temp > likelihood_threshold)).sum(1)

def detect_negations(a_list, device):
    is_negated = [re.match("not\((\w|\s)+\)", a.strip()) is not None for a in a_list]
    any_negated = any(is_negated)
    
    if any_negated:
        b_list = []
        for i, a in enumerate(a_list):
            if is_negated[i]:
                b_list.append(a.strip()[4:-1])
            else:
                b_list.append(a.strip())
    else:
        b_list = a_list
    
    # neg = torch.tensor(is_negated, dtype=torch.float32, device=device)
    # return any_negated, neg, b_list

    return any_negated, is_negated, b_list

def find_sparse_pair_indices(membership_ind1, membership_ind2, device, exclude_self_relations=True):
    size1 = membership_ind1.size()[0]
    size2 = membership_ind2.size()[0]
    flags = (membership_ind1.unsqueeze(1) == membership_ind2.unsqueeze(0))

    if exclude_self_relations and size1 == size2:
        if (membership_ind1 == membership_ind2).prod() == 1:
            flags = (flags.float() - torch.eye(size1, dtype=torch.float32, device=device)).bool()
    
    ind = torch.nonzero(flags)
    ind1 = ind[:, 0]
    ind2 = ind[:, 1]

    ind0 = membership_ind2.repeat(size1, 1)[flags]
    # _, ind0 = torch.unique(membership_ind2.repeat(size1, 1)[flags], sorted=False, return_inverse=True)

    return ind0, ind1, ind2

def to_cuda(data, device, non_blocking):
    
    def transfer_map(obj):
        if isinstance(obj, torch.Tensor):
            return obj.cuda(device, non_blocking)
        if isinstance(obj, tuple) and len(obj) > 0:
            return tuple(map(transfer_map, obj))
        if isinstance(obj, list) and len(obj) > 0:
            return list(map(transfer_map, obj))
        if isinstance(obj, dict) and len(obj) > 0:
            return dict(map(transfer_map, obj.items()))
        return obj

    try:
        res = transfer_map(data)
    finally:
        transfer_map = None
    return res

def to(data, dtype):
    
    def transfer_map(obj):
        if isinstance(obj, np.ndarray):
            if obj.dtype in type_map.values():
                return obj.astype(type_map[dtype])
            else:
                return obj

        if isinstance(obj, torch.Tensor):
            if obj.dtype in type_map.keys():
                return obj.to(dtype)
            else:
                return obj

        if isinstance(obj, tuple) and len(obj) > 0:
            return tuple(map(transfer_map, obj))
        if isinstance(obj, list) and len(obj) > 0:
            return list(map(transfer_map, obj))
        if isinstance(obj, dict) and len(obj) > 0:
            return dict(map(transfer_map, obj.items()))
        return obj

    try:
        res = transfer_map(data)
    finally:
        transfer_map = None
    return res

def reverse_dependencies(pred):
    length = len(pred)
    pred_length = np.array([len(p) for p in pred], dtype=np.int32)
    flat_pred = np.concatenate([np.sort(sublist) for sublist in pred])
    flat_ind = [[i]*pred_length[i] for i in range(length)]
    flat_ind = np.array([item for sublist in flat_ind for item in sublist], dtype=np.int32)
    reversed_pred = [np.sort(flat_ind[np.where(flat_pred == i)]).tolist() for i in range(length)]

    return reversed_pred

def mm(a, b):
    if a.dtype == torch.float16 or b.dtype == torch.float16:
        return torch.mm(a.to(torch.float32), b.to(torch.float32)).to(torch.float16)

    return torch.mm(a, b)

class ClusteredLogSoftmax(torch.nn.Module):

    def __init__(self, cluster_index):
        super(ClusteredLogSoftmax, self).__init__()
        self._log_sigmoid = torch.nn.LogSigmoid()
        self.register_buffer('_zero_index', (cluster_index == 0).nonzero())
        self.register_buffer('_cluster_index', cluster_index)
        self._cluster_map = None
        self._cluster_map_transposed = None

    def _build_map(self):
        size = self._cluster_index.size(0)
        cluster_num = self._cluster_index.max().item() + 1
        
        device = self._cluster_index.device
        all_ones = torch.ones(size, device=device)
        y_ind = torch.arange(size, dtype=torch.int64, device=device)
        ind = torch.stack([self._cluster_index, y_ind])

        if isinstance(device, int) or device.type == 'cuda':
            self._cluster_map = torch.cuda.sparse.FloatTensor(ind, all_ones, 
                        torch.Size([cluster_num, size])).to(device)
        else:
            self._cluster_map = torch.sparse.FloatTensor(ind, all_ones, 
                        torch.Size([cluster_num, size]))

        self._cluster_map_transposed = self._cluster_map.transpose(0, 1)

    def forward(self, logits):
        if self._cluster_map is None:
            self._build_map()

        denom = mm(self._cluster_map_transposed, safe_log(mm(self._cluster_map, logits.transpose(0, 1).exp()))).transpose(0, 1)
        res = logits - denom
        res[:, self._zero_index] = self._log_sigmoid(logits[:, self._zero_index])

        return res

def weight_activation(w):
    return 1.0 - torch.exp(-w.pow(2) / 2.0)
