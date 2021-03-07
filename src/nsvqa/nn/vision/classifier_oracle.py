# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file
# in the project root for full license information.

import torch
import numpy as np
from operator import itemgetter
from nsvqa.nn.vision.base_oracle import OracleBase
from nsvqa.nn.interpreter import util

class ClassifierOracle(OracleBase):

    def __init__(self, ontology, attribute_network, relation_network, embedding_network, normalize=False, cached=False):
        super(ClassifierOracle, self).__init__(ontology, feature_dim=1)

        self._attribute_network = attribute_network
        self._relation_network = relation_network
        self._embedding_network = embedding_network
        self._normalize = normalize
        self._cached = cached

    def _build_map(self, attribute_image_map):
        _, cluster_index = torch.unique_consecutive(attribute_image_map, return_inverse=True)
        size = cluster_index.size(0)
        cluster_num = cluster_index.max().item() + 1

        if size == cluster_num:
            return None
        
        device = cluster_index.device
        all_ones = torch.ones(size, device=device)
        y_ind = torch.arange(size, dtype=torch.int64, device=device)
        ind = torch.stack([cluster_index, y_ind])

        if isinstance(device, int) or device.type == 'cuda':
            cluster_map = torch.cuda.sparse.FloatTensor(ind, all_ones, 
                        torch.Size([cluster_num, size])).to(device)
        else:
            cluster_map = torch.sparse.FloatTensor(ind, all_ones, 
                        torch.Size([cluster_num, size]))

        return cluster_map

    def _compute_attribute_log_likelihood(self, device, attribute_list, object_features, meta_data, object_num, attribute_image_map, \
        object_image_map, default_log_likelihood=-30, normalized_probability=True):
        object_num = object_features.size()[0]
        attribute_num = len(attribute_list)
        
        temp = itemgetter(*attribute_list)(self._ontology._vocabulary['arg_to_idx'])
        # if self._cached:
        #     temp = np.array(temp, dtype=np.int64) - 1
        #     temp = np.array(itemgetter(*temp.tolist())(self._ontology._attribute_reveresed_index), dtype=np.int64)
        #     ind = torch.tensor(temp, dtype=torch.int64, device=device)
        # else:
        ind = torch.tensor(temp, dtype=torch.int64, device=device)
        ind = ind - 1
        
        if ind.dim() == 0:
            ind = ind.unsqueeze(0)

        _, ind1, ind2 = util.find_sparse_pair_indices(attribute_image_map, object_image_map, device, exclude_self_relations=False)
        if self._cached:
            output_features = object_features[ind2, ind[ind1]]
        else:
            output_features = self._embedding_network(self._attribute_network(object_features))[ind2, ind[ind1]]
        
        # Reshape into the dense version
        if self._normalize and normalized_probability:
            result = default_log_likelihood * torch.ones(attribute_num, object_num, dtype=object_features.dtype, device=device)
            result[ind1, ind2] = output_features

            cluster_map = self._build_map(attribute_image_map)
            if cluster_map is not None:
                denom = util.mm(cluster_map.transpose(0, 1), util.safe_log(util.mm(cluster_map, result.exp())))
                result = result - denom

            result = result.unsqueeze(2)
        else:
            result = default_log_likelihood * torch.ones(attribute_num, object_num, 1, dtype=object_features.dtype, device=device)
            result[ind1, ind2, 0] = output_features

        return result

    def _compute_relation_log_likelihood(self, device, relation_list, pair_object_features, meta_data, object_num, relation_image_map, \
        object_image_map, default_log_likelihood=-30, normalized_probability=True):
        pair_num = pair_object_features['features'].size()[0]
        relation_num = len(relation_list)

        temp = itemgetter(*relation_list)(self._ontology._vocabulary['arg_to_idx'])
        if self._cached:
            if not isinstance(temp, (list, tuple)):
                temp = [temp]

            temp = (np.array(temp, dtype=np.int64) - 1).tolist()
            temp = itemgetter(*temp)(self._ontology._relation_reveresed_index)
            ind = torch.tensor(temp, dtype=torch.int64, device=device)
        else:
            ind = torch.tensor(temp, dtype=torch.int64, device=device)
            ind = ind - 1

        if ind.dim() == 0:
            ind = ind.unsqueeze(0)

        if 'relation_pairobject_map' in meta_data: # Only for direct-supervision, object-level training/testing
            relation_seq = list(range(relation_num))
            if self._cached:
                res = pair_object_features['features'][meta_data['relation_pairobject_map'], ind]
            else:
                res = self._embedding_network(self._relation_network(pair_object_features['features']))[meta_data['relation_pairobject_map'], ind]

            result = default_log_likelihood * torch.ones(relation_num, object_num, object_num, 1, dtype=pair_object_features['features'].dtype, device=device)
            result[relation_seq, pair_object_features['index'][1], pair_object_features['index'][2], 0] = res
        else:
            _, ind1, ind2 = util.find_sparse_pair_indices(relation_image_map, pair_object_features['index'][0], device, exclude_self_relations=False)
            if self._cached:
                result = pair_object_features['features'][ind2, ind[ind1]]
            else:
                result = self._embedding_network(self._relation_network(pair_object_features['features']))[ind2, ind[ind1]]

            if self._normalize and normalized_probability:
                temp = default_log_likelihood * torch.ones(relation_num, pair_num, dtype=pair_object_features['features'].dtype, device=device)
                temp[ind1, ind2] = result

                cluster_map = self._build_map(relation_image_map)
                if cluster_map is not None:
                    denom = util.mm(cluster_map.transpose(0, 1), util.safe_log(util.mm(cluster_map, temp.exp())))
                    temp = temp - denom

                temp = temp.unsqueeze(2)
            else:
                temp = default_log_likelihood * torch.ones(relation_num, pair_num, 1, dtype=pair_object_features['features'].dtype, device=device)
                temp[ind1, ind2, 0] = result

            result = default_log_likelihood * torch.ones(relation_num, object_num, object_num, 1, dtype=pair_object_features['features'].dtype, device=device)
            result[:, pair_object_features['index'][1], pair_object_features['index'][2], :] = temp

        return result

    def compute_all_log_likelihood(self, object_features, pair_object_features):
        attr_output = self._embedding_network(self._attribute_network(object_features))[:, self._ontology._attribute_index]
        rel_output = self._embedding_network(self._relation_network(pair_object_features['features']))[:, self._ontology._relation_index]

        return attr_output, rel_output

    def compute_all_log_likelihood_2(self, object_features, pair_object_features):
        if self._embedding_network is None or self._attribute_network is None:
            attr_output = object_features
        else:
            attr_output = self._embedding_network(self._attribute_network(object_features))

        if self._embedding_network is None or self._relation_network is None:
            rel_output = pair_object_features
        else:
            rel_output = self._embedding_network(self._relation_network(pair_object_features))[:, self._ontology._relation_index]

        return attr_output, rel_output
