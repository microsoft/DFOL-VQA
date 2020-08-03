# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file
# in the project root for full license information.

import torch
import re
from operator import itemgetter
from nsvqa.nn.interpreter import util
from nsvqa.nn.interpreter.batch_base_types import TokenType

class OracleBase(torch.nn.Module):
    
    def __init__(self, ontology, feature_dim=1):
        super(OracleBase, self).__init__()
        self._feature_dim = feature_dim
        self._ontology = ontology

    def forward(self, token_type, token_list, token_image_map, world, default_log_likelihood=-30, normalized_probability=True):
        if not isinstance(token_list, list):
            token_list = [token_list]
        t_list = [a.strip() for a in token_list]
        
        if token_type == TokenType.ATTRIBUTE:
            res = self._compute_attribute_log_likelihood(world._device, t_list, world._attribute_features, world._meta_data, world._object_num, token_image_map, \
                world._object_image_map, default_log_likelihood=default_log_likelihood, normalized_probability=normalized_probability)

            if isinstance(res, torch.Tensor):
                res = res.view(len(token_list), world._object_num, self._feature_dim)
        
        elif token_type == TokenType.RELATION:
            res = self._compute_relation_log_likelihood(world._device, t_list, world._relation_features, world._meta_data, world._object_num, token_image_map, \
                world._object_image_map, default_log_likelihood=default_log_likelihood, normalized_probability=normalized_probability)
            
            if isinstance(res, torch.Tensor):
                res = res.view(len(token_list), world._object_num, world._object_num, self._feature_dim)

        return res

    def _compute_attribute_log_likelihood(self, device, attribute_list, object_features, meta_data, object_num, attribute_image_map, object_image_map, default_log_likelihood=-30, normalized_probability=True):
        pass

    def _compute_relation_log_likelihood(self, device, relation_list, pair_object_features, meta_data, object_num, relation_image_map, object_image_map, default_log_likelihood=-30, normalized_probability=True):
        pass

    def get_embedding(self, tokens, meta_data, device):
        if meta_data is None:
            embedding = torch.from_numpy(self._ontology.get_embedding(tokens)).float().to(device)
        else:
            try:
                ind = itemgetter(*tokens)(meta_data['index'])
                embedding = meta_data['embedding'][ind, :]
            except KeyError as e:
                embedding = torch.from_numpy(self._ontology.get_embeddings(tokens)).float().to(device)

        return embedding

######################################################################################################################################

class RandomOracle(OracleBase):
    
    def __init__(self, ontology, device):
        super(RandomOracle, self).__init__(ontology)
        self._device = device

    def _compute_attribute_log_likelihood(self, device, attribute_list, object_features, meta_data, object_num, attribute_image_map, object_image_map, default_log_likelihood=-30):
        res = torch.rand(len(attribute_list) * self._object_num, device=self._device)
        # print("\nAttribute likelihood:")
        # print(res.view(len(attribute_list), -1))
        return util.safe_log(res)

    def _compute_relation_log_likelihood(self, device, relation_list, pair_object_features, meta_data, object_num, relation_image_map, object_image_map, default_log_likelihood=-30):
        res = torch.rand(len(relation_list) * (self._object_num**2), device=self._device)
        # print("\nRelation likelihood:")
        # print(res.view(len(relation_list), self._object_num, self._object_num))
        return util.safe_log(res)

######################################################################################################################################

class StaticOracle(OracleBase):
    
    def __init__(self, ontology, feature_dim=1):
        super(StaticOracle, self).__init__(ontology, feature_dim=feature_dim)

    def _extract_entries(self, a_list, features):
        i = [features['index'][c] if c in features['index'] else 0 for c in a_list]
        ind = torch.tensor(i, dtype=torch.int64, device=features['log_likelihood'].device)
        return features['log_likelihood'][ind]

    def _compute_attribute_log_likelihood(self, device, attribute_list, object_features, meta_data, object_num, attribute_image_map, object_image_map, default_log_likelihood=-30):
        return self._extract_entries(attribute_list, object_features)

    def _compute_relation_log_likelihood(self, device, relation_list, pair_object_features, meta_data, object_num, relation_image_map, object_image_map, default_log_likelihood=-30):
        return self._extract_entries(relation_list, pair_object_features)
