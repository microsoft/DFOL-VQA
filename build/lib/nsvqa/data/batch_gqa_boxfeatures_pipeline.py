# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file
# in the project root for full license information.

import torch
import numpy as np
import json
# import h5pickle as h5py
import h5py

from os.path import join
from nsvqa.data.data_pipeline import ProgramCollaterBase
from nsvqa.nn.interpreter import util

class BatchGQABoxFeaturesCollator(ProgramCollaterBase):
    
    def __init__(self, object_h5_path, file_prefix, chunk_num, object_info_json_path, ontology, split_num):
        super(BatchGQABoxFeaturesCollator, self).__init__('select', 'relate', 'filter', split_num)

        self._object_h5_path = object_h5_path
        self._file_prefix = file_prefix
        self._chunk_num = chunk_num

        self._file_handles = None

        with open(object_info_json_path, 'r') as json_file:
            self._object_info = json.load(json_file)

        with h5py.File(join(self._object_h5_path, self._file_prefix + '_0.h5'), 'r') as file:
            self._chunck_size, self._max_object_per_image, self._feature_dim = file['features'].shape

            if 'relationsNum' in list(self._object_info.values())[0]: # Pre-featurized relations
                _, self._max_relation_per_image, self._relation_feature_dim = file['relation_features'].shape
        
        self._ontology = ontology

    def collate_object_features(self, questions):
        if self._file_handles is None:
            self._file_handles = [h5py.File(join(self._object_h5_path, self._file_prefix + '_' + str(i) + '.h5'), 'r') for i in range(self._chunk_num)]

        image_ids = [q['image_id'] for q in questions]
        object_nums = [self._object_info[im_id]['objectsNum'] for im_id in image_ids]
        widths = [self._object_info[im_id]['width'] for im_id in image_ids]
        heights = [self._object_info[im_id]['height'] for im_id in image_ids]
        image_idx = [self._object_info[im_id]['idx'] for im_id in image_ids]
        image_chk = [self._object_info[im_id]['file'] for im_id in image_ids]

        images = np.zeros((len(image_idx), self._max_object_per_image, self._feature_dim))
        bboxes = np.zeros((len(image_idx), self._max_object_per_image, 4))
        image_sizes = np.zeros((len(image_idx), self._max_object_per_image, 2))

        for j, (chunck_id, offset) in enumerate(zip(image_chk, image_idx)):
            images[j, :, :] = self._file_handles[chunck_id]['features'][offset, :, :]
            bboxes[j, :, :] = self._file_handles[chunck_id]['bboxes'][offset, :, :]
            image_sizes[j, :, :] = np.array([[widths[j], heights[j]]], dtype=np.float32)

        images = images.reshape((-1, self._feature_dim))
        bboxes = bboxes.reshape((-1, 4))
        image_sizes = image_sizes.reshape((-1, 2))
        bboxes[:, 2] -= bboxes[:, 0]
        bboxes[:, 3] -= bboxes[:, 1]

        batch_ind = [(i * np.ones(n, dtype=np.int64)).tolist() for i, n in enumerate(object_nums)]
        batch_ind = [i for i, sublist in enumerate(batch_ind) for item in sublist]
        batch_ind = torch.tensor(batch_ind, dtype=torch.int64)
        
        object_idx = [(i * self._max_object_per_image + np.arange(n)).tolist() for i, n in enumerate(object_nums)]
        object_idx = [item for i, sublist in enumerate(object_idx) for item in sublist]
        object_idx = np.array(object_idx, dtype=np.int64)

        object_features = torch.from_numpy(np.concatenate((images[object_idx, :], image_sizes[object_idx, :], bboxes[object_idx, :]), 1)).float()

        return object_features, batch_ind
    
    def collate_meta_data(self, questions):
        all_tokens = []
        original_questions = []
        image_ids = []
        question_ids = []

        # Collect all tokens in the questions
        for q in questions:
            all_tokens += q['tokens']
            original_questions.append(q['question'])
            image_ids.append(q['image_id'])
            question_ids.append(q['question_id'])         

        all_tokens = list(set(all_tokens))
        object_nums = [self._object_info[im_id]['objectsNum'] for im_id in image_ids]
        index = {token: i for i, token in enumerate(all_tokens)}
        embedding = torch.from_numpy(self._ontology.get_embeddings(all_tokens)).float()
        result = {'index': index, 'embedding': embedding, 'questions': original_questions, 'image_ids': image_ids, 'question_ids': question_ids}

        if any(['object_pairs' in q for q in questions]):
            subject_id = [q['object_pairs']['subject_id'] if 'object_pairs' in q and 'subject_id' in q['object_pairs'] else [] for q in questions]
            object_id = [q['object_pairs']['object_id'] if 'object_pairs' in q and 'object_id' in q['object_pairs'] else [] for q in questions]
            result['object_pairs'] = {'subject_id': subject_id, 'object_id': object_id}

        if any(['weights' in q for q in questions]):
            weights = [q['weights'] for q in questions]
            result['weights'] = list(sum(weights, []))

        if any(['attribute_dict' in q for q in questions]):
            # 'attribute_dict': {object_index: [(attribute, weight), ...], ...} 
            total_obj_num = sum(object_nums)
            answer = np.zeros((total_obj_num, len(self._ontology._attribute_index)))
            weight = np.zeros((total_obj_num, len(self._ontology._attribute_index)))

            offset = 0
            for i, q in enumerate(questions):
                ind1 = []
                ind2 = []
                w = []
                for obj_index, att_list in q['attribute_dict'].items():
                    w_ind = set(self._ontology._noun_subindex)
                    for a in att_list:
                        if a[0] in self._ontology._vocabulary['arg_to_idx'] and a[0] in self._ontology._attributes:
                            ind1.append(int(obj_index) + offset)
                            ind2.append(self._ontology._attribute_reveresed_index[self._ontology._vocabulary['arg_to_idx'][a[0]] - 1])
                            w.append(a[1])
                            w_ind = w_ind | set(self._ontology.get_family_subindex(a[0]))
        
                    weight[int(obj_index) + offset, list(w_ind)] = 1.0

                answer[ind1, ind2] = 1.0
                weight[ind1, ind2] = w
                offset += object_nums[i]
            
            result['attribute_answer'] = answer
            result['attribute_weight'] = weight

        if any(['relation_list' in q for q in questions]):
            # 'relation_list': [(relation, weight), ...] 
            relation_nums = [len(q['relation_list']) for q in questions]
            total_rel_num = sum(relation_nums)
            answer = np.zeros((total_rel_num, len(self._ontology._relation_index)))
            weight = np.ones((total_rel_num, len(self._ontology._relation_index)))

            offset = 0
            for i, q in enumerate(questions):
                ind2 = []
                ind1 = []
                w = []
                for j, rel in enumerate(q['relation_list']):
                    if rel[0] in self._ontology._vocabulary['arg_to_idx'] and rel[0] in self._ontology._relations:
                        ind1.append(j + offset)
                        ind2.append(self._ontology._relation_reveresed_index[self._ontology._vocabulary['arg_to_idx'][rel[0]] - 1])
                        w.append(rel[1])

                answer[ind1, ind2] = 1.0
                weight[ind1, ind2] = w
                offset += relation_nums[i]
            
            result['relation_answer'] = answer
            result['relation_weight'] = weight

        if 'relationsNum' in list(self._object_info.values())[0]: # Pre-featurized relations
            relation_nums = [self._object_info[im_id]['relationsNum'] for im_id in image_ids]
            image_idx = [self._object_info[im_id]['idx'] for im_id in image_ids]
            image_chk = [self._object_info[im_id]['file'] for im_id in image_ids]

            relations = np.zeros((len(image_idx), self._max_relation_per_image, self._relation_feature_dim))
            pair_indices = np.zeros((len(image_idx), self._max_relation_per_image, 2))

            for j, (chunck_id, offset) in enumerate(zip(image_chk, image_idx)):
                relations[j, :, :] = self._file_handles[chunck_id]['relation_features'][offset, :, :]
                pair_indices[j, :, :] = self._file_handles[chunck_id]['relation_indices'][offset, :, :] + (object_nums[j - 1] if j > 0 else 0)

            relations = relations.reshape((-1, self._relation_feature_dim))
            pair_indices = pair_indices.reshape((-1, 2))

            rel_img_map = [(i * np.ones(n, dtype=np.int64)).tolist() for i, n in enumerate(relation_nums)]
            rel_img_map = [i for i, sublist in enumerate(rel_img_map) for item in sublist]
            rel_img_map = torch.tensor(rel_img_map, dtype=torch.int64)
            
            relation_idx = [(i * self._max_relation_per_image + np.arange(n)).tolist() for i, n in enumerate(relation_nums)]
            relation_idx = [item for i, sublist in enumerate(relation_idx) for item in sublist]
            relation_idx = np.array(relation_idx, dtype=np.int64)
            
            relations = torch.from_numpy(relations[relation_idx, :]).float()
            pair_indices = torch.from_numpy(pair_indices[relation_idx, :]).long()

            result['relation_features'] = relations
            result['relation_indices'] = pair_indices
            result['relation_image_map'] = rel_img_map
        
        return result

######################################################################################################################################

class BatchGQABoxFeaturizer(torch.nn.Module):

    def __init__(self, featurizer_network=None):
        super(BatchGQABoxFeaturizer, self).__init__()
        self._featurizer_network = featurizer_network

    def featurize_scene(self, device, objects_list, batch_index, meta_data):
        object_num = objects_list.size()[0]
        feature_size = objects_list.size()[1] - 6

        if self._featurizer_network is not None:
            object_features = self._featurizer_network(objects_list[:, :-6])
        else:
            object_features = objects_list[:, :-6]

        positional_features = objects_list[:, -4:] / \
            torch.cat([objects_list[:, -6].unsqueeze(1), objects_list[:, -5].unsqueeze(1), objects_list[:, -6].unsqueeze(1), objects_list[:, -5].unsqueeze(1)], dim=1).clamp(1)

        object_features = torch.cat([object_features, positional_features], dim=1)

        if 'relation_features' in meta_data: # Pre-featurized relations
            relation_features = meta_data['relation_features'].to(device=device)
            meta_data['relation_features'] = None
            
            ind0 = meta_data['relation_image_map'].to(device=device)
            meta_data['relation_image_map'] = None
            
            ind1 = meta_data['relation_indices'][:, 0].to(device=device)
            ind2 = meta_data['relation_indices'][:, 1].to(device=device)
            meta_data['relation_indices'] = None

        else:
            if 'object_pairs' in meta_data: # For direct supervision
                batch_size = batch_index.max().cpu().numpy().tolist() + 1
                all_ones = torch.ones(object_num, device=device)
                y_ind = torch.arange(object_num, dtype=torch.int64, device=device)
                ind = torch.stack([batch_index, y_ind])
                
                if all_ones.device.type == 'cuda':
                    batch_object_map = torch.cuda.sparse.FloatTensor(ind, all_ones, 
                                    torch.Size([batch_size, object_num]), device=device)
                else:
                    batch_object_map = torch.sparse.FloatTensor(ind, all_ones, 
                                    torch.Size([batch_size, object_num]), device=device)

                object_nums = batch_object_map.to_dense().sum(1)
                object_nums_cumsum = object_nums.cumsum(0).cpu().numpy().tolist()
                object_nums_cumsum = [0] + object_nums_cumsum[:-1]

                ind1 = [item+i for i, sublist in zip(object_nums_cumsum, meta_data['object_pairs']['subject_id']) for item in sublist]
                ind1 = torch.tensor(ind1, dtype=torch.int64, device=device)

                ind2 = [item+i for i, sublist in zip(object_nums_cumsum, meta_data['object_pairs']['object_id']) for item in sublist]
                ind2 = torch.tensor(ind2, dtype=torch.int64, device=device)

                _, ind0 = util.flatten_list(meta_data['object_pairs']['subject_id'])
                ind0 = torch.tensor(ind0, dtype=torch.int64, device=device)
            else:
                # For relational feature vectors, only consider pairs of objects belonging to the same image minus self-relations.
                ind0, ind1, ind2 = util.find_sparse_pair_indices(batch_index, batch_index, device, exclude_self_relations=True)
            
            relation_features = None
            
            if ind1.size()[0] > 0 and ind2.size()[0] > 0:
                relation_features = torch.cat([object_features[ind1, :], object_features[ind2, :]], dim=1)

                # Add relative postition relational features
                x1 = positional_features[ind1, 0]
                y1 = positional_features[ind1, 1]
                w1 = positional_features[ind1, 2]
                h1 = positional_features[ind1, 3]

                x2 = positional_features[ind2, 0]
                y2 = positional_features[ind2, 1]
                w2 = positional_features[ind2, 2]
                h2 = positional_features[ind2, 3]

                # with torch.no_grad():
                distance = torch.sqrt((x1 + w1 / 2.0 - x2 - w2 / 2.0) ** 2 +\
                            (y1 + h1 / 2.0 - y2 - h2 / 2.0) ** 2)

                # angle = torch.asin((y1 + h1 / 2.0 - y2 - h2 / 2.0) / torch.max(distance, torch.tensor([1e-10], device=device))).abs().unsqueeze(1)
                angle = torch.asin((y1 + h1 / 2.0 - y2 - h2 / 2.0) / distance.clamp(min=1e-10)).unsqueeze(1)
                h_side = (x2 - x1).sign().unsqueeze(1)
                v_side = (y2 - y1).sign().unsqueeze(1)
                distance = distance.unsqueeze(1)
                relation_features = torch.cat([relation_features, distance, angle, h_side, v_side], dim=1)

        return {'attribute_features': object_features, 'relation_features': {'features': relation_features, 'index':[ind0, ind1, ind2]}, 'object_num': object_num}
