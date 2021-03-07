# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file
# in the project root for full license information.

import os, cv2
import time
import math, json
import multiprocessing

import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
from itertools import compress, repeat, chain

from nsvqa.data.data_pipeline import GQADataManager
from nsvqa.nn.interpreter.batch_base_types import QuestionType
from nsvqa.nn.interpreter import util
from nsvqa.nn.interpreter.data_parallel import ProgramDataParallel

class VQATrainer(object):
    "Base class of the Factor Graph trainer pipeline (abstract)."

    # pylint: disable=unused-argument
    def __init__(self, use_cuda, config, logger, ontology, hardset_path=None):
        self._config = config
        self._logger = logger
        self._ontology = ontology
        self._use_cuda = use_cuda and torch.cuda.is_available()
        self._hardset_path = hardset_path
        
        if self._hardset_path is not None:
            self._hardset_prefix = '_'.join([os.path.basename(self._config['test_path']), self._config['model_name'], self._config['version']])
            self._hardset_path = os.path.join(self._hardset_path, self._hardset_prefix)
            if not os.path.exists(self._hardset_path):
                os.makedirs(self._hardset_path)

            self._hard_subdir = os.path.join(self._hardset_path, 'hard')
            if not os.path.exists(self._hard_subdir):
                os.makedirs(self._hard_subdir)

            self._easy_subdir = os.path.join(self._hardset_path, 'easy')
            if not os.path.exists(self._easy_subdir):
                os.makedirs(self._easy_subdir)
        
        # Set the device
        if self._use_cuda:
            self._device = torch.device("cuda")
            if config['verbose']:
                logger.info('Using GPU...')
        else:
            self._device = torch.device("cpu")
            if config['verbose']:
                logger.info('Using CPU...')

        self._num_cores = config['cpu_cores_num'] if 'cpu_cores_num' in config else multiprocessing.cpu_count()

        if config['verbose']:
            self._logger.info("The number of CPU workers is %s." % self._num_cores)

        torch.set_num_threads(self._num_cores)
        self._model = None
        self._op_index = OrderedDict({
            'query_attr': 1,
            'choose_attr': 2,
            'verify_attrs': 3,
            'choose_rel': 4,
            'verify_rel': 5,
            'exist': 6,
            'and': 7,
            'or': 8,
            'all_same': 9,
            'all_different': 10,
            'two_same': 11,
            'two_different': 12,
            'compare': 13,
            'object_attr': 14,
            'object_rel': 15,
            'scene': 16
        })

        self._error_dim = len(self._op_index) + 1 #config['error_dim'] if 'error_dim' in config else 1

    def _prepare_output_metric_dict(self, error):
        return dict(zip(['over_all'] + list(self._op_index.keys()), error.flatten().tolist()))

    @property
    def model(self):
        return self._model.module if isinstance(self._model, ProgramDataParallel) else self._model

    @model.setter
    def model(self, value):
        self._model = self._set_device(value)

    def _run_model(self, program_batch_list, is_training):
        result = self._model(program_batch_list, is_training, return_trace=False, modulator_switch=is_training or program_batch_list[0]._question_type != QuestionType.QUERY)

        # if self._config['activate_attention_transfer'] and (not is_training):
        #     temp = self._model(program_batch_list, is_training, return_trace=False, modulator_switch=False)

        #     ans = []
        #     for i, (a, b) in enumerate(zip(result['answer'], temp['answer'])):
        #         ans.append(a if result['answer_log_probability'][i][0] >= temp['answer_log_probability'][i][0] else b)

        #     result['answer'] = ans
            
        # if self._config['activate_attention_transfer'] and (not is_training) and result['type'] == QuestionType.QUERY:# program_batch_list[0]._op_batch_list[-1]._op_name == 'query_attr':
        #     temp = self._model(program_batch_list, is_training, return_trace=False, modulator_switch=False)

        #     ans = []
        #     for a, b in zip(result['answer'], temp['answer']):
        #         ans.append((list(set(a) - set(b)) if not set(a).issubset(set(b)) else a) if len(a) <= len(b) else b)

        #     result['answer'] = ans
        
        return result

    def _load(self, import_path_base):
        "Loads the model(s) from file."
        
        self.model.load(import_path_base)

    def _save(self, export_path_base):
        "Saves the model(s) to file."

        self.model.save(export_path_base)

    def _reset_global_step(self):
        "Resets the global step counter."
        
        self.model._global_step.data = torch.tensor([0], dtype=torch.float, device=self._device)

    def _set_device(self, m):
        "Sets the CPU/GPU device."

        if self._use_cuda:
            device_count = torch.cuda.device_count() if self._config['gpu_num'] is None else min(torch.cuda.device_count(), self._config['gpu_num'])            
            if device_count > 1:
                return ProgramDataParallel(m).cuda(self._device)
            else:
                return m.cuda(self._device)
        return m.cpu()
    
    def _to_cuda(self, data):
        if self._use_cuda and not isinstance(self.model, ProgramDataParallel):
            data = [d.to_cuda(self._device, True) for d in data]
        
        return data

    def get_parameter_list(self):
        "Returns a dictionary of model's parameters."
        return [{'params': filter(lambda p: p.requires_grad, self.model.parameters())}]

    # def _compute_loss(self, program_batch_list, prediction):
    #     if prediction['type'] == QuestionType.STATEMENT:
    #         return -prediction['log_probability'].sum()

    #     if prediction['type'] == QuestionType.BINARY:
    #         target = []
    #         for program_batch in program_batch_list:
    #             target.append([a in ['yes', 'yeah', 'yep', 'yup', 'aye', 'yea'] for a in program_batch._answers])

    #         target = list(sum(target, []))
    #         norm = torch.ones(len(target), device=self._device)
        
    #     elif prediction['type'] == QuestionType.QUERY:
    #         all_answers = [program_batch._answers for program_batch in program_batch_list]
    #         all_answers = list(sum(all_answers, []))
            
    #         target = [[a == o for o in op] for a, op in zip(all_answers, prediction['options'])]
    #         target = list(sum(target, []))
    #         norm = [[len(op) if len(op) > 0 else 1 for _ in op] for op in prediction['options']]
    #         norm = list(sum(norm, []))
    #         norm = torch.tensor(norm, dtype=torch.float32, device=self._device)

    #     target = torch.tensor(target, dtype=torch.float32, device=self._device)
    #     loss = nn.functional.binary_cross_entropy(prediction['log_probability'].exp(), target, weight=1.0 / norm, reduction='sum')
    #     return loss

    def _compute_loss(self, program_batch_list, prediction):
        if prediction['type'] == QuestionType.STATEMENT:
            return -prediction['log_probability'].sum()

        if prediction['type'] == QuestionType.BINARY:
            target = []
            for program_batch in program_batch_list:
                target.append([a in ['yes', 'yeah', 'yep', 'yup', 'aye', 'yea'] for a in program_batch._answers])

            target = list(sum(target, []))
            norm = torch.ones(len(target), device=self._device)

            target = torch.tensor(target, dtype=torch.float32, device=self._device)
            loss = nn.functional.binary_cross_entropy(prediction['log_probability'].exp(), target, weight=1.0 / norm, reduction='sum')
        
        elif prediction['type'] == QuestionType.OBJECT_STATEMENT:
            target = []
            for program_batch in program_batch_list:
                target.append([a in ['yes', 'yeah', 'yep', 'yup', 'aye', 'yea'] for sublist in program_batch._answers for a in sublist])

            target = list(sum(target, []))
            weights = torch.tensor(list(sum([program_batch._meta_data['weights'] for program_batch in program_batch_list], [])), device=self._device)

            target = torch.tensor(target, dtype=torch.float32, device=self._device)
            loss = nn.functional.binary_cross_entropy(prediction['log_probability'].exp(), target, weight=weights, reduction='sum')
        
        elif prediction['type'] == QuestionType.QUERY:
            predicate_num = prediction['log_probability'].size()[0]
            all_answers = [program_batch._answers for program_batch in program_batch_list]
            all_answers = list(sum(all_answers, []))
            
            target = [[a == o for o in op] for a, op in zip(all_answers, prediction['options'])]
            question_num = len(target)
            x_ind = torch.tensor(list(chain.from_iterable(repeat(a, len(b)) for a, b in enumerate(target))), dtype=torch.int64, device=self._device)
            y_ind = torch.arange(predicate_num, dtype=torch.int64, device=self._device)
            all_ones = torch.ones(predicate_num, device=self._device)
            index = torch.stack([x_ind, y_ind])
            
            if self._device.type == 'cuda':
                question_predicate_map = torch.cuda.sparse.FloatTensor(index, all_ones, 
                                torch.Size([question_num, predicate_num]), device=self._device)
            else:
                question_predicate_map = torch.sparse.FloatTensor(index, all_ones, 
                                torch.Size([question_num, predicate_num]), device=self._device)

            target = list(sum(target, []))
            target = torch.tensor(target, dtype=torch.float32, device=self._device)

            score = prediction['log_probability'] #.exp()
            loss = util.safe_log(util.mm(question_predicate_map, score.unsqueeze(1).exp())).sum() - (target * score).sum()

            # norm = torch.ones(len(target), device=self._device)
            # loss = nn.functional.binary_cross_entropy(prediction['log_probability'].exp(), target, weight=1.0 / norm, reduction='sum')

        elif prediction['type'] == QuestionType.SCENE_GRAPH:
            attr_target = torch.tensor(np.concatenate([pb._meta_data['attribute_answer'] for pb in program_batch_list], axis=0), dtype=torch.float32, device=self._device)
            attr_weight = torch.tensor(np.concatenate([pb._meta_data['attribute_weight'] for pb in program_batch_list], axis=0), dtype=torch.float32, device=self._device)

            rel_target = torch.tensor(np.concatenate([pb._meta_data['relation_answer'] for pb in program_batch_list], axis=0), dtype=torch.float32, device=self._device)
            rel_weight = torch.tensor(np.concatenate([pb._meta_data['relation_weight'] for pb in program_batch_list], axis=0), dtype=torch.float32, device=self._device)

            attr_loss = nn.functional.binary_cross_entropy(prediction['log_probability'][0].exp(), attr_target, weight=attr_weight, reduction='sum')
            rel_loss = nn.functional.binary_cross_entropy(prediction['log_probability'][1].exp(), rel_target, weight=rel_weight, reduction='sum')

            # attr_loss = nn.functional.binary_cross_entropy(prediction['log_probability'][0][:, self._ontology._non_noun_subindex].exp(), \
            #     attr_target[:, self._ontology._non_noun_subindex], weight=attr_weight[:, self._ontology._non_noun_subindex], reduction='sum')

            # w = (attr_weight[:, self._ontology._noun_subindex] * attr_target[:, self._ontology._noun_subindex]).sum(1)

            # score = prediction['log_probability'][0][:, self._ontology._noun_subindex].exp()
            # attr_loss += ((w * util.safe_log(score.exp().sum(1))).sum() - (attr_weight[:, self._ontology._noun_subindex] * attr_target[:, self._ontology._noun_subindex] * score).sum())
            
            # r_score = prediction['log_probability'][1].exp()
            # rel_loss = (rel_weight.sum(1) * util.safe_log(r_score.exp().sum(1))).sum() - (rel_weight * rel_target * r_score).sum()
            
            loss = attr_loss + rel_loss

        if 'l1_lambda' in self._config and self._config['l1_lambda'] > 0:
            all_params = torch.cat([x.view(-1) for x in filter(lambda p: p.requires_grad, self.model.parameters())])
            loss += (self._config['l1_lambda'] * torch.norm(all_params, 1) / max(1, all_params.numel()))

        return loss

    def _compute_evaluation_metrics(self, program_batch_list, prediction):
        if prediction['type'] == QuestionType.SCENE_GRAPH:
            attr_target = np.concatenate([pb._meta_data['attribute_answer'] for pb in program_batch_list], axis=0)
            attr_weight = np.concatenate([pb._meta_data['attribute_weight'] for pb in program_batch_list], axis=0) * (attr_target + prediction['answer'][0] > 0)

            rel_target = np.concatenate([pb._meta_data['relation_answer'] for pb in program_batch_list], axis=0)
            rel_weight = np.concatenate([pb._meta_data['relation_weight'] for pb in program_batch_list], axis=0) * (rel_target + prediction['answer'][1] > 0)

            nom = ((attr_target != prediction['answer'][0]) * attr_weight).sum() + ((rel_target != prediction['answer'][1]) * rel_weight).sum()
            denom = attr_weight.sum() + rel_weight.sum()

            return nom / denom

        if prediction['type'] == QuestionType.OBJECT_STATEMENT:
            answers = [list(sum(program_batch._answers, [])) for program_batch in program_batch_list]
        else:
            answers = [program_batch._answers for program_batch in program_batch_list]
        
        answers = list(sum(answers, []))
        # print(program_batch_list[0]._op_batch_list[-1]._op_name, list(zip(answers, prediction['answer'])))
        if self._config['first_answer']:
            match = [a in op[0] if len(op) > 0 else False for a, op in zip(answers, prediction['answer'])]
        else:
            if prediction['type'] == QuestionType.QUERY:
                # match = [float(any([a in o for o in op])) * float(len(opt) - len(op)) / float(len(opt) - 1) if len(op) > 0 and len(opt) > 1 else 0 \
                #     for a, op, opt in zip(answers, prediction['answer'], prediction['options'])]
                match = [float(any([a in o for o in op])) / float(len(op)) if len(op) > 0 else 0 \
                    for a, op in zip(answers, prediction['answer'])]
            else:
                match = [any([a in o for o in op]) if len(op) > 0 else False for a, op in zip(answers, prediction['answer'])]
        
        match = np.array(match, dtype=np.float32)

        if prediction['type'] == QuestionType.OBJECT_STATEMENT:
            weights = np.array(list(sum([program_batch._meta_data['weights'] for program_batch in program_batch_list], [])), dtype=np.float32)
            return 1.0 - np.average(match, weights=weights)

        if self._hardset is not None:       
            j = 0
            with open(os.path.join(self._hard_subdir, 'hard_' + program_batch_list[0]._op_batch_list[-1]._op_name + '.json'), 'a') as hard_file:
                with open(os.path.join(self._easy_subdir, 'easy_' + program_batch_list[0]._op_batch_list[-1]._op_name + '.json'), 'a') as easy_file:
                    for program_batch in program_batch_list:
                        b = program_batch.batch_size()
                        for i in range(b):
                            q_dict = program_batch._original_dicts[i]
                            if match[j] == 1:
                                easy_file.write(json.dumps(q_dict) + '\n')
                                self._easyset[q_dict['question_id']] = q_dict
                            else:
                                hard_file.write(json.dumps(q_dict) + '\n')
                                self._hardset[q_dict['question_id']] = q_dict

                            j += 1

        return 1.0 - np.mean(match)

    def _print_predictions(self, program_batch_list, prediction, is_submission):
        question_ids = [program_batch._meta_data['question_ids'] for program_batch in program_batch_list]
        question_ids = list(sum(question_ids, []))

        if is_submission:
            answers = [p[0] for p in prediction['answer']]
            self._predictions += [{'questionId': qid, 'prediction': a} for qid, a in zip(question_ids, answers)]
        else:
            if prediction['type'] == QuestionType.QUERY:
                answers = [p for p in prediction['answer']]
            else:
                answers = [p[0] for p in prediction['answer']]

            types = [['open' if program_batch._op_batch_list[-1]._op_name == 'query_attr' else 'binary' for _ in range(program_batch.batch_size())] for program_batch in program_batch_list]
            types = list(sum(types, []))

            if prediction['type'] == QuestionType.QUERY:
                self._predictions += [{'questionId': qid, 'prediction': a, 'type': t, 'options': opt} for qid, a, t, opt in zip(question_ids, answers, types, prediction['options'])]
            else:
                self._predictions += [{'questionId': qid, 'prediction': a, 'type': t} for qid, a, t in zip(question_ids, answers, types)]

        message = ''
        # if program_batch_list[0]._op_batch_list[-1]._op_name == 'query_attr':
        #     for i, (a, op) in enumerate(list(zip(answers, prediction['options']))):
        #         if len(op) < 2:
        #             message += ('\n\nAnswer: ' + a + '\nOptions: ' + str(op) +'\nProgram:\n' + str(program_batch_list[0].retrieve_instance(i)))

        # if prediction['type'] == QuestionType.QUERY:
        #     prob = prediction['log_probability'].exp()
        #     ind = 0
        #     res = []
            
        #     for op in prediction['options']:
        #         temp = []
        #         for o in op:
        #             temp.append((o, prob[ind].cpu().numpy()))
        #             ind += 1
                
        #         res.append(temp)

        #     for a, p, o, r in zip(answers, prediction['answer'], prediction['options'], res):
        #         if len(p) == 0 or all([a not in x for x in p]):
        #             temp = 'yes' if a in o else 'no'
        #             message += ('\nGold answer: ' + a + ', Prediction: ' + str(p) + \
        #                 ', Answer in options? ' + temp + '\n')
        #             message += ('Probability: ' + str(r) + '\n')
        # else:
        #     i = 0
        #     for a, p in zip(answers, prediction['answer']):
        #         if a != p[0]:
        #             prob = prediction['log_probability'].exp().cpu().numpy()
        #             message += ('Gold answer: ' + a + ', Prediction: ' + p[0] + ', Probability: ' + str(prob[i]) + '\n')
                
        #         i += 1

        return message

    def _train_epoch(self, train_loader, optimizer, batch_size, validation_loader, last_export_path_base, best_export_path_base, metric_index):

        self.model.train()
        train_batch_num = math.ceil(len(train_loader.dataset) / batch_size)

        total_loss = 0.0
        total_example_num = 0

        for (j, data) in enumerate(train_loader):
            if len(data) > 0:
                for d in data:
                    d.create_sparse_tensors()

                data = self._to_cuda(data)

                data_batch_size = sum([d.batch_size() for d in data])
                total_example_num += data_batch_size
                total_loss = self._train_batch(total_loss, optimizer, data)

                if self._config['verbose']:
                    print("Training epoch with batch of size {:4d} ({:6d}/{:6d}): {:3d}% complete...".format(
                        data_batch_size, j, train_batch_num,
                        int(j * 100.0 / train_batch_num)), end='\r')

                for d in data:
                    del d

                if (j + 1) % self._config['ckeckpointing_frequency'] == 0:
                    err = self._test_epoch(validation_loader, batch_size)
                    self.model.train()

                    if last_export_path_base is not None:
                        self._save(last_export_path_base)

                    # Checkpoint the best models so far
                    if best_export_path_base is not None:
                        if err[metric_index] <= self._best_error:
                            self._best_error = err[metric_index]
                            self._save(best_export_path_base)

                    if self._config['verbose']:
                        message = 'Step {:d}, Best Err {:5.5f}: error={:s}, loss={:5.5f}'.format(
                            self.model._global_step.int()[0], self._best_error,
                            str(self._prepare_output_metric_dict(err)),
                            total_loss / total_example_num)

                    self._logger.info('Checkpointing: {:s}'.format(message))
        
            self.model._global_step += 1

        return total_loss / total_example_num

    def _train_batch(self, total_loss, optimizer, data):

        optimizer.zero_grad()
        result = self._run_model(data, True)
        loss = self._compute_loss(data, result)
        data_batch_size = sum([d.batch_size() for d in data])
        loss /= data_batch_size
        loss.backward()
        
        nn.utils.clip_grad_norm_(self.model.parameters(), self._config['clip_norm'])
        total_loss += (data_batch_size * loss.detach().cpu().numpy())

        optimizer.step()
        return total_loss

    def _test_epoch(self, validation_loader, batch_size):

        self.model.eval()
        test_batch_num = math.ceil(len(validation_loader.dataset) / batch_size)

        with torch.no_grad():
            error = np.zeros(self._error_dim, dtype=np.float32)
            total_example_num = np.zeros(self._error_dim, dtype=np.float32)

            for (j, data) in enumerate(validation_loader):
                if len(data) > 0:
                    for d in data:
                        d.create_sparse_tensors()

                    data = self._to_cuda(data)

                    data_batch_size = sum([d.batch_size() for d in data])
                    # total_example_num += data_batch_size
                    self._test_batch(error, total_example_num, data)

                    if self._config['verbose']:
                        print("Testing epoch with batch of size {:4d} ({:6d}/{:6d}): {:3d}% complete...".format(
                            data_batch_size, j, test_batch_num,
                            int(j * 100.0 / test_batch_num)), end='\r')

                    for d in data:
                        del d

                # if self._use_cuda:
                #     torch.cuda.empty_cache()

        return error / total_example_num

    def _test_batch(self, error, total_example_num, data):
        result = self._run_model(data, False)
        data_batch_size = sum([d.batch_size() for d in data])
        err = data_batch_size * self._compute_evaluation_metrics(data, result)
        error[0] += err
        error[self._op_index[data[0]._op_batch_list[-1]._op_name]] += err

        total_example_num[0] += data_batch_size
        total_example_num[self._op_index[data[0]._op_batch_list[-1]._op_name]] += data_batch_size        

    def _predict_epoch(self, validation_loader, post_processor, file, batch_size, is_submission):

        self.model.eval()
        with torch.no_grad():

            for (j, data) in enumerate(validation_loader):
                if len(data) > 0:
                    for d in data:
                        d.create_sparse_tensors()

                    data = self._to_cuda(data)
                    self._predict_batch(data, post_processor, file, is_submission)

                    for d in data:
                        del d

    def _predict_batch(self, data, post_processor, file, is_submission):
        result = self._run_model(data, False)

        if post_processor is not None and callable(post_processor):
            message = post_processor(data, result)
        else:
            message = self._print_predictions(data, result, is_submission)
        
        # if message not in (None, ''):
        #     print('-----------------------------------------------------------------------------------', file=file)
        #     print(message, file=file)
        #     input("Press Enter to continue...")

    def _visualize_epoch(self, validation_loader, image_path):

        self.model.eval()
        with torch.no_grad():

            for (j, data) in enumerate(validation_loader):
                if len(data) > 0:
                    for d in data:
                        d.create_sparse_tensors()

                    data = self._to_cuda(data)
                    self._visualize_batch(data, image_path)

                    for d in data:
                        del d

    def _draw_boxes(self, image, bboxes, alpha):
        output = image.copy()

        for i in range(bboxes.shape[0]):
            overlay = output.copy()
            overlay = cv2.rectangle(overlay, tuple(bboxes[i, 0:2]), tuple(bboxes[i, 0:2] + bboxes[i, 2:]), (0, 255, 0), 3)
            cv2.addWeighted(overlay, alpha[i], output, 1 - alpha[i], 0, output)

        return output         

    def _visualize_batch(self, data, image_path):
        font = cv2.FONT_HERSHEY_SIMPLEX 
        fontScale = 0.4
        thickness = 1
        vertical_offset = 100

        result, trace = self.model(data, False, return_trace=True, modulator_switch=data[0]._question_type != QuestionType.QUERY)
        
        for k, d in enumerate(data):
            batch_size = d.batch_size()
            
            for j in range(batch_size):
                image = cv2.imread(os.path.join(image_path, d._meta_data['image_ids'][j]) + '.jpg')
                image = cv2.copyMakeBorder(image, vertical_offset, 0, 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
                question = d._meta_data['questions'][j]
                print('\n-----------------------------------------')
                print(question)
                image = cv2.putText(image, question, (10, 20), font, fontScale, (0, 255, 0), thickness, cv2.LINE_AA)
                cv2.imshow(question, image)
                cv2.waitKey(0)
                # input("Press Enter to continue...")

                bboxes = d._object_features[d._object_batch_index == j, -4:].cpu().numpy()
                bboxes[:, 1] += vertical_offset
                object_num = bboxes.shape[0]
                cv2.imshow(question, self._draw_boxes(image, bboxes, np.ones(object_num, dtype=np.float32)))
                cv2.waitKey(0)

                program_string = ''
                for i, op_batch in enumerate(d._op_batch_list):
                    args = '(' + ','.join([str(a[j]) for a in op_batch._arguments]) + ')'
                    if i > 0:
                        program_string += ' > '
                    
                    program_string += (op_batch._op_name + args)
                    print(program_string)

                    if i == len(d._op_batch_list) - 1:
                        answer = 'Predicted answer: ' + ','.join(result['answer'][j])
                        print(answer)
                        ground_truth = 'True answer: ' + d._answers[j]
                        print(ground_truth)
                        im = cv2.putText(im, program_string, (10, 40), font, fontScale, (255, 255, 0), thickness, cv2.LINE_AA)
                        im = cv2.putText(im, answer, (10, 60), font, fontScale, (255, 255, 255), thickness, cv2.LINE_AA)
                        im = cv2.putText(im, ground_truth, (10, 80), font, fontScale, (0, 255, 255), thickness, cv2.LINE_AA)
                        cv2.imshow(question, im)
                    else:
                        im = image.copy()
                        im = cv2.putText(im, program_string, (10, 40), font, fontScale, (255, 255, 0), thickness, cv2.LINE_AA)
                        alpha = trace[k][i]._log_attention[j, :].exp().cpu().numpy()
                        im = self._draw_boxes(im, bboxes, alpha)
                        cv2.imshow(question, im)
                    
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

    def train(self, train_path, validation_path, train_batch_size, test_batch_size, collater, optimizer, metric_index=0, last_export_path_base=None,
              best_export_path_base=None, load_model=None, reset_step=False):
        "Trains the model."
        assert self._model is not None, "No model is set."

        train_data_manager = GQADataManager(train_path, self._ontology, self._config['in_memory'], max_cache_size=self._config['max_cache_size'], keep_original_dict=False)
        train_loader = train_data_manager.get_loader(train_batch_size, collater, self._num_cores, self._use_cuda)

        validation_data_manager = GQADataManager(validation_path, self._ontology, self._config['in_memory'], max_cache_size=self._config['max_cache_size'], keep_original_dict=False)
        validation_loader = validation_data_manager.get_loader(test_batch_size, collater, self._num_cores, self._use_cuda)

        errors = np.zeros(
            (self._error_dim, self._config['epoch_num'],
             self._config['repetition_num']), dtype=np.float32)

        losses = np.zeros(
            (self._config['epoch_num'], self._config['repetition_num']),
            dtype=np.float32)

        self._best_error = np.inf
        self._hardset = None
        self._easyset = None

        # if self._use_cuda:
        #     torch.backends.cudnn.benchmark = True
        # torch.autograd.set_detect_anomaly(True)

        for rep in range(self._config['repetition_num']):

            if load_model == "best" and best_export_path_base is not None:
                self._load(best_export_path_base)
            elif load_model == "last" and last_export_path_base is not None:
                self._load(last_export_path_base)

            if reset_step:
                self._reset_global_step()

            for epoch in range(self._config['epoch_num']):

                # Training
                try:
                    start_time = time.time()
                    losses[epoch, rep] = self._train_epoch(train_loader, optimizer, train_batch_size, validation_loader, last_export_path_base, best_export_path_base, metric_index)

                    if self._use_cuda:
                        torch.cuda.empty_cache()

                    # Validation
                    errors[:, epoch, rep] = self._test_epoch(validation_loader, test_batch_size)
                    duration = time.time() - start_time
                
                finally:
                    if last_export_path_base is not None:
                        self._save(last_export_path_base)

                # Checkpoint the best models so far
                if best_export_path_base is not None:
                    if errors[metric_index, epoch, rep] < self._best_error:
                        self._best_error = errors[metric_index, epoch, rep]
                        self._save(best_export_path_base)

                if self._use_cuda:
                    torch.cuda.empty_cache()

                if self._config['verbose']:
                    message = 'Step {:d}, Best Err {:5.5f}: error={:s}, loss={:5.5f}'.format(
                        self.model._global_step.int()[0], self._best_error,
                        str(self._prepare_output_metric_dict(errors[:, epoch, rep])),
                        losses[epoch, rep])

                    self._logger.info('Rep {:2d}, Epoch {:2d}: {:s}'.format(rep + 1, epoch + 1, message))
                    self._logger.info('Time spent: %s seconds' % duration)

        # if self._use_cuda:
        #     torch.backends.cudnn.benchmark = False

        if best_export_path_base is not None:
            # Save losses and errors
            base = os.path.relpath(best_export_path_base)
            np.save(os.path.join(base, "losses"), losses, allow_pickle=False)
            np.save(os.path.join(base, "errors"), errors, allow_pickle=False)

            # # Save the model
            # self._save(best_export_path_base)

        return self.model, errors, losses

    def test(self, test_path, batch_size, collater, import_path_base=None):
        "Tests the model and generates test stats."
        assert self._model is not None, "No model is set."

        test_data_manager = GQADataManager(test_path, self._ontology, self._config['in_memory'], max_cache_size=self._config['max_cache_size'], keep_original_dict=self._hardset_path is not None)
        test_loader = test_data_manager.get_loader(batch_size, collater, self._num_cores, self._use_cuda, is_random=False)

        if import_path_base is not None:
            self._load(import_path_base)

        if self._hardset_path is not None:
            self._hardset = {}
            self._easyset = {}
        else:
            self._hardset = None
            self._easyset = None

        start_time = time.time()
        error = self._test_epoch(test_loader, batch_size)
        duration = time.time() - start_time

        if self._use_cuda:
            torch.cuda.empty_cache()

        if self._config['verbose']:
            message = 'error={:s}'.format(str(self._prepare_output_metric_dict(error)))
            self._logger.info(message)
            self._logger.info('Time spent: %s seconds' % duration)

        if self._hardset_path is not None:
            with open(os.path.join(self._hardset_path, '_'.join([self._hardset_prefix, 'hard.json'])), 'w') as f:
                json.dump(self._hardset, f)
            with open(os.path.join(self._hardset_path, '_'.join([self._hardset_prefix, 'easy.json'])), 'w') as f:
                json.dump(self._easyset, f)

        return error, duration

    def predict(self, input_path, batch_size, collater, out_file, import_path_base=None, post_processor=None, is_submission=False):
        "Produces predictions for the trained model."
        assert self._model is not None, "No model is set."

        test_data_manager = GQADataManager(input_path, self._ontology, self._config['in_memory'], max_cache_size=self._config['max_cache_size'], keep_original_dict=self._hardset_path is not None)
        test_loader = test_data_manager.get_loader(batch_size, collater, self._num_cores, self._use_cuda, is_random=False)

        if import_path_base is not None:
            self._load(import_path_base)

        self._predictions = []
        if self._hardset_path is not None:
            self._hardset = {}
            self._easyset = {}
        else:
            self._hardset = None
            self._easyset = None

        start_time = time.time()
        self._predict_epoch(test_loader, post_processor, out_file, batch_size, is_submission)
        duration = time.time() - start_time

        json.dump(self._predictions, out_file)

        if self._use_cuda:
            torch.cuda.empty_cache()

        if self._hardset_path is not None:
            with open(os.path.join(self._hardset_path, '_'.join([self._hardset_prefix, 'hard.json'])), 'w') as f:
                json.dump(self._hardset, f)
            with open(os.path.join(self._hardset_path, '_'.join([self._hardset_prefix, 'easy.json'])), 'w') as f:
                json.dump(self._easyset, f)

        if self._config['verbose']:
            self._logger.info('Time spent: %s seconds' % duration)

    def visualize(self, input_path, collater, image_path, import_path_base=None):
        assert self._model is not None, "No model is set."

        test_data_manager = GQADataManager(input_path, self._ontology, self._config['in_memory'], max_cache_size=self._config['max_cache_size'], keep_original_dict=False)
        test_loader = test_data_manager.get_loader(1, collater, self._num_cores, self._use_cuda)

        if import_path_base is not None:
            self._load(import_path_base)
        
        self._visualize_epoch(test_loader, image_path)

        if self._use_cuda:
            torch.cuda.empty_cache()
