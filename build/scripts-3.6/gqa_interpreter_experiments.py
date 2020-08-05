# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file
# in the project root for full license information.

import torch
import torch.nn as nn
import torch.optim as optim

import argparse, math

from nsvqa.base_experiment import ExperimentBase
from nsvqa.nn.interpreter.batch_gqa_interpreter import BatchGQAInterpreter
from nsvqa.nn.vision.classifier_oracle import ClassifierOracle
from nsvqa.nn.interpreter.batch_gqa_ops import GQAOntology
from nsvqa.data.batch_gqa_boxfeatures_pipeline import BatchGQABoxFeaturizer, BatchGQABoxFeaturesCollator
from nsvqa.nn.interpreter.util import ClusteredLogSoftmax

class RegularMLP(nn.Module):

    def __init__(self, input_dim, output_dim, layers_config, dropout):
        super(RegularMLP, self).__init__()

        if layers_config is None:
            self._network = None
        else:
            layers = []
            last_dim = input_dim
            for i in layers_config:
                layers += [nn.Dropout(dropout), nn.Linear(last_dim, i), nn.ELU()]
                last_dim = i

            layers += [nn.Dropout(dropout), nn.Linear(last_dim, output_dim), nn.Sigmoid()]
            self._network = nn.Sequential(*layers)

    def forward(self, input_tensor):
        return self._network(input_tensor) if self._network is not None else input_tensor

######################################################################################################################################

class LoglikelihoodMLP(nn.Module):

    def __init__(self, input_dim, layers_config, dropout):
        super(LoglikelihoodMLP, self).__init__()

        layers = []        
        last_dim = input_dim
        for i in layers_config:
            layers += [nn.Dropout(dropout), nn.Linear(last_dim, i), nn.ELU()]
            last_dim = i

        layers += [nn.Dropout(dropout), nn.Linear(last_dim, 1), nn.LogSigmoid()]
        self._network = nn.Sequential(*layers)

    def forward(self, input_tensor):
        # return -self._network(input_tensor) - 1.0
        return self._network(input_tensor)

######################################################################################################################################

class EmbeddingLayer(nn.Module):

    def __init__(self, input_dim, output_dim, dropout, weights=None, biases=None, freeze_bias=False, cluster_index=None):
        super(EmbeddingLayer, self).__init__()

        linear = nn.Linear(input_dim, output_dim, bias=not freeze_bias)

        if weights is not None:
            linear.weight = nn.Parameter(weights)

        if biases is not None and not freeze_bias:
            linear.bias = nn.Parameter(biases)

        self._network = nn.Sequential(nn.Dropout(dropout), linear, \
            ClusteredLogSoftmax(cluster_index) if cluster_index is not None else nn.LogSigmoid())

    def forward(self, input_tensor):
        return self._network(input_tensor)

######################################################################################################################################

class GQAObjectBoxExperiment(ExperimentBase):

    def build_ontology(self, config, logger):
        if config['verbose'] and self._local_rank == 0:
            logger.info('Building the ontology...')

        rel = config['relation_file'] if 'relation_file' in config else None
        frequency = config['frequency_file'] if 'frequency_file' in config else None

        return GQAOntology(config['attribute_file'], config['class_file'], config['vocabulary_file'], config['word_embedding_file'], \
            relation_json_path=rel, frequency_json_path=frequency)

    def build_embedding_clusters(self, ontology):
        concept_num = len(ontology._vocabulary['idx_to_arg'])
        result = torch.zeros(concept_num, dtype=torch.int64)
        result[ontology._relation_index] = 2

        for i, (k, v) in enumerate(ontology._attribute_dict.items()):
            for val in v:
                if val in ontology._vocabulary['arg_to_idx']:
                    result[ontology._vocabulary['arg_to_idx'][val] - 1] = i + 3
        
        result[ontology._noun_index] = 1

        return result

    def build_neural_modules(self, config, ontology, logger):
        featurizer_network = RegularMLP(config['box_features_dim'], config['oracle_input_dim'],\
            config['featurizer_layers_config'], config['dropout'])

        if config['freeze_featurizer']:
            featurizer_network.requires_grad_(requires_grad=False)

        # Build the attention transfer modulator
        if config['activate_attention_transfer']:
            if config['verbose'] and self._local_rank == 0:
                logger.info('Building the Attention Transfer Network...')
            # atm = AttentionTransferModulator(config['word_embedding_dim'] + 17, config['attention_transfer_state_dim'], 1, dropout=config['dropout'])
            output_dim = 4
            max_activation = 10.0

            forward_attention_network = nn.LSTMCell(config['word_embedding_dim'] + 1 + 17, config['attention_transfer_state_dim'])
            backward_attention_network = nn.LSTMCell(config['word_embedding_dim'] + 1 + 17, config['attention_transfer_state_dim']) 
            attention_output_network = nn.Sequential(nn.Linear(2 * config['attention_transfer_state_dim'], output_dim), nn.Sigmoid())
            
            attention_output_network[0].weight = nn.Parameter(torch.zeros(output_dim, 2 * config['attention_transfer_state_dim']))
            temp_tensor = -math.log(max_activation - 1) * torch.ones(output_dim)
            if output_dim >= 4:
                temp_tensor[3] = 0
            if output_dim >= 5:
                temp_tensor[4] = 10
            attention_output_network[0].bias = nn.Parameter(temp_tensor)
            
            if config['freeze_attention_network']:
                # atm.requires_grad_(requires_grad=False)
                forward_attention_network.requires_grad_(requires_grad=False)
                backward_attention_network.requires_grad_(requires_grad=False)
                attention_output_network.requires_grad_(requires_grad=False)
        else:
            # atm = None
            forward_attention_network = None
            backward_attention_network = None 
            attention_output_network = None
        
        if config['oracle_output_dim'] == 1:
            if config['classifier_oracle']:
                attribute_network = RegularMLP(config['oracle_input_dim'] + 4, config['word_embedding_dim'], config['attribute_network_layers_config'], config['dropout'])

                concept_num = len(ontology._vocabulary['idx_to_arg'])
                embedding_input_dim = config['oracle_input_dim'] + 4 if config['attribute_network_layers_config'] is None else config['word_embedding_dim']
                weights = torch.zeros(concept_num, embedding_input_dim)
                torch.nn.init.normal_(weights)
                weights[:, :config['word_embedding_dim']] = torch.from_numpy(ontology.get_embeddings(ontology._vocabulary['idx_to_arg']))
                biases = torch.zeros(concept_num)

                # if 'normalize_oracle' in config and config['normalize_oracle']:
                #     cluster_index = self.build_embedding_clusters(ontology)
                #     embedding_network = EmbeddingLayer(embedding_input_dim, concept_num, config['dropout'], \
                #         weights, biases, config['freeze_embedding_bias'] if 'freeze_embedding_bias' in config else False, cluster_index=cluster_index)
                # else:
                embedding_network = EmbeddingLayer(embedding_input_dim, concept_num, config['dropout'], \
                    weights, biases, config['freeze_embedding_bias'] if 'freeze_embedding_bias' in config else False)

                if 'relation_features_dim' in config:
                    relation_network = RegularMLP(config['relation_features_dim'], embedding_input_dim, config['relation_network_layers_config'], config['dropout'])
                else:
                    relation_network = RegularMLP(2*config['oracle_input_dim'] + 2*4 + 4, embedding_input_dim, config['relation_network_layers_config'], config['dropout'])

                if config['freeze_attribute_network']:
                    attribute_network.requires_grad_(requires_grad=False)

                if config['freeze_relation_network']:
                    relation_network.requires_grad_(requires_grad=False)

                if config['freeze_embedding_network']:
                    embedding_network.requires_grad_(requires_grad=False)

                return {'featurizer_network': featurizer_network, 'attribute_network': attribute_network, 'relation_network': relation_network, 'embedding_network': embedding_network, \
                        'forward_attention_network': forward_attention_network, 'backward_attention_network': backward_attention_network, 'attention_output_network': attention_output_network}
            else:
                attribute_network = LoglikelihoodMLP(config['oracle_input_dim'] + config['word_embedding_dim'] + 2, config['attribute_network_layers_config'], config['dropout'])
                relation_network = LoglikelihoodMLP(2*config['oracle_input_dim'] + config['word_embedding_dim'] + 2*4 + 4, config['relation_network_layers_config'], config['dropout'])
        else:
            attribute_network = RegularMLP(config['oracle_input_dim'] + config['word_embedding_dim'] + 2, config['oracle_output_dim'], config['attribute_network_layers_config'], config['dropout'])
            
            if 'relation_features_dim' in config:
                relation_network = RegularMLP(config['relation_features_dim'] + config['word_embedding_dim'], config['oracle_output_dim'], config['relation_network_layers_config'], config['dropout'])
            else:
                relation_network = RegularMLP(2*config['oracle_input_dim'] + config['word_embedding_dim'] + 2*4 + 4, config['oracle_output_dim'], config['relation_network_layers_config'], config['dropout'])

        if config['freeze_attribute_network']:
            attribute_network.requires_grad_(requires_grad=False)

        if config['freeze_relation_network']:
            relation_network.requires_grad_(requires_grad=False)

        return {'featurizer_network': featurizer_network, 'attribute_network': attribute_network, 'relation_network': relation_network, \
            'forward_attention_network': forward_attention_network, 'backward_attention_network': backward_attention_network, 'attention_output_network': attention_output_network}

    def build_interpreter(self, config, neural_dict, ontology, logger):
        # Build the featurizer
        if config['verbose'] and self._local_rank == 0:
            logger.info('Building the Box Featurizer...')
        featurizer = BatchGQABoxFeaturizer(featurizer_network=neural_dict['featurizer_network'])

        # Build the oracle
        if config['verbose'] and self._local_rank == 0:
            logger.info('Building the Classifier Oracle...')
        oracle = ClassifierOracle(ontology, neural_dict['attribute_network'], neural_dict['relation_network'], neural_dict['embedding_network'], \
            normalize='normalize_oracle' in config and config['normalize_oracle'], cached=True)

        hard_mode = config['hard_mode'] if 'hard_mode' in config else False
        
        # Build the interpreter
        if config['oracle_output_dim'] == 1:
            if config['verbose'] and self._local_rank == 0:
                logger.info('Building the non-trainable interpreter...')

            calibrator = None
            visual_rule_learner = None

            interpreter = BatchGQAInterpreter(config['model_name'], oracle, ontology, featurizer, trainable_gate=config['trainable_gate'], likelihood_threshold=config['likelihood_threshold'], hard_mode=hard_mode,#attention_transfer_modulator=atm)
                    attention_transfer_state_dim=config['attention_transfer_state_dim'], \
                    forward_attention_network=neural_dict['forward_attention_network'], \
                    backward_attention_network=neural_dict['backward_attention_network'], \
                    attention_output_network=neural_dict['attention_output_network'],
                    apply_modulation_everywhere=config['apply_modulation_everywhere'] if 'apply_last_modulation' in config else True,
                    cached=True, visual_rule_learner=visual_rule_learner, calibrator=calibrator)
        else:
            if config['verbose'] and self._local_rank == 0:
                logger.info('Building the trainable interpreter...')
            interpreter = BatchGQAInterpreter(config['model_name'], oracle, ontology, featurizer, lambda x: LoglikelihoodMLP(x, config['operator_layers_config'], config['dropout']),\
                    feature_dim=config['oracle_output_dim'], trainable_gate=config['trainable_gate'], likelihood_threshold=config['likelihood_threshold'], hard_mode=hard_mode, #attention_transfer_modulator=atm)
                    attention_transfer_state_dim=config['attention_transfer_state_dim'], \
                    forward_attention_network=neural_dict['forward_attention_network'], \
                    backward_attention_network=neural_dict['backward_attention_network'], \
                    attention_output_network=neural_dict['attention_output_network'],
                    apply_modulation_everywhere=config['apply_modulation_everywhere'] if 'apply_last_modulation' in config else True)

        return interpreter

    def build_collater(self, config, ontology, logger, use_cuda):
        if config['verbose'] and self._local_rank == 0:
            logger.info('Building the Box Collater...')

        split_num = 1
        if use_cuda:
            if config['gpu_num'] is not None:
                split_num = min(config['gpu_num'], torch.cuda.device_count())
            else:
                split_num = torch.cuda.device_count()
        
        return BatchGQABoxFeaturesCollator(config['train_object_path'], config['h5_prefix'], config['h5_chunk_num'], config['train_object_info_path'], ontology,\
                                            split_num=split_num)

    def build_optimizer(self, trainer, config, logger):

        if config['verbose'] and self._local_rank == 0:
            logger.info('Building the Adam optimizer...')

        return optim.Adam(trainer.get_parameter_list(), lr=config['learning_rate'],
            weight_decay=config['weight_decay'])

######################################################################################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='The configuration yaml file')
    parser.add_argument('-t', '--test', help='The test mode', action='store_true')
    parser.add_argument('-l', '--load_model', help='Load the previous model')
    parser.add_argument('-c', '--cpu_mode', help='Run on CPU', action='store_true')
    parser.add_argument('-r', '--reset', help='Reset the global step', action='store_true')
    parser.add_argument('-s', '--seed', help='Random seed', type=int, default=0)
    parser.add_argument('-p', '--predict', help='Make predictions', action='store_true')
    parser.add_argument('-v', '--visualize', help='Visualize reasoning', action='store_true')
    parser.add_argument('-o', '--hardset_path', help='The output path for hardset', type=str, default=None)
    parser.add_argument('-u', '--submission', help='Is the prediction file for submission', action='store_true')
    parser.add_argument("--local_rank", default=0, type=int)
    
    args = parser.parse_args()

    experiment = GQAObjectBoxExperiment()
    experiment.run(args.local_rank, args.config, not args.test, args.load_model, 
            not args.cpu_mode, args.reset, args.predict, args.visualize, args.seed, hardset_path=args.hardset_path, is_submission=args.submission)