# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file
# in the project root for full license information.

import yaml, logging, os, sys
import torch
import numpy as np

from nsvqa.train.trainer import VQATrainer

class ExperimentBase(object):

    def build_neural_modules(self, config, ontology, logger):
        raise NotImplementedError("Subclass must implement abstract method")

    def build_interpreter(self,config, neural_dict, ontology, logger):
        raise NotImplementedError("Subclass must implement abstract method")

    def build_collater(self, config, ontology, logger):
        raise NotImplementedError("Subclass must implement abstract method")

    def build_optimizer(self, trainer, config, logger):
        raise NotImplementedError("Subclass must implement abstract method")

    def build_ontology(self, config, logger):
        raise NotImplementedError("Subclass must implement abstract method")

    def build_model(self, config, ontology, logger):
        neural_dict = self.build_neural_modules(config, ontology, logger)
        return self.build_interpreter(config, neural_dict, ontology, logger)

    def run(self, local_rank, config_file, is_training, load_model, use_cuda, reset_step, predict, visualize, random_seed=None, hardset_path=None, is_submission=False):
        "Runs the train/test/predict procedures."
        self._local_rank = local_rank

        if random_seed is not None:
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)

        use_cuda = use_cuda and torch.cuda.is_available()

        # Set the configurations (from either JSON or YAML file)
        if isinstance(config_file, dict):
            config = config_file
        else:
            with open(config_file, 'r') as f:
                config = yaml.load(f, Loader=yaml.FullLoader)

        # Set the logger
        format = '[%(levelname)s] %(asctime)s - %(name)s: %(message)s'
        logging.basicConfig(level=logging.DEBUG, format=format)
        logger = logging.getLogger(config['model_name'] + ' (' + config['version'] + ')')

        best_model_path_base = os.path.join(os.path.relpath(config['model_path']),
                                            config['model_name'], config['version'], "best")

        last_model_path_base = os.path.join(os.path.relpath(config['model_path']),
                                            config['model_name'], config['version'], "last")

        if not os.path.exists(best_model_path_base):
            os.makedirs(best_model_path_base)

        if not os.path.exists(last_model_path_base):
            os.makedirs(last_model_path_base)

        # Build ontology
        ontology = self.build_ontology(config, logger)

        # Build the trainer
        if 'apex' in config and config['apex']:
            from nsvqa.train.apex_trainer import ApexVQATrainer
            if config['verbose'] and local_rank == 0:
                logger.info("Building Apex Trainer...")
            trainer = ApexVQATrainer(local_rank, config, logger, ontology)
        else:
            if config['verbose'] and local_rank == 0:
                logger.info("Building Regular Trainer...")
            trainer = VQATrainer(use_cuda, config, logger, ontology, hardset_path=hardset_path)

        # Build the model
        trainer.model = self.build_model(config, ontology, logger)

        # Build the collator
        collater = self.build_collater(config, ontology, logger, use_cuda)

        # Build the optimizer
        optimizer = self.build_optimizer(trainer, config, logger)

        train_error, train_loss = None, None
        test_error, test_time = None, None

        if config['verbose'] and local_rank == 0:
            logger.info("The model parameter count is %d." % trainer.model.parameter_count())

        # Training
        if is_training:
            if config['verbose'] and local_rank == 0:
                logger.info("Starting the training phase...")
                logger.info("Train path: %s" % config['train_path'])
                logger.info("Validation path: %s" % config['validation_path'])

            _, train_error, train_loss = trainer.train(config['train_path'], config['validation_path'], config['train_batch_size'], config['test_batch_size'], collater, 
                        optimizer, metric_index=config['metric_index'], last_export_path_base=last_model_path_base,
                        best_export_path_base=best_model_path_base, load_model=load_model, reset_step=reset_step)

        if config['verbose'] and local_rank == 0:
            logger.info("Starting the test phase...")
            logger.info("Test path: %s" % config['test_path'])

        if load_model == "last":
            import_path_base = last_model_path_base
        elif load_model == "best":
            import_path_base = best_model_path_base
        else:
            import_path_base = None

        if visualize:
            trainer.visualize(config['test_path'], collater, config['image_path'], import_path_base=import_path_base)
        elif predict:
            directory, file_name = os.path.split(config['test_path'])
            if file_name[0:2] == 'h5':
                config['test_path'] = os.path.join(directory, 'p' + file_name[2:])

            prediction_path = os.path.join(os.path.relpath(config['model_path']), 'predictions',
                                            config['model_name'], config['version'])
            if not os.path.exists(prediction_path):
                os.makedirs(prediction_path)
            
            with open(os.path.join(prediction_path, 'prediction_' + file_name + '.json'), 'w') as out_file:
                trainer.predict(config['test_path'], config['test_batch_size'], collater, out_file, import_path_base=import_path_base, is_submission=is_submission)

        if hardset_path is not None:
            directory, file_name = os.path.split(config['test_path'])
            if file_name[0:2] == 'h5':
                config['test_path'] = os.path.join(directory, 'p' + file_name[2:])
            
        if not is_submission:
            test_error, test_time = trainer.test(config['test_path'], config['test_batch_size'], collater, import_path_base=import_path_base)
        
        return {'model': trainer.model, 'train_loss': train_loss, 'train_error': train_error, 'test_error': test_error, 'test_time': test_time} 

