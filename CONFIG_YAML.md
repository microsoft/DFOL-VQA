# The Full List of Hyper-parameters Provided in the Config YAML File

+ **model_name**: The model name.

+ **version**: The model version.

+ **train_path**: The path to the pre-processed training questions.

+ **train_object_path**: The path to the GQA HDF5 file containing the object (Faster-RCNN) featurizations.

+ **train_object_info_path**: The path to the GQA objects JSON file containing the meta-data for the HDF5 file.

+ **validation_path**: The path to the pre-processed validation questions.

+ **test_path**: The path to the test path.

+ **image_path**: The path to the original GQA images (only required for visualization).

+ **model_path**: The output path for the model's checkpoints as well as the prediction output JSON.

+ **attribute_file**: The absolute path to [gqa_all_attribute.json](https://github.com/microsoft/DFOL-VQA/blob/main/src/nsvqa/data/metadata/gqa_all_attribute.json).

+ **class_file**: The absolute path to [gqa_all_class.json](https://github.com/microsoft/DFOL-VQA/blob/main/src/nsvqa/data/metadata/gqa_all_class.json).

+ **relation_file**: The absolute path to [gqa_relation.json](https://github.com/microsoft/DFOL-VQA/blob/main/src/nsvqa/data/metadata/gqa_relation.json).

+ **word_embedding_file**: The path to the Glove word-embedding text file.

+ **vocabulary_file**: The absolute path to [gqa_vocab.json](https://github.com/microsoft/DFOL-VQA/blob/main/src/nsvqa/data/metadata/gqa_vocab.json).

+ **h5_prefix**: The prefix for the HDF5 object features chunk files (default: "gqa_objects").

+ **h5_chunk_num**: The number of chunks for the HDF5 object features (default: 16).

+ **repetition_num**: The number of runs for the entire training loop.

+ **epoch_num**: The number of epochs for each training run.

+ **error_dim**: The number of test metrics reported by the model on the validation/test splits (default: 1).

+ **metric_index**: The index of the metric on the validation split based on which the best model checkpoint is selected across the training steps. 

+ **train_batch_size**: The number of questions in a single train batch.

+ **test_batch_size**: The number of questions in a single test/validation batch.

+ **learning_rate**: The learning rate.

+ **weight_decay**: The weight decay.

+ **dropout**: The dropout rate.

+ **clip_norm**: The clip norm rate used for gradient clipping.

+ **verbose**: The flag indicating whether to show the execution logs (true/false).

+ **max_cache_size**: The maximum cache size (default: 100000).

+ **box_features_dim**: The dimension of the GQA objects feature vectors (default: 2048).

+ **oracle_input_dim**: The input (output) dimension of the visual oracle (the initial featurizer network).

+ **oracle_output_dim**: The output dimension of the visual oracle (default: 1).

+ **word_embedding_dim**: The word embedding dimension (default: 300 for Glove)

+ **classifier_oracle**: The flag indicating whether to use the classifier-based architecture for the visual oracle (default: true)

+ **featurizer_layers_config**: A list containing the dimensions of the hidden layers for the MLP representing the featurizer network.

+ **attribute_network_layers_config**: A list containing the dimensions of the hidden layers for the MLP representing the attribute classifier in the visual oracle.

+ **relation_network_layers_config**: A list containing the dimensions of the hidden layers for the MLP representing the relation classifier in the visual oracle.

+ **operator_layers_config**: N/A (default: [])

+ **normalize_oracle**: The flag indicating whether to normalize the output probabilities of the visual oracle based on their categories (default: true).

+ **freeze_featurizer**: The flag indicating whether to freeze the parameters of the featurizer network.

+ **freeze_attribute_network**: The flag indicating whether to freeze the parameters of the attribute network within the visual oracle.

+ **freeze_relation_network**: The flag indicating whether to freeze the parameters of the relation network within the visual oracle.

+ **freeze_embedding_network**: The flag indicating whether to freeze the parameters of the last layer of the visual oracle (aka the embedding layer).

+ **activate_attention_transfer**: The flag indicating whether to activate the attention calibration mechanism.

+ **attention_transfer_state_dim**: The hidden dimention of the LSTM cell used in the attention calibration network.

+ **freeze_attention_network**: he flag indicating whether to freeze the parameters of the attention calibration network.

+ **trainable_gate**: N/A (default: false)

+ **likelihood_threshold**: The minimum likelihood an answer must have to be considered as a potential option for an open question.

+ **hard_mode**: The flag indicating whether to use Min/Max for logical conjunction/disjunction at the test time (default: false).

+ **cpu_cores_num**: Number of CPU cores used for fetching data (automatically set to the maximum cores available if not specified).

+ **in_memory**: N/A (default: true)

+ **gpu_num**: The maximum number of GPUs the framework is allowed to use for parallelization.

+ **ckeckpointing_frequency**: The number of training steps (batches) after which checkpointing happens.

+ **first_answer**: The flag indicating whether to return only the first answer for an open question in the case of tied likelihoods for mutiple options at the test time (default: false).
