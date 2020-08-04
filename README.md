# Differentiable First Order Logic Reasoning for Visual Question Answering

The differentiable first order logic reasoning framework (termed as **&#8711;-FOL**) is a neuro-symbolic architecture for visual question answering (VQA) built upon formulating questions about visual scenes as first-order logic (FOL) formulas. For more technical details, please refer to our paper:

**Saeed Amizadeh, Hamid Palangi, Alex Polozov, Yichen Huang and Kazuhito Koishida, *Neuro-Symbolic Visual Reasoning: Disentangling “Visual” from “Reasoning”*, In Proceedings of the 37th International Conference on Machine Learning (ICML), pp. 10696--10707, Vienna, Austria, 2020. [[PDF]](https://proceedings.icml.cc/static/paper_files/icml/2020/6156-Paper.pdf) [[Supplement]](https://proceedings.icml.cc/static/paper_files/icml/2020/6156-Supplemental.pdf) [[Video]](https://icml.cc/virtual/2020/poster/6760) [[bib]](https://proceedings.icml.cc/static/paper_files/icml/2020/6156-Bibtex.bib)**

If you are using this code for any research/publication purposes, please make sure to cite our paper:

```
@incollection{icml2020_6156,
 author = {Amizadeh, Saeed and Palangi, Hamid and Polozov, Oleksandr and Huang, Yichen and Koishida, Kazuhito},
 booktitle = {Proceedings of the 37th International Conference on Machine Learning (ICML-2020)},
 pages = {10696--10707},
 title = {Neuro-Symbolic Visual Reasoning: Disentangling "Visual" from "Reasoning"},
 year = {2020}
}
```

# Setup

## Prerequisites

* Python 3.6 or higher.
* PyTorch 1.3.0 or higher.

Run:

```
> python setup.py
```

# Data Preparation and Model Configuration

## Getting the Data

Download the [GQA dataset](https://cs.stanford.edu/people/dorarad/gqa/download.html).\
Download the [GloVe word-embedding 42B-300d](http://nlp.stanford.edu/data/glove.42B.300d.zip).

## Preprocessing the GQA Question JSON Files 

In this project, in order to efficiently process the GQA questions, we use our own JSON format to represent the GQA questions. For increasing the data loading efficiency even further, our pipeline also accepts the question files in the HDF5 binary format. The latter is strongly recommended when training on large scale data. In order to convert the original GQA question files into our JSON and binary formats, one needs to run the following: 

`> cd DFOL-VQA/src`\
`> python gqa_preprocess.py path/to/gqa_question_json output/path -b -g`

Here the **-b** option makes sure to create the binary HDF5 files in addition to the pre-processed JSON files. The **-g** option drops the "global" type questions from the outputted files as we currently do not support encoding the global questions in our code base. Note that in the output directory, the questions are put in separate files based on their type. This is merely for more efficient data loading. One can furthermore segragate the questions based on the number of hops in their corresponding programs by deploying the **-l** option. This is especially useful when for example one is implementing a curriculum training strategy where each curriculum contains questions of a certain length.

For example to generate the train split, the following command must be run:

`> python gqa_preprocess.py .../GQA/questions1.2/train_all_questions .../GQA/p_train_all_questions/p_train_all_questions.json -b -g`

The output preprocessed files appear in a sub-directory with the same name as the input file except that it is prefixed by 'p_'. 
Since the **-b** option is deployed, the preprocessed data is also generated in the HDF5 format (the corresponding sub-directory is prefixed by 'h5_').\
**Note that the training config file needs to point to 'h5_' directories for efficient data loading.**

## Preparing the Config YAML

The config YAML file contains all the hyper-parameters and global variables required for training/testing a **&#8711;-FOL** model and should be prepared beforehand as such. A sample config file is provided [here](https://github.com/microsoft/DFOL-VQA/config/sample_config.yaml). For the complete list of hyper-parameters and their descriptions, please see [here](https://github.com/microsoft/DFOL-VQA/CONFIG_YAML.md).

# Running the **&#8711;-FOL** Framework

## Training

Once the config YAML is ready, the training loop can be started by running:

`> cd DFOL-VQA/src`\
`> python gqa_interpreter_experiments.py path/to/config/yaml -s 0`

The **-s** option specifies the random seed. The model parameters are initialized by random values unless the **-l** option is deployed. In particular, the **-l last** option initializes the weights from the latest saved checkpoint, while **-l best** option initializes the weights from the bast saved checkpoint (measured based on the validation accuracy). e.g.

`> python gqa_interpreter_experiments.py path/to/config/yaml -s 0 -l last`

Once the training is over, the test loop is performed on the specified test split.

## Testing

The test loop can be invoked independept of training by deploying the **-t** option:

`> python gqa_interpreter_experiments.py path/to/config/yaml -s 0 -l best -t`

## Prediction

Furthermore, a trained model can be run to produce a prediction JSON file for the specified test split by deploying the **-p** option: 

`> python gqa_interpreter_experiments.py path/to/config/yaml -s 0 -l best -t -p`

## Visualization

**&#8711;-FOL** is an extremly interpretable VQA framework. The whole question answering process can be visualized hop-by-hop for the specified test split by deploying the **-v** option:

`> python gqa_interpreter_experiments.py path/to/config/yaml -s 0 -l best -t -v`

# Main Contributors

+ [Saeed Amizadeh](mailto:saamizad@microsoft.com), Microsoft Inc.
+ [Alex Polozov](mailto:Alex.Polozov@microsoft.com), Microsoft Inc.
+ [Hamid Palangi](mailto:hpalangi@microsoft.com), Microsoft Inc.

# Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

# Trademarks 

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow Microsoft's Trademark & Brand Guidelines. Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship. Any use of third-party trademarks or logos are subject to those third-party's policies.