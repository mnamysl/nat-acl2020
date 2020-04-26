# Noise-Aware Training (NAT)

This repository contains the code and the data that can be used to reproduce the experiments described in the "**NAT: Noise-Aware Training for Robust Neural Sequence Labeling**" paper, which was accepted to be presented the [ACL 2020](https://acl2020.org/) conference. 

## Background

The standard sequence labeling systems are usually trained on clean text. Such systems exhibit substantially lower accuracy when applied on imperfect textual input. NAT aims to improve robustness of sequence labeling performed on data from noisy sources, like Optical Character Recognition ([OCR](https://en.wikipedia.org/wiki/Optical_character_recognition)), Automatic Speech Recognition ([ASR](https://en.wikipedia.org/wiki/Speech_recognition)) or misspelled user-generated text. NAT uses a standard sequence labeling architecture, but modifies the training objective of the neural model. To this end, it employs two auxiliary training objectives:

1. *Data augmentation* objective, which directly induces noise in the input data and trains the model on a mixture of clean and noisy samples.
2. *Stability training* objective that encourages similarity between the original and the perturbed input, which helps the model to produce a noise-invariant latent representation.

NAT was implemented as an extension to the [FLAR](https://github.com/zalandoresearch/flair) library, but it can be integrated into any other sequence labeling framework. 

## Project Structure

```
├── flair
├── flair_ext
│   ├── models
│   ├── trainers
│   └── visual
├── resources
|   ├── cmx
|   ├── taggers
│   ├── tasks
|   └── typos
└── robust_ner
```

The [flair](./flair) directory includes the basic FLAIR framework. See the [Quick Start section](README.md#getting-the-code) for more information, how to get it.

The [flair_ext](./flair_ext) directory contains extensions to the basic FLAIR library:
* [An extended sequence labeling model](./flair_ext/models/nat_sequence_tagger_model.py), which implements both NAT objectives.
* [A modified trainer class](./flair_ext/trainers/trainer.py), which performs training using the extended sequence labeling model. 

The [robust_ner](./robust_ner) directory comprises of the modules that are used for noise induction, spelling correction and the helper function/classes.

The [resources](./resources) directory includes the data files. [Confusion matrices](./resources/cmx) are included in the project. See the Quick Start notes for more information, how to get the [typos](README.md#misspellings-typos) files and [the data sets](README.md#named-entity-recognition-ner-data-sets).

## Quick Start

### Getting the Code

1. Clone or download the NAT GitHub repository:

```
git clone https://github.com/mnamysl/nat-acl2020
```

2. Download the FLAIR framework (v0.4.2) from: https://github.com/zalandoresearch/flair/releases/tag/v0.4.2
3. Rename the extracted *flair-0.4.2* to *flair* and move it to the *NAT* directory. 

### Getting the data

#### Named Entity Recognition (NER) Data Sets

Please follow the instruction on the websites of the corresponding shared tasks:

* **CoNLL 2003**: https://www.clips.uantwerpen.be/conll2003/ner/

* **GermEval 2014**: https://sites.google.com/site/germeval2014ner/data/

#### Misspellings, typos

1. Download typos lists from the following websites:
* *Misspelling Oblivious Word Embeddings (MOE)*: https://github.com/facebookresearch/moe/tree/master/data (*moe_misspellings_train.tsv*)
* Typos released by *Belinkov & Bisk*: https://github.com/ybisk/charNMT-noise/tree/master/noise (*de.natural* and *en.natural* files).
2. Move the downloaded files to the *resources/tasks* sub-directory.

### Installing the Prerequisites

1. Please install all required packages as shown below:
```
pip install -r requirements.txt
```

2. To use the [ELMo embeddings](https://allennlp.org/elmo), you need to install the [AllenNLP](https://github.com/allenai/allennlp) library:
```
pip install allennlp
```

3. If you plan to use the spell checking functionality, you need to install the packages required to run [hunspell](http://hunspell.github.io/):
```
sudo apt-get install hunspell hunspell-de-de hunspell-en-us libhunspell-dev python-dev
pip install hunspell
```

## Using the code

You can use the NAT functionality by calling the [main.py](./main.py) python script. The following command-line parameters can be specified (in the order of importance; parameters in bold are required):

| Parameter           | Description              | Value                                                      |
| ------------------- | ------------------------ | ---------------------------------------------------------- |
| **--mode**          | Execution mode           | One of: *train*, *tune*, *eval*.                           |
| **--corpus**        | Data set to use          | One of: *conll03_en* (default), *conll03_de*, *germeval*.  |
| **--model**         | Model name               | Arbitrary string.                                          |
| --train_mode        | Training mode            | One of: *standard* (default), *augmentation*, *stability*. |
| --alpha             | Auxiliary loss weight    | Floating point number (default: 1.0).                      |
| --misspelling_rate  | Noise level              | Floating point number (default: 0.0).                      |
| --type              | Type of embeddings       | One of: *flair* (default), *bert*, *elmo*, *word+char*.    |
| --cmx_file          | Confusion matrix file    | e.g.: *tesseract3-RS*.                                     |
| --typos_file        | Typos file               | e.g.: *en.natural* or *moe_misspellings_train.tsv*.        |
| --spell_check       | Use spell checking       | no parameters, turned off by default.                      |
| --lr                | Initial learning rate    | Floating point number (default: 0.1).                      |
| --train_with_dev    | Train with dev. set      | no parameters, turned off by default.                      |
| --col_idx           | Index of a tag column    | Integer value (default: 3).                                |
| --text_idx          | Index of the text column | Integer value (default: 0).                                |
| --device            | Device to use            | Torch device type string (default: cuda).                  |
| --downsample        | Downsample rate          | Floating point value (default: 1.0).                       |
| --num_hidden        | Tagger hidden state size | Integer value (default: 256).                              |
| --max_epochs        | Max. training epochs     | Integer value (default: 100).                              |
| --batch_size        | Mini batch size          | Integer value (default: 32).                               |
| --checkpoint        | Checkpoint file name     | String (default: best-model.pt).                           |
| --no_valid_misspell | No validation with misspellings | No parameters, turned off by default.               |
| --verbose           | Print verbose messages   | No parameters, turned off by default.                      |
| -h                  | Print help               | No parameters.                                             |

### Training a model from scratch

The following call will start the training of a new model called *my_new_model* on the English CoNLL 2003 data set using the data augmentation objective with the weight factor of 1.0 and the noise level of 10%.
```
python3 main.py --mode train --corpus conll03_en --model my_new_model --train_mode augmentation --misspelling_rate 0.1 --alpha 1.0
```

All your models will be stored in the *resources/taggers* directory.

### Fine-tuning an existing model

The following call will start the fine-tuning process of the previously trained model using the stability objective with different parameters and a lower learning rate:
```
python3 main.py --mode tune --corpus conll03_en --model my_trained_model --train_mode stability --misspelling_rate 0.05 --alpha 0.5 --lr 0.01
```

### Evaluation

Finally, the prepared model can be evaluated on the real OCR erros by using the following call:

```
python3 main.py --mode eval --corpus conll03_en --model my_trained_model --cmx_file tesseract3-RS
```

## Authors

* Marcin Namysl

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
