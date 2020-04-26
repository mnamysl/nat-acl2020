from torch.optim.sgd import SGD
import os.path
import sys, csv, random, logging
import numpy as np

FIXED_RANDOM_SEEDS = False

if FIXED_RANDOM_SEEDS:
    random.seed(0)
    np.random.seed(0)

EXIT_SUCCESS=0
EXIT_FAILURE=-1

def evaluate(model_path, corpus, mini_batch_size=256, misspelling_rate=0.0, 
             cmx_file="", typos_file="", spell_check = None):
    """
    Evaluates the model on the test set of the given corpus. 
    Appends the results to the eval.txt file in the model's directory.

    Parameters:    
        model_path (str): path to the model to be evaluated
        corpus (ColumnCorpus): loaded corpus
        mini_batch_size (int): size of batches used by the evaluation function
        misspelling_rate (float): misspelling rate (used in case of 'random' misspelling mode)
        cmx_file (str): confusion matrix file (used in case of 'confusion matrix' misspelling mode)
        typos_file (str): file with typos (used in case of 'typos' misspelling mode)
        spell_check (HunSpell): spell checking module (optional)
    
    """
    from robust_ner.enums import EvalMode, MisspellingMode

    if cmx_file:
        eval_mode = EvalMode.Misspellings
        misspell_mode = MisspellingMode.ConfusionMatrixBased
    elif typos_file:
        eval_mode = EvalMode.Misspellings
        misspell_mode = MisspellingMode.Typos
    elif misspelling_rate > 0.0:
        eval_mode = EvalMode.Misspellings
        misspell_mode = MisspellingMode.Random    
    else:
        eval_mode = EvalMode.Standard
        misspell_mode = MisspellingMode.Random

    # load the tagger model 
    from flair_ext.models import NATSequenceTagger
    tagger = NATSequenceTagger.load(model_path)
  
    eval_data = corpus.test

    from robust_ner.noise import make_char_vocab
    from robust_ner.confusion_matrix import load_confusion_matrix, filter_cmx
    from robust_ner.typos import load_typos
    
    char_vocab = make_char_vocab(eval_data)

    cmx, lut, typos = None, {}, {}

    # initialize resources used for evaluation
    if misspell_mode == MisspellingMode.ConfusionMatrixBased:
        cmx, lut = load_confusion_matrix(cmx_file)
        cmx, lut = filter_cmx(cmx, lut, char_vocab)
    elif misspell_mode == MisspellingMode.Typos:
        typos = load_typos(typos_file, char_vocab, False)        

    # fixed parameters
    num_workers = 8

    # evaluate the model
    result, loss = tagger.evaluate(eval_data, mini_batch_size, num_workers=num_workers,
        eval_mode=eval_mode, misspell_mode=misspell_mode, misspelling_rate=misspelling_rate,  
        char_vocab=char_vocab, cmx=cmx, lut=lut, typos=typos, spell_check=spell_check)

    # append the evaluation results to a file
    model_dir = os.path.dirname(model_path)
    eval_txt = os.path.join(model_dir, "eval.txt")

    with open(eval_txt, "a") as f:
        
        f.write(f"eval_mode: {eval_mode}\n")
        f.write(f"spell_checking: {spell_check != None}\n")

        if eval_mode == EvalMode.Misspellings:
            f.write(f"misspell_mode: {misspell_mode}\n")            
            if misspell_mode == MisspellingMode.Random:
                f.write(f"misspelling_rate: {misspelling_rate}\n")
            elif misspell_mode == MisspellingMode.ConfusionMatrixBased:
                f.write(f"cmx_file: {cmx_file}\n")
            elif misspell_mode == MisspellingMode.Typos:
                f.write(f"typos_file: {typos_file}\n")

        f.write(f"Loss: {loss:.6} {result.detailed_results}\n")
        f.write("-" * 100 + "\n")

    log.info("'{}' function finished!".format(sys._getframe().f_code.co_name))

def train_tagger(model_dir, corpus, corpus_name, tag_type, embedding_type, train_mode, alpha=1.0, 
    misspelling_rate=0.0, cmx_file="", num_hidden=256, learning_rate=0.1, mini_batch_size=32,  
    max_epochs=100, train_with_dev=False, checkpoint=False, valid_with_misspellings=True):
    """
    Trains a tagger model from scratch.

    Parameters:
        model_dir (str): output model path
        corpus (ColumnCorpus): loaded corpus
        corpus_name (str): name of the corpus used to load proper embeddings
        tag_type (str): type of the tag to train
        embedding_type (str): type of embeddings (e.g. flair, elmo, bert, word+char)
        train_mode (TrainingMode): training mode
        alpha (float): auxiliary loss weighting factor
        misspelling_rate (float): misspelling rate (used in case of 'random' misspelling mode)    
        cmx_file (float): a confusion matrix file (used in case of 'confusion matrix' misspelling mode)
        num_hidden (int): number of hidden layers of the tagger's LSTM 
        learning_rate (float): initial learning rate
        mini_batch_size (int): the size of batches used by the evaluation function
        max_epochs (int): maximum number of epochs to run
        train_with_dev (bool): train using the development set
        checkpoint (bool): save checkpoint files
        valid_with_misspellings (bool): use validation with misspelling as additional measure
    """
    
    # load embeddings
    embeddings, embeddings_in_memory = init_embeddings(corpus_name, embedding_type=embedding_type)
        
    # fixed parameters
    use_crf = True
    rnn_layers = 1
    dropout, word_dropout, locked_dropout = 0.0, 0.05, 0.5
    optimizer = SGD

    # create the tagger model
    from flair_ext.models import NATSequenceTagger
    tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
    tagger: NATSequenceTagger = NATSequenceTagger(hidden_size=num_hidden, embeddings=embeddings,
        tag_dictionary=tag_dictionary, tag_type=tag_type, use_crf=use_crf, use_rnn=rnn_layers>0, 
        rnn_layers=rnn_layers, dropout=dropout, word_dropout=word_dropout, locked_dropout=locked_dropout,
        train_mode=train_mode, alpha=alpha, misspelling_rate=misspelling_rate, cmx_file=cmx_file)
        
    # fixed parameters
    anneal_factor = 0.5
    patience = 3
    anneal_with_restarts = False
    num_workers = 8
    
    # train the model
    from flair_ext.trainers import ParameterizedModelTrainer     
    trainer: ParameterizedModelTrainer = ParameterizedModelTrainer(tagger, corpus, optimizer=optimizer, epoch=0, loss=10000.0)
    trainer.train(model_dir, learning_rate=learning_rate, mini_batch_size=mini_batch_size, max_epochs=max_epochs,
        anneal_factor=anneal_factor, patience=patience, train_with_dev=train_with_dev, monitor_train=False, 
        embeddings_in_memory=embeddings_in_memory, checkpoint=checkpoint, anneal_with_restarts=anneal_with_restarts, 
        shuffle=True, param_selection_mode=False, num_workers=num_workers, valid_with_misspellings=valid_with_misspellings)
    
    plot_training_curves(model_dir)

    log.info("'{}' function finished!".format(sys._getframe().f_code.co_name))

def fine_tune(model_dir, corpus, checkpoint_name, train_mode, alpha=1.0, 
    misspelling_rate = 0.0, cmx_file="", learning_rate=0.01, mini_batch_size=32, max_epochs=100, 
    train_with_dev=False, checkpoint=True, valid_with_misspellings=True):
    """
    Fine-tunes an existing tagger model.

    Parameters:
        model_dir (str): output model path
        corpus (str): loaded corpus
        checkpoint_name (str): name of the checkpoint file
        train_mode (TrainingMode): training mode
        alpha (float): auxiliary loss weighting factor
        misspelling_rate (float): misspelling rate (used in case of 'random' misspelling mode)    
        cmx_file (str): a confusion matrix file (used in case of 'confusion matrix' misspelling mode)    
        learning_rate (float): initial learning rate
        mini_batch_size (int): the size of batches used by the evaluation function
        max_epochs (int): maximum number of epochs to run
        train_with_dev (bool): train using the development set
        checkpoint (bool): save checkpoint files
        valid_with_misspellings (bool): use validation with misspelling as additional measure
    """
    
    checkpoint_path = os.path.join(model_dir, checkpoint_name)
    
    # https://github.com/zalandoresearch/flair/issues/770
    # from flair.models import NATSequenceTagger
    from flair_ext.models import NATSequenceTagger

    # load checkpoint file
    checkpoint = NATSequenceTagger.load_checkpoint(checkpoint_path)
    checkpoint['epoch'] = 0
    checkpoint['model'].set_training_params(train_mode=train_mode, alpha=alpha, misspelling_rate=misspelling_rate, cmx_file=cmx_file)
    
    # fixed parameters
    optimizer = SGD
    anneal_factor = 0.5
    patience = 3
    anneal_with_restarts = False
    num_workers = 8

    # train the model
    from flair_ext.trainers import ParameterizedModelTrainer        
    trainer: ParameterizedModelTrainer = ParameterizedModelTrainer.load_from_checkpoint(checkpoint, corpus, optimizer=optimizer)
    
    trainer.train(model_dir, learning_rate=learning_rate, mini_batch_size=mini_batch_size, max_epochs=max_epochs,
        anneal_factor=anneal_factor, patience=patience, train_with_dev=train_with_dev, monitor_train=False, 
        embeddings_in_memory=True, checkpoint=checkpoint, anneal_with_restarts=anneal_with_restarts, 
        shuffle=True, param_selection_mode=False, num_workers=num_workers, valid_with_misspellings=valid_with_misspellings)

    plot_training_curves(model_dir)

    log.info("'{}' function finished!".format(sys._getframe().f_code.co_name))
    
def init_embeddings(corpus_name, embedding_type):
    """
    Initializes embeddings for a given corpus.

    Parameters:
        corpus_name (str): name of the corpus used to load proper embeddings
        embedding_type (str): type of embeddings (e.g. flair, elmo, bert, word+char)
    
    Returns:
        tuple(StackedEmbeddings, bool): loaded embeddings
    """
    
    from typing import List    
    from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings
    from flair.embeddings import FlairEmbeddings
    from flair.embeddings import BertEmbeddings, ELMoEmbeddings
    from flair.embeddings import WordEmbeddings, CharacterEmbeddings

    embedding_types: List[TokenEmbeddings] = []
    
    if corpus_name in ['conll03_en']:
        if embedding_type == 'flair':
            embedding_types.append(WordEmbeddings('glove'))
            embedding_types.append(FlairEmbeddings('news-forward'))
            embedding_types.append(FlairEmbeddings('news-backward'))
            embeddings_in_memory = True 
        elif embedding_type == 'bert':            
            embedding_types.append(BertEmbeddings(bert_model_or_path='bert-base-cased'))            
            #embedding_types.append(BertEmbeddings(bert_model_or_path='bert-large-cased'))
            embeddings_in_memory = True
        elif embedding_type == 'elmo':
            embedding_types.append(ELMoEmbeddings())
            embeddings_in_memory = True        
        elif embedding_type == 'word+char':
            # similar to Lample et al. (2016)
            embedding_types.append(WordEmbeddings('glove'))
            embedding_types.append(CharacterEmbeddings())
            embeddings_in_memory = False # because it contains a char model (problem with deepcopy)
        else:
            log.error(f"no settings for '{embedding_type}'!")
            exit(EXIT_FAILURE)

    elif corpus_name in ["conll03_de", "germeval"]:
        if embedding_type == 'flair':
            embedding_types.append(WordEmbeddings('de'))
            embedding_types.append(FlairEmbeddings('german-forward'))
            embedding_types.append(FlairEmbeddings('german-backward'))
            embeddings_in_memory = True
        elif embedding_type == 'word+char':
            # similar to Lample et al. (2016)
            embedding_types.append(WordEmbeddings('de'))
            embedding_types.append(CharacterEmbeddings())
            embeddings_in_memory = False # because it contains a char model (problem with deepcopy)
        else:
            log.error(f"no settings for '{embedding_type}'!")
            exit(EXIT_FAILURE)
    else:
        log.error(f"unknown corpus or embeddings '{corpus_name}'!")
        exit(EXIT_FAILURE)
        
    embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)
    
    log.info("'{}' function finished!".format(sys._getframe().f_code.co_name))

    return embeddings, embeddings_in_memory

def load_corpus(corpus_name, col_idx, text_idx, tag_type='ner', downsample_perc=1.0, 
                name_train=None, name_dev=None, name_test=None, verbose=False):
    """
    Loads a corpus with a given name.
    Optionally performs downsampling of the data.

    Parameters:
        corpus_name (str): name of the corpus used to load proper embeddings
        col_idx (int): index of the column's tag
        text_idx (int): index of the text's tag
        tag_type (str): type of the tag to load
        downsample_rate (float): downsample rate (1.0 = full corpus)
        name_train (str): name of a file containing the train set
        name_dev (str): name of a file containing the development set
        name_test (str): name of a file containing the test set
    
    Returns:
        ColumnCorpus: the loaded corpus
    """        
    
    from pathlib import Path

    data_dir = f'resources/tasks/'
        
    if corpus_name in ["conll03_en"]:
        from flair.datasets import CONLL_03
        corpus = CONLL_03(base_path=Path(data_dir), tag_to_bioes=tag_type)
    elif corpus_name in ["conll03_de"]:
        from flair.datasets import CONLL_03_GERMAN
        corpus = CONLL_03_GERMAN(base_path=Path(data_dir), tag_to_bioes=tag_type)
    elif corpus_name in ["germeval"]:
        from flair.datasets import GERMEVAL
        corpus = GERMEVAL(base_path=Path(data_dir), tag_to_bioes=tag_type)    
    else:        
        corpus_dir = f"{data_dir}{corpus_name}"
        if not os.path.exists(corpus_dir):
            log.error(f"Data directory '{corpus_dir}' does not exists!")
            exit(EXIT_FAILURE)

        from flair.datasets import ColumnCorpus
        
        columns = { text_idx: 'text', col_idx: tag_type }
        train_set = None if name_train is None else f'{name_train}'
        dev_set = None if name_dev is None else f'{name_dev}'
        test_set = None if name_test is None else f'{name_test}'
        
        corpus: ColumnCorpus = ColumnCorpus(corpus_dir, columns, train_file=train_set, test_file=test_set, dev_file=dev_set,
            tag_to_bioes=tag_type)

    if downsample_perc >= 0.0 and downsample_perc < 1.0:
        corpus.downsample(downsample_perc)
    
    if verbose:
        log.info(corpus.obtain_statistics(tag_type=tag_type))
                
    log.info("'{}' function finished!".format(sys._getframe().f_code.co_name))

    return corpus

def plot_training_curves(model_dir):
    """
    Plots training curves given the model directory.

    Parameters:
        model_dir (str): model's directory
    """
    
    from flair_ext.visual.training_curves import Plotter
    plotter = Plotter()
    plotter.plot_training_curves('{}/loss.tsv'.format(model_dir))
    plotter.plot_weights('{}/weights.txt'.format(model_dir))

    log.info("'{}' function finished!".format(sys._getframe().f_code.co_name))

def parse_args():
    """
    Parses command-line arguments.
    
    Returns:
        parsed arguments
    """
        
    import argparse
    from robust_ner.enums import TrainingMode

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--mode', dest='mode', type=str, help="execution mode",
        choices=['train', 'tune', 'eval'], default='', required=True)
    parser.add_argument('--corpus', dest='corpus', type=str, help="data set to use", default='', required=True)
    parser.add_argument('--type', dest='embedding_type', type=str, help="embedding type",
        choices=['flair', 'bert', 'word+char', 'elmo'], default='flair')
    parser.add_argument('--model', dest='model', type=str, help="model path", default='', required=True)
    parser.add_argument('--col_idx', dest='col_idx', type=int, help="ner tag column index", default=3)
    parser.add_argument('--text_idx', dest='text_idx', type=int, help="text tag column index", default=0)
    parser.add_argument('--device', dest='device', type=str, help="device to use", default='cuda')
    parser.add_argument('--downsample', dest='downsample', type=float, help="downsample rate", default='1.0')
    parser.add_argument('--checkpoint', dest='checkpoint', type=str, help="checkpoint file", default='best-model.pt')
    parser.add_argument('--alpha', dest='alpha', type=float, help="auxiliary loss weight factor", default=1.0)
    parser.add_argument('--misspelling_rate', dest='misspelling_rate', type=float, 
        help="misspellings rate used during training", default=0.0)
    parser.add_argument('--train_mode', dest='train_mode', type=TrainingMode, help="training mode", 
        choices=list(TrainingMode), default=TrainingMode.Standard)
    parser.add_argument('--verbose', dest='verbose', action='store_true', help="print verbose messages", default=False)
    parser.add_argument('--num_hidden', dest='num_hidden', type=int, help="the number of hidden units of a tagger LSTM", 
        default=256)
    parser.add_argument('--max_epochs', dest='max_epochs', type=int, help="max number of epochs to train", default=100)
    parser.add_argument('--batch_size', dest='batch_size', type=int, help="mini batch size", default=32)
    parser.add_argument('--lr', dest='learning_rate', type=float, help="initial learning rate", default=0.1)
    parser.add_argument('--train_with_dev', dest='train_with_dev', action='store_true', 
        help="train using development data set", default=False)
    parser.add_argument('--cmx_file', dest='cmx_file', type=str, help="confusion matrix file for training or evaluation", 
        default='')        
    parser.add_argument('--typos_file', dest='typos_file', type=str, help="typos file for evaluation", default='')        
    parser.add_argument('--spell_check', dest='spell_check', action='store_true', 
        help="use hunspell to automaticaly correct misspellings", default=False)
    parser.add_argument('--no_valid_misspell', dest='no_valid_with_misspellings', action='store_true', 
        help="turns off the validation component that uses misspellings", default=False)

    args = parser.parse_args()

    log.info(args)
    
    if args.device not in ['cpu', 'cuda', 'mkldnn', 'opengl', 'opencl', 'ideep', 'hip', 'msnpu']:
        log.error("unknown args.device: '{}'".format(args.device))
        exit(EXIT_FAILURE)    
           
    import torch, flair 

    if FIXED_RANDOM_SEEDS:
        torch.manual_seed(0)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        args.device = "cpu"

    flair.device = torch.device(args.device)
    
    if args.col_idx < 0:
        log.error("invalid args.col_idx: '{}'".format(args.col_idx))
        exit(EXIT_FAILURE)
    
    if not 0.0 < args.downsample <= 1.0:
        log.error("invalud args.downsample: '{}'".format(args.downsample))
        exit(EXIT_FAILURE)

    if len(args.corpus) == 0:
        log.error("invalid args.corpus: '{}'".format(args.corpus))
        exit(EXIT_FAILURE)
        
    log.info("'{}' function finished!".format(sys._getframe().f_code.co_name))
            
    return args

if __name__ == "__main__":

    logging.basicConfig()        
    logging.getLogger(__name__).setLevel(logging.INFO)    
    log = logging.getLogger(__name__)

    current_directory = os.path.dirname(os.path.abspath(__file__))

    # add the current directory to the system path to use functions from the robust_ner library
    sys.path.append(current_directory)

    # add sub-folder containint the flair library to the system path
    sys.path.append(os.path.join(current_directory, "flair"))
    
    # parse command-line arguments
    args = parse_args()    
                                   
    model_name = args.model

    # if the model_name is not an absolute path - assume it is placed in the 'resources/taggers' sub-directory
    if os.path.isabs(model_name):
        model_dir = model_name
    else:
        model_dir = os.path.join("resources/taggers", model_name) 
    
    # join the full model path
    model_path = os.path.join(model_dir, args.checkpoint)

    # if the given path does not exists, check whether it could be a built-in model
    if not os.path.exists(model_path) and model_name in ['ner', 'de-ner']:
        model_path = model_name
    
    # load the corpus
    tag_type = 'ner'
    corpus = load_corpus(args.corpus, args.col_idx, args.text_idx, tag_type, args.downsample, verbose=args.verbose)
    
    # optionaly, initialize the spell checker
    if args.spell_check:
        from robust_ner.spellcheck import init_spellchecker
        spell_check = init_spellchecker(args.corpus)        
    else:
        spell_check = None

    print(f"Using '{spell_check}' spell checker")

    if args.mode == 'train':
        train_tagger(model_dir, corpus, args.corpus, tag_type, embedding_type=args.embedding_type,            
            train_mode=args.train_mode, alpha=args.alpha, misspelling_rate=args.misspelling_rate, 
            cmx_file=args.cmx_file, num_hidden=args.num_hidden, max_epochs=args.max_epochs, 
            learning_rate=args.learning_rate, train_with_dev=args.train_with_dev, mini_batch_size=args.batch_size,
            valid_with_misspellings=not args.no_valid_with_misspellings)
    elif args.mode == 'tune':
        fine_tune(model_dir, corpus, args.checkpoint, train_mode=args.train_mode, alpha=args.alpha,  
            misspelling_rate=args.misspelling_rate, max_epochs=args.max_epochs, cmx_file=args.cmx_file,
            learning_rate=args.learning_rate, train_with_dev=args.train_with_dev, mini_batch_size=args.batch_size,
            valid_with_misspellings=not args.no_valid_with_misspellings)    
    elif args.mode == 'eval':
        evaluate(model_path, corpus, misspelling_rate=args.misspelling_rate, cmx_file=args.cmx_file, typos_file=args.typos_file, 
            spell_check=spell_check)
    else:
        print("unknown mode")
        exit(EXIT_FAILURE)
