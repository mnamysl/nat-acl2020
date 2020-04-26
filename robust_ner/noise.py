import math
import logging
import random
import numpy as np

from robust_ner.confusion_matrix import noise_sentences_cmx
from robust_ner.vanilla_noise import noise_sentences_vanilla
from robust_ner.typos import noise_sentences_typos
from robust_ner.enums import MisspellingMode


def make_char_vocab(sentences):
    """
    Construct the character vocabulary from the given sentences
    """

    char_vocab = set()  

    for sentence in sentences:
        _update_char_vocab(sentence, char_vocab)
    
    return char_vocab

def _update_char_vocab(sentence, char_vocab: set):
    """
    Updates the character vocabulary using a single sentence
    """

    for token in sentence:     

        if len(token.text) > 0:
            char_vocab.update([s for s in set(token.text) if not s.isspace()])

def noise_sentences(sentences, misspell_mode, noise_level = 0.0, char_vocab = {}, cmx = None, lut = {}, typos = {}, verbose: bool = False):
    """
    Induces noise on the given list of sentences
    """

    if misspell_mode == MisspellingMode.ConfusionMatrixBased:
        return noise_sentences_cmx(sentences, cmx, lut)
    elif misspell_mode == MisspellingMode.Typos:
        return noise_sentences_typos(sentences, typos, noise_level)
    else:
        return noise_sentences_vanilla(sentences, char_vocab, noise_level, verbose)
        

