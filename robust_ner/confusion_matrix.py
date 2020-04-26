import os.path
import csv
import math
import logging
import random
import numpy as np


def load_confusion_matrix(cmx_file_name, separator=' '):
    """
    Loads a confusion matrix from a given file. 

    NULL - token that represents the epsilon character used to define.
    the deletion and insertion operations.
    WS - white-space character.
    Rows represent original characters, column - perturbed characters.
    Deletion of a character - NULL token in a column header (original->NULL)
    Insertion of a character - NULL token in a row header (NULL->result)
    ConfusionMatrix[NULL][NULL] == 0.0
    File format:
        - 1st row: vocabulary, e.g., VOCAB a b c ... x y z
        - next |V| rows: rows of the confusion matrix, where V is vocabulary of characters
    """
    log = logging.getLogger("flair")

    # read input file (e.g., ocr.cmx)
    source_path = os.path.join(f"resources/cmx/", f"{cmx_file_name}.cmx")
    log.info(f"Confusion matrix path: {source_path}")
       
    vocab = None
    cmx = None # confusion matrix    

    with open(source_path, "r") as input_file:
        reader = csv.reader(input_file, delimiter=separator)        

        for row in reader:                
            if len(row) == 0 or row[0].startswith('#'):
                continue            
            # print(row)
            
            if vocab is None:                
                row = [c[1:-1] for c in row if len(c) > 0] # strip first and last character
                vocab = row
            else:
                row = [c for c in row if len(c) > 0]
                cmx = np.array(row) if cmx is None else np.vstack([cmx, row])

    cmx = cmx.astype(np.float)

    lut = make_lut_from_vocab(vocab)

    # remove rows and columns of some characters (e.g., WS)
    to_delete = [c for c in ['WS'] if c in lut]
    for c in to_delete:
        cmx, lut, vocab = _remove_char_from_cmx(c, cmx, lut, vocab)        
    
    # np.set_printoptions(precision=4, floatmode='fixed', suppress=True)
    # log.info(f"Vocabulary:\n{vocab}")
    # log.info(f"LUT:\n{lut}")
    # log.info(f"Confusion matrix:\n{cmx}")    
    # log.info(f"p(c -> b): {query_cmx('c', 'b', cmx, lut)}") 
    # log.info(f"p(NULL -> c): {query_cmx('NULL','c', cmx, lut)}")
        
    cmx = _normalize_cmx(cmx)

    return cmx, lut

def print_cmx(print_func, cmx, precision=2):  
    """
    Prints the confusion matrix with a given print function
    """

    np.set_printoptions(precision=precision, floatmode='fixed', suppress=True)    
    print_func(cmx)

def query_cmx(orig_char, result_char, cmx, lut):
    """
    Helper function for querying a value from the confusion matrix
    """

    return cmx[lut[orig_char], lut[result_char]]
    #return cmx[vocab.index('c'), vocab.index('b')]

def _normalize_cmx(cmx):
    """
    Normalizes the rows of the confusion matrix, so that they form
    valid probability distributions.
    Assigns zero probability if all elements in a row are zero.
    """

    cmx_row_sums = cmx.sum(axis=1)[:, np.newaxis]
    cmx = np.divide(cmx, cmx_row_sums, out=np.zeros_like(cmx), where=cmx_row_sums!=0)
    return cmx

def filter_cmx(cmx, lut, corpus_vocab):
    """
    Removes from the confusion matrix all characters that do not appear in the corpus
    """
    log = logging.getLogger("flair")

    # re-create vocabulary from LUT
    cmx_vocab = make_vocab_from_lut(lut)
    to_delete = [c for c in cmx_vocab if c not in corpus_vocab and c not in ['NULL']]

    log.info(f"Characters to delete from the confusion matrix: {to_delete}")

    # remove rows and columns of confusion matrix that do not appear in a given vocabulary
    for c in to_delete:
        cmx, lut, cmx_vocab =_remove_char_from_cmx(c, cmx, lut, cmx_vocab)

    cmx = _normalize_cmx(cmx)

    return cmx, lut

def _remove_char_from_cmx(c, cmx, lut, vocab):
    """
    Removes a given character from the confusion matrix
    """

    idx = lut.get(c, -1)    
    if idx >= 0:
        # log.info(f"'{c}' removed from the confusion matrix.")
        cmx = np.delete(cmx, (idx), axis=0) # delete row
        cmx = np.delete(cmx, (idx), axis=1) # delete column
        vocab.pop(idx)
        # lut.pop(c, None)
        # LUT must be re-calculated
        lut = make_lut_from_vocab(vocab)
    
    return cmx, lut, vocab

def make_vocab_from_lut(lut):
    return [k for k,v in lut.items()]

def make_lut_from_vocab(vocab):
    return { c:i for i, c in enumerate(vocab) } 

def induce_noise_cmx(input_text, cmx, lut):
    """
    Induces noise into the input text using the confusion matrix
    """

    log = logging.getLogger("flair")

    # re-create vocabulary from LUT
    vocab = make_vocab_from_lut(lut)
    # print(f"vocab={vocab}")

    n_classes = len(lut)

    input_chars, output_chars = list(input_text), []
    cnt_modifications = 0

    # _i_t_e_m_
    # 012345678
    for i in range(len(input_chars) * 2 + 1):

        input_char = input_chars[i // 2] if (i % 2 == 1) else 'NULL'        
        row_idx = lut.get(input_char, -1)
        
        result_char = input_char

        if row_idx >= 0:
            prob = cmx[row_idx] 
            prob_sum = prob.sum()
            if math.isclose(prob_sum, 1.0): 
                rand_idx = np.random.choice(n_classes, p=prob)
                result_char = vocab[rand_idx]
            else:
                log.warning(f"Probabilities do not sum to 1 ({prob_sum}) for row_idx={row_idx} (input_char={input_char})!")    
        else:
            log.warning(f"LUT key for '{input_char}' does not exists!")            

        # print(f"{input_char} -> {result_char}")

        if result_char != 'NULL':
            output_chars.append(result_char)

        if input_char != result_char:
            # print(f"{input_char} -> {result_char}")
            cnt_modifications += 1

    output_text = "".join(output_chars)

    if len(output_text) == 0:
        output_text = input_text
        cnt_modifications = 0

    return output_text, cnt_modifications

def noise_sentences_cmx(sentences, cmx, lut):
    """
    Induces noise on the list of sentences using the confusion matrix
    """
    from copy import deepcopy
    noised_sentences = deepcopy(sentences)
    
    cnt_token_modifications = 0
    for sentence in noised_sentences:            
        for token in sentence:
            token.text, cnt_modif = induce_noise_cmx(token.text, cmx, lut)
            if cnt_modif > 0:
                cnt_token_modifications += 1

    return noised_sentences, cnt_token_modifications

