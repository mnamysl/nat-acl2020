import math
import logging
import random
import numpy as np

from robust_ner.confusion_matrix import make_lut_from_vocab


def induce_noise_vanilla(input_text, char_vocab, noise_level):
    """
    Induces noise into the input text using a vanilla noise model.
    """

    log = logging.getLogger("flair")

    vocab = list(char_vocab)
    vocab.insert(len(vocab), "NULL")
    # print(f"vocab={vocab}")
    lut = make_lut_from_vocab(vocab)
    
    n_classes = len(lut)

    input_chars, output_chars = list(input_text), []
    cnt_modifications, cnt_subst, cnt_ins, cnt_del = 0, 0, 0, 0

    cnt_chars = len(input_chars)    
    weight_ins = cnt_chars / (cnt_chars + 1)
    prob_change = noise_level / 3

    row_idx_null = lut.get('NULL', -1)

    # _i_t_e_m_
    # 012345678
    for i in range(cnt_chars * 2 + 1):

        input_char = input_chars[i // 2] if (i % 2 == 1) else 'NULL'        
        row_idx = lut.get(input_char, -1)
        
        result_char = input_char

        if row_idx >= 0:

            # P(no_change) = 1.0 - noise_level
            # P(change) = noise_level, spreaded across all elements, except the 'no_change' element

            if input_char == 'NULL':
                prob_insert = prob_change * weight_ins
                prob = np.full((n_classes), prob_insert / (n_classes - 1))
                prob[row_idx] = 1 - prob_insert # no-change
            else:
                prob = np.full((n_classes), prob_change / (n_classes - 2))                
                prob[row_idx_null] = prob_change # prob-delete
                prob[row_idx] = 1 - 2 * prob_change # no-change

            prob_sum = prob.sum()
            if math.isclose(prob_sum, 1.0): 
                rand_idx = np.random.choice(n_classes, p=prob)
                result_char = vocab[rand_idx]
            else:
                log.warning(f"Probabilities do not sum to 1 ({prob_sum}) for row_idx={row_idx} (input_char={input_char})!")    
        else:
            log.warning(f"LUT key for '{input_char}' does not exists!")            

        # print(f"{input_char} -> {result_char}")

        if result_char == 'NULL' and result_char != input_char:
            cnt_del += 1
        elif input_char == 'NULL' and result_char != input_char:
            cnt_ins += 1
        elif input_char != result_char:
            cnt_subst += 1

        if result_char != 'NULL':
            output_chars.append(result_char)

        if input_char != result_char:
            cnt_modifications += 1

    output_text = "".join(output_chars)

    if len(output_text) == 0:
        output_text = input_text

    return output_text, cnt_modifications, cnt_subst, cnt_ins, cnt_del

def _noise_sentences_vanilla_verbose(sentences, char_vocab, noise_level):
    """
    Induces noise on the given list of sentences with verbose logging
    """
    
    log = logging.getLogger("flair")

    from copy import deepcopy
    noised_sentences = deepcopy(sentences)
        
    cnt_chars = sum([len(token.text) for sentence in sentences for token in sentence], 0)
    cnt_tokens = sum([len(sentence.tokens) for sentence in sentences], 0)
    cnt_sentences = len(sentences)
    cnt_char_modifications, cnt_token_modifications, cnt_sent_modifications = 0, 0, 0
                
    for sentence in noised_sentences:          
        sent_modified = False          
        
        for token in sentence:                
            noised_text, cnt_modifications, _, _, _ = induce_noise_vanilla(token.text, char_vocab, noise_level)                
            
            # if verbose and cnt_modifications > 0:                    
            #     log.info("{0} -> {1} (cnt_modif: {2})".format(token.text, noised_text, cnt_modifications))
            
            token.text = noised_text
            
            if cnt_modifications > 0:
                cnt_char_modifications += cnt_modifications
                cnt_token_modifications += 1
                sent_modified = True

        if sent_modified:
            cnt_sent_modifications += 1
    
    SER = cnt_sent_modifications * 100.0 / cnt_sentences
    TER = cnt_token_modifications * 100.0 / cnt_tokens
    CER = cnt_char_modifications * 100.0 / cnt_chars
    log.info(
        f"SER:{SER:.1f}({cnt_sent_modifications}/{cnt_sentences}), "
        f"TER:{TER:.1f}({cnt_token_modifications}/{cnt_tokens}), "
        f"CER:{CER:.1f}({cnt_char_modifications}/{cnt_chars})")

    # for i, sentence in enumerate(sentences):
    #     modified_sentence = noised_sentences[i]
    #     print("{} -> {}".format(sentence, modified_sentence))

    return noised_sentences, cnt_token_modifications

def _noise_sentences_vanilla_quiet(sentences, char_vocab, noise_level):
    """
    Induces noise on the given list of sentences without verbose logging
    """

    from copy import deepcopy
    noised_sentences = deepcopy(sentences)
    
    cnt_noised_tokens = 0
    for sentence in noised_sentences:            
        for token in sentence:
            token.text, cnt_modif, _, _, _ = induce_noise_vanilla(token.text, char_vocab, noise_level)    
            if cnt_modif > 0:
               cnt_noised_tokens += 1

    return noised_sentences, cnt_noised_tokens

def noise_sentences_vanilla(sentences, char_vocab, noise_level, verbose: bool):
    """
    Induces noise on the given list of sentences using the vanilla noise model
    """

    if verbose:
        return _noise_sentences_vanilla_verbose(sentences, char_vocab, noise_level)
    else:
        return _noise_sentences_vanilla_quiet(sentences, char_vocab, noise_level)
