import hunspell

def init_spellchecker(corpus):
    """
    Initializes the spell checker.
    It uses the corpus information to choose a proper language for spell checker.
    Returns the initialied spell checker
    """

    if corpus in ["conll03_en", "ontonotes"]:
        spell_check = hunspell.HunSpell('/usr/share/hunspell/en_US.dic', '/usr/share/hunspell/en_US.aff')        
    elif corpus in ["conll03_de", "germeval"]:
        spell_check = hunspell.HunSpell('/usr/share/hunspell/de_DE.dic', '/usr/share/hunspell/de_DE.aff')
    else:
        spell_check = None

    return spell_check

def correct_text(spellcheck, input):
    """
    Checks whether the input is correctly spelled and correct it otherwise
    Returns the corrected input
    """
    output = input

    ok = spellcheck.spell(input)
    if not ok:
        suggestions = spellcheck.suggest(input)
        if len(suggestions) > 0:
            output = suggestions[0]
            # print(f"{input} -> {output}")

    return output

def correct_sentences(spellcheck, sentences):
    """
    Corrects all tokens in the given sentences using the given spell checker
    Returns the corrected sentences
    """

    from copy import deepcopy
    corrected_sentences = deepcopy(sentences)

    for sentence in corrected_sentences:            
        for token in sentence:
            token.text = correct_text(spellcheck, token.text)    

    return corrected_sentences