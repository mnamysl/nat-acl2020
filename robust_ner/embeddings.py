import logging
import torch

from typing import List
from flair.data import Sentence

log = logging.getLogger("flair")


def check_embeddings(sentList1: List[Sentence], sentList2: List[Sentence], embed1: torch.tensor, embed2: torch.tensor):
    """
    Checks embeddings of the original and perturbed sentences.
    Returns false if any token of the first sentence has the same embeddings but different text as the
    corresponding token of the second sentence
    """

    for i, (s1, s2) in enumerate(zip(sentList1, sentList2)):
        for j, (tok1, tok2) in enumerate(zip(s1, s2)):
            text1, text2 = tok1.text, tok2.text
            e1, e2 = embed1[i][j], embed2[i][j]
            
            diff = torch.sum(e1 - e2).item()
            if text1 != text2 and diff == 0.0:
                log.error(
                    f"ERROR: same embeddings, different text! "
                    f"diff={diff} text1: {text1} text2: {text2}"
                )            
                return False

    return True