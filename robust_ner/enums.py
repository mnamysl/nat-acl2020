from enum import Enum

class TrainingMode(Enum):
    """
    Training mode (one of: standard, stability, augmentation)
    """

    Standard = 'standard'
    Stability = 'stability'
    Augmentation = 'augmentation'

    def __str__(self):
        return self.name


class EvalMode(Enum):
    """
    Evaluation mode (one of: standard, misspellings)
    """

    Standard = 'standard'
    Misspellings = 'misspellings'

    def __str__(self):
        return self.name


class MisspellingMode(Enum):
    """
    Misspellings mode (one of: rand, cmx, typos)
    """

    Random = 'rand'
    ConfusionMatrixBased = 'cmx'
    Typos = 'typos'

    def __str__(self):
        return self.name