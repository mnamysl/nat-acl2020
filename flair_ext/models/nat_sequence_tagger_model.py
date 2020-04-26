import logging
import sys
import numpy as np

from pathlib import Path

import torch.nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset

import flair.nn
import torch

import flair.embeddings
from flair.data import Dictionary, Sentence, Token, Label
from flair.datasets import DataLoader

from typing import List, Union
from enum import Enum

from flair.training_utils import clear_embeddings, Metric, Result

from flair.models import SequenceTagger

from robust_ner.enums import TrainingMode, EvalMode, MisspellingMode
from robust_ner.noise import noise_sentences
from robust_ner.embeddings import check_embeddings

from flair_ext.nn import ParameterizedModel

from tqdm import tqdm

log = logging.getLogger("flair")


def get_masked_sum(loss_unreduced, lengths):

    loss_sum = 0
    for batch_idx, length in enumerate(lengths):
        loss_sum += loss_unreduced[batch_idx][:length].sum()
    
    return loss_sum

def get_per_token_mean(loss_sum, lengths):
    return loss_sum / float(sum(lengths))

def get_per_batch_mean(loss_sum, lengths):
    return loss_sum / float(len(lengths))


class NATSequenceTagger(SequenceTagger, ParameterizedModel):
    def __init__(
        self,
        hidden_size: int,
        embeddings: flair.embeddings.TokenEmbeddings,
        tag_dictionary: Dictionary,
        tag_type: str,
        use_crf: bool = True,
        use_rnn: bool = True,
        rnn_layers: int = 1,
        dropout: float = 0.0,
        word_dropout: float = 0.05,
        locked_dropout: float = 0.5,
        pickle_module: str = "pickle",
        train_mode: TrainingMode = TrainingMode.Standard,
        alpha: float = 1.0,
        misspelling_rate: float = 0.0,
        cmx_file = "",        
    ):

        super(NATSequenceTagger, self).__init__(hidden_size, embeddings, tag_dictionary, tag_type,
              use_crf, use_rnn, rnn_layers, dropout, word_dropout, locked_dropout, pickle_module)

        self.set_training_params(train_mode, alpha, misspelling_rate, cmx_file)

    def set_training_params(self, train_mode: TrainingMode, alpha: float = 1.0, misspelling_rate: float = 0.0, cmx_file = ""):
        self.train_mode = train_mode
        self.alpha = alpha
        self.misspelling_rate_train = misspelling_rate
        self.cmx_file_train = cmx_file

        if self.cmx_file_train:
            self.misspell_mode = MisspellingMode.ConfusionMatrixBased
        else:
            self.misspell_mode = MisspellingMode.Random

    def _get_state_dict(self):
        model_state = super(NATSequenceTagger, self)._get_state_dict()
        model_state["train_mode"] = self.train_mode        
        return model_state

    def _init_model_with_state_dict(state):

        use_dropout = 0.0 if not "use_dropout" in state.keys() else state["use_dropout"]
        use_word_dropout = (
            0.0 if not "use_word_dropout" in state.keys() else state["use_word_dropout"]
        )
        use_locked_dropout = (
            0.0
            if not "use_locked_dropout" in state.keys()
            else state["use_locked_dropout"]
        )

        train_mode = TrainingMode.Standard if not "train_mode" in state.keys() else state["train_mode"]

        model = NATSequenceTagger(
            hidden_size=state["hidden_size"],
            embeddings=state["embeddings"],
            tag_dictionary=state["tag_dictionary"],
            tag_type=state["tag_type"],
            use_crf=state["use_crf"],
            use_rnn=state["use_rnn"],
            rnn_layers=state["rnn_layers"],
            dropout=use_dropout,
            word_dropout=use_word_dropout,
            locked_dropout=use_locked_dropout,
            train_mode=train_mode
        )
        model.load_state_dict(state["state_dict"])
        return model

    def evaluate(
        self,
        sentences: Dataset,
        eval_mini_batch_size: int = 32,
        embeddings_in_memory: bool = True,
        out_path: Path = None,
        num_workers: int = 8,
        eval_mode: EvalMode = EvalMode.Standard,
        misspell_mode: MisspellingMode = MisspellingMode.Random,
        misspelling_rate: float = 0.0,
        char_vocab: set = {}, 
        lut: dict = {},
        cmx: np.array = None,
        typos: dict = {},
        spell_check = None,
    ) -> (Result, float):

        eval_params = {}
        eval_params["eval_mode"] = eval_mode
        eval_params["misspelling_rate"] = misspelling_rate
        eval_params["misspell_mode"] = misspell_mode
        eval_params["char_vocab"] = char_vocab
        eval_params["lut"] = lut
        eval_params["cmx"] = cmx
        eval_params["typos"] = typos
        eval_params["embeddings_in_memory"] = embeddings_in_memory 
        eval_params["spell_check"] = spell_check       

        with torch.no_grad():
            eval_loss = 0

            batch_no: int = 0

            batch_loader = DataLoader(
                sentences,
                batch_size=eval_mini_batch_size,
                shuffle=False,
                num_workers=num_workers,
            )

            metric = Metric("Evaluation")

            lines: List[str] = []
            for batch in batch_loader:
                batch_no += 1

                with torch.no_grad():
                    features = self.forward(batch, eval_params)
                    loss = self._calculate_loss(features, batch)
                    tags, _ = self._obtain_labels(features, batch)

                eval_loss += loss

                for (sentence, sent_tags) in zip(batch, tags):
                    for (token, tag) in zip(sentence.tokens, sent_tags):
                        token: Token = token
                        token.add_tag_label("predicted", tag)

                        # append both to file for evaluation
                        eval_line = "{} {} {} {}\n".format(
                            token.text,
                            token.get_tag(self.tag_type).value,
                            tag.value,
                            tag.score,
                        )
                        lines.append(eval_line)
                    lines.append("\n")
                for sentence in batch:
                    # make list of gold tags
                    gold_tags = [
                        (tag.tag, str(tag)) for tag in sentence.get_spans(self.tag_type)
                    ]
                    # make list of predicted tags
                    predicted_tags = [
                        (tag.tag, str(tag)) for tag in sentence.get_spans("predicted")
                    ]

                    # check for true positives, false positives and false negatives
                    for tag, prediction in predicted_tags:
                        if (tag, prediction) in gold_tags:
                            metric.add_tp(tag)
                        else:
                            metric.add_fp(tag)

                    for tag, gold in gold_tags:
                        if (tag, gold) not in predicted_tags:
                            metric.add_fn(tag)
                        else:
                            metric.add_tn(tag)

                clear_embeddings(
                    batch, also_clear_word_embeddings=not embeddings_in_memory
                )

            eval_loss /= batch_no

            if out_path is not None:
                with open(out_path, "w", encoding="utf-8") as outfile:
                    outfile.write("".join(lines))

            detailed_result = (
                f"\nMICRO_AVG: acc {metric.micro_avg_accuracy()} - f1-score {metric.micro_avg_f_score()}"
                f"\nMACRO_AVG: acc {metric.macro_avg_accuracy()} - f1-score {metric.macro_avg_f_score()}"
            )
            for class_name in metric.get_classes():
                detailed_result += (
                    f"\n{class_name:<10} tp: {metric.get_tp(class_name)} - fp: {metric.get_fp(class_name)} - "
                    f"fn: {metric.get_fn(class_name)} - tn: {metric.get_tn(class_name)} - precision: "
                    f"{metric.precision(class_name):.4f} - recall: {metric.recall(class_name):.4f} - "
                    f"accuracy: {metric.accuracy(class_name):.4f} - f1-score: "
                    f"{metric.f_score(class_name):.4f}"
                )

            result = Result(
                main_score=metric.micro_avg_f_score(),
                log_line=f"{metric.precision()}\t{metric.recall()}\t{metric.micro_avg_f_score()}",
                log_header="PRECISION\tRECALL\tF1",
                detailed_results=detailed_result,
            )

            return result, eval_loss

    def predict(
        self,
        sentences: Union[List[Sentence], Sentence], 
        eval_mode: EvalMode = EvalMode.Standard,
        mini_batch_size=32,
        embeddings_in_memory: bool = True, 
        verbose=False,
        misspell_mode: MisspellingMode = MisspellingMode.Random,
        misspelling_rate: float = 0.0,
        char_vocab: set = {}, 
        lut: dict = {},
        cmx: np.array = None,
        typos: dict = {},
        spell_check = None,
    ) -> List[Sentence]:

        predict_params = {}
        predict_params["eval_mode"] = eval_mode
        predict_params["misspelling_rate"] = misspelling_rate
        predict_params["misspell_mode"] = misspell_mode
        predict_params["char_vocab"] = char_vocab
        predict_params["lut"] = lut
        predict_params["cmx"] = cmx
        predict_params["typos"] = typos
        predict_params["embeddings_in_memory"] = embeddings_in_memory 
        predict_params["spell_check"] = spell_check 

        with torch.no_grad():
            if isinstance(sentences, Sentence):
                sentences = [sentences]

            filtered_sentences = self._filter_empty_sentences(sentences)

            # remove previous embeddings
            clear_embeddings(filtered_sentences, also_clear_word_embeddings=True)

            # revere sort all sequences by their length
            filtered_sentences.sort(key=lambda x: len(x), reverse=True)

            # make mini-batches
            batches = [
                filtered_sentences[x : x + mini_batch_size]
                for x in range(0, len(filtered_sentences), mini_batch_size)
            ]

            # progress bar for verbosity
            if verbose:
                batches = tqdm(batches)

            for i, batch in enumerate(batches):

                if verbose:
                    batches.set_description(f"Inferencing on batch {i}")

                with torch.no_grad():
                    feature = self.forward(batch, predict_params)
                    tags, all_tags = self._obtain_labels(feature, batch)

                for (sentence, sent_tags, sent_all_tags) in zip(batch, tags, all_tags):
                    for (token, tag, token_all_tags) in zip(
                        sentence.tokens, sent_tags, sent_all_tags
                    ):
                        token.add_tag_label(self.tag_type, tag)
                        token.add_tags_proba_dist(self.tag_type, token_all_tags)

                # clearing token embeddings to save memory
                clear_embeddings(batch, also_clear_word_embeddings=True)

            return sentences

    def forward_loss(
        self, sentences: Union[List[Sentence], Sentence], params: dict = {}
    ) -> (torch.tensor, dict):

        verbose = params.get("verbose", False)        
        char_vocab = params.get("char_vocab", {})
        cmx = params.get("cmx", {})
        lut = params.get("lut", {})
        embeddings_in_memory = params.get("embeddings_in_memory", True)
        
        auxilary_losses = {}

        alpha = self.alpha
        misspelling_rate_train = self.misspelling_rate_train

        self.zero_grad()
      
        if self.train_mode == TrainingMode.Standard:
            loss = self._forward_loss_standard(sentences)
        elif self.train_mode == TrainingMode.Stability:
            loss, auxilary_losses = self._forward_loss_stability(sentences, alpha=alpha, 
                misspelling_rate=misspelling_rate_train, embeddings_in_memory=embeddings_in_memory, 
                char_vocab=char_vocab, cmx=cmx, lut=lut, verbose=verbose)
        elif self.train_mode == TrainingMode.Augmentation:
            loss, auxilary_losses = self._forward_loss_data_augmentation(sentences, alpha=alpha, 
                misspelling_rate=misspelling_rate_train, cmx=cmx, lut=lut, 
                embeddings_in_memory=embeddings_in_memory, char_vocab=char_vocab, verbose=verbose)
        else:
            raise Exception("Training mode '{}' is not supported!".format(self.train_mode))
        
        return loss, auxilary_losses

    def _forward_loss_standard(
        self, sentences: Union[List[Sentence], Sentence]
    ) -> torch.tensor:
                    
        features = self._forward_standard(sentences)
        return self._calculate_loss(features, sentences)

    def _forward_loss_data_augmentation(
        self, sentences: Union[List[Sentence], Sentence], alpha: float, misspelling_rate: float, 
        char_vocab: dict, lut: dict = {}, cmx: np.array = None, embeddings_in_memory: bool = True, verbose: bool = False
    ) -> (torch.tensor, dict):
        """
        Data augmentation objective. Returns the auxiliary loss as the sum of standard objectives calculated on the
        original and the perturbed samples.
        """
                
        misspelled_sentences, _ = noise_sentences(sentences, self.misspell_mode, misspelling_rate, char_vocab, cmx, lut, {}, verbose)
        clear_embeddings(misspelled_sentences, also_clear_word_embeddings=True)
    
        embeddings, lengths = self._embed_sentences(sentences)                
        embeddings_misspell, lengths_misspell = self._embed_sentences(misspelled_sentences)
                
        if not check_embeddings(sentences, misspelled_sentences, embeddings, embeddings_misspell):
            log.warning("WARNING: embedding of the misspelled text may be invalid!")

        outputs_base, _ = self._forward(embeddings, lengths)
        outputs_misspell, _ = self._forward(embeddings_misspell, lengths_misspell)
        
        loss_base = self._calculate_loss(outputs_base, sentences)
        loss_misspell = alpha * self._calculate_loss(outputs_misspell, misspelled_sentences)

        auxilary_losses = { 'loss_base': loss_base, 'loss_misspell': loss_misspell }
        
        return (loss_base + loss_misspell), auxilary_losses

    def _forward_loss_stability(
        self, sentences: Union[List[Sentence], Sentence], alpha: float, misspelling_rate: float, char_vocab: dict, 
        lut: dict = {}, cmx: np.array = None, embeddings_in_memory: bool = True, verbose: bool = False
    ) -> (torch.tensor, dict):
        """
        stability objective for classification -> KL divergence (see Zheng 2016 Eq.10)
        L_stab(x,x') = -sum_j(P(yj|x)*log(P(yj|x')))
        The output loss is the sum of the standard loss and the similarity objective.
        """

        misspelled_sentences, _ = noise_sentences(sentences, self.misspell_mode, misspelling_rate, char_vocab, cmx, lut, {}, verbose)
        clear_embeddings(misspelled_sentences, also_clear_word_embeddings=True)

        embeddings, lengths = self._embed_sentences(sentences)                
        embeddings_misspell, lengths_misspell = self._embed_sentences(misspelled_sentences)
                
        if not check_embeddings(sentences, misspelled_sentences, embeddings, embeddings_misspell):
            log.warning("WARNING: embedding of the misspelled text may be invalid!")

        outputs_base, features_base = self._forward(embeddings, lengths)
        outputs_misspell, features_misspell = self._forward(embeddings_misspell, lengths_misspell)
                                
        loss_base = self._calculate_loss(outputs_base, sentences)

        target_distrib = F.softmax(outputs_base, dim=2).transpose(1, 2).detach()
        input_log_distrib = F.log_softmax(outputs_misspell, dim=2).transpose(1, 2)
        loss_stability = alpha * F.kl_div(input_log_distrib, target_distrib, reduction='none').transpose(2, 1)            
        loss_sum = get_masked_sum(loss_stability, lengths)
        loss_mean = get_per_batch_mean(loss_sum, lengths)
        # log.info(f"loss_base: {loss_base.item():.4f} loss_stability: {loss_mean.item():.4f}")
                                             
        auxilary_losses = { 'loss_base': loss_base, 'loss_kldiv': loss_mean }
        return (loss_base + loss_mean), auxilary_losses

    def _embed_sentences(self, sentences: List[Sentence]) -> (torch.tensor, List[int]):
                        
        self.embeddings.embed(sentences)        

        sentences.sort(key=lambda x: len(x), reverse=True)

        lengths: List[int] = [len(sentence.tokens) for sentence in sentences]
        tag_list: List = []
        longest_token_sequence_in_batch: int = lengths[0]

        # initialize zero-padded word embeddings tensor
        embeddings = torch.zeros(
           [
               len(sentences),
               longest_token_sequence_in_batch,
               self.embeddings.embedding_length,
           ],
           dtype=torch.float,
           device=flair.device,
        )

        for s_id, sentence in enumerate(sentences):
            # fill values with word embeddings
            embeddings[s_id][: len(sentence)] = torch.cat(
               [token.get_embedding().unsqueeze(0) for token in sentence], 0
            )            
        
            # get the tags in this sentence
            tag_idx: List[int] = [
                self.tag_dictionary.get_idx_for_item(token.get_tag(self.tag_type).value)
                for token in sentence
            ]
            # add tags as tensor
            tag = torch.LongTensor(tag_idx).to(flair.device)
            tag_list.append(tag)

        return embeddings, lengths

    def _forward(self, embeddings: torch.tensor, lengths: List[int]):

        encoder_features = embeddings.transpose(0, 1)        

        # --------------------------------------------------------------------
        # FF PART
        # --------------------------------------------------------------------
        if self.use_dropout > 0.0:
            encoder_features = self.dropout(encoder_features)
        if self.use_word_dropout > 0.0:
            encoder_features = self.word_dropout(encoder_features)
        if self.use_locked_dropout > 0.0:
            encoder_features = self.locked_dropout(encoder_features)

        if self.relearn_embeddings:
            encoder_features = self.embedding2nn(encoder_features)

        if self.use_rnn:
            packed = torch.nn.utils.rnn.pack_padded_sequence(encoder_features, lengths)

            rnn_output, hidden = self.rnn(packed)

            decoder_features, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(
                rnn_output
            )

            if self.use_dropout > 0.0:
                decoder_features = self.dropout(decoder_features)
            # word dropout only before LSTM - TODO: more experimentation needed
            # if self.use_word_dropout > 0.0:
            #     decoder_features = self.word_dropout(decoder_features)
            if self.use_locked_dropout > 0.0:
                decoder_features = self.locked_dropout(decoder_features)
        
        outputs = self.linear(decoder_features)
        
        return outputs.transpose(0, 1), decoder_features.transpose(0, 1)

    def forward(self, sentences: List[Sentence], params: dict = {}):
        
        verbose = params.get("verbose", False)                
        eval_mode = params.get("eval_mode", TrainingMode.Standard)
        misspell_mode = params.get("misspell_mode", MisspellingMode.Random)
        misspelling_rate = params.get("misspelling_rate", 0.0)
        char_vocab = params.get("char_vocab", {})
        lut = params.get("lut", {})
        cmx = params.get("cmx", {})
        typos = params.get("typos", {})
        spell_check = params.get("spell_check", None)

        self.zero_grad()

        if eval_mode is EvalMode.Standard:
            outputs = self._forward_standard(sentences, spell_check)
        elif eval_mode is EvalMode.Misspellings:
            outputs = self._forward_misspelled(sentences, misspelling_rate=misspelling_rate, misspell_mode=misspell_mode,
                char_vocab=char_vocab, lut=lut, cmx=cmx, typos=typos, spell_check=spell_check, verbose=verbose)
        else:
            raise Exception("Evaluation mode '{}' is not supported!".format(eval_mode))
                
        return outputs

    def _forward_standard(self, sentences: List[Sentence], spell_check = None):

        # self.zero_grad()        
        if spell_check != None:
            from robust_ner.spellcheck import correct_sentences
            corrected_sentences = correct_sentences(spell_check, sentences)
            clear_embeddings(corrected_sentences, also_clear_word_embeddings=True)
             
            embeddings, lengths = self._embed_sentences(corrected_sentences)
        else:
            embeddings, lengths = self._embed_sentences(sentences)        
        
        outputs, _ = self._forward(embeddings, lengths)        
        
        return outputs

    def _forward_misspelled(
        self, sentences: Union[List[Sentence], Sentence], misspelling_rate: float, misspell_mode: MisspellingMode, char_vocab: set, 
        cmx: np.array, lut: dict, typos:dict, spell_check = None, verbose: bool = False
    ) -> (torch.tensor, dict):
        
        misspelled_sentences, _ = noise_sentences(sentences, misspell_mode, misspelling_rate, char_vocab, cmx, lut, typos, verbose)
        clear_embeddings(misspelled_sentences, also_clear_word_embeddings=True)        

        outputs_misspell = self._forward_standard(misspelled_sentences, spell_check)
        
        return outputs_misspell    

    