from pathlib import Path
from typing import List, Union

import datetime

from torch.optim.sgd import SGD
from torch.utils.data.dataset import ConcatDataset

import flair
import flair.nn
from flair.data import Sentence, MultiCorpus, Corpus
from flair.datasets import DataLoader
from flair.training_utils import (
    init_output_file,
    WeightExtractor,
    clear_embeddings,
    EvaluationMetric,
    log_line,
    add_file_handler,
    Result,
)
from flair.optim import *

from flair.trainers import ModelTrainer

from robust_ner.noise import (
    make_char_vocab,
)

from robust_ner.confusion_matrix import (
    load_confusion_matrix,
    filter_cmx,
    make_vocab_from_lut,
)

from robust_ner.enums import (
    TrainingMode,
    MisspellingMode,
    EvalMode,
)

log = logging.getLogger("flair")


class ParameterizedModelTrainer(ModelTrainer):
    def __init__(
        self,
        model: flair.nn.Model,
        corpus: Corpus,
        optimizer: Optimizer = SGD,
        epoch: int = 0,
        loss: float = 10000.0,
        optimizer_state: dict = None,
        scheduler_state: dict = None,
    ):
        super(ParameterizedModelTrainer, self).__init__(model, corpus, optimizer, epoch, loss, optimizer_state, scheduler_state)

    def train(
        self,
        base_path: Union[Path, str],
        evaluation_metric: EvaluationMetric = EvaluationMetric.MICRO_F1_SCORE,
        learning_rate: float = 0.1,
        mini_batch_size: int = 32,
        eval_mini_batch_size: int = None,
        max_epochs: int = 100,
        anneal_factor: float = 0.5,
        patience: int = 3,
        train_with_dev: bool = False,
        monitor_train: bool = False,
        embeddings_in_memory: bool = True,
        checkpoint: bool = False,
        save_final_model: bool = True,
        anneal_with_restarts: bool = False,
        shuffle: bool = True,
        param_selection_mode: bool = False,
        num_workers: int = 8,
        valid_with_misspellings: bool = True,
        **kwargs,
    ) -> dict:

        if eval_mini_batch_size is None:
            eval_mini_batch_size = mini_batch_size

        # cast string to Path
        if type(base_path) is str:
            base_path = Path(base_path)

        log_handler = add_file_handler(log, base_path / "training.log")

        log_line(log)
        log.info(f'Model: "{self.model}"')
        log_line(log)
        log.info(f'Corpus: "{self.corpus}"')
        log_line(log)
        log.info("Parameters:")
        log.info(f' - learning_rate: "{learning_rate}"')
        log.info(f' - mini_batch_size: "{mini_batch_size}"')
        log.info(f' - patience: "{patience}"')
        log.info(f' - anneal_factor: "{anneal_factor}"')
        log.info(f' - max_epochs: "{max_epochs}"')
        log.info(f' - shuffle: "{shuffle}"')
        log.info(f' - train_with_dev: "{train_with_dev}"')
        log.info(f' - valid_with_misspellings: "{valid_with_misspellings}"')        
        log.info("Model:")                
        log.info(f' - hidden_size: "{self.model.hidden_size}"')
        log.info(f' - train_mode: "{self.model.train_mode}"')
        log.info(f' - alpha: "{self.model.alpha}"')      
        log.info(f' - misspell_mode: "{self.model.misspell_mode}"')      
        log.info(f' - misspelling_rate: "{self.model.misspelling_rate_train}"')
        log.info(f' - cmx_file: "{self.model.cmx_file_train}"')        
        log_line(log)
        log.info(f'Model training base path: "{base_path}"')
        log_line(log)
        log.info(f"Evaluation method: {evaluation_metric.name}")

        # determine what splits (train, dev, test) to evaluate and log
        log_train = True if monitor_train else False
        log_test = True if (not param_selection_mode and self.corpus.test) else False
        log_dev = True if not train_with_dev else False

        log_test = not log_dev

        eval_misspelling_rate = 0.05

        log_suffix = lambda prefix, rate, cm, mode: f"{prefix} (misspell: cmx={cm})" if mode == MisspellingMode.ConfusionMatrixBased else f"{prefix} (misspell: rate={rate})"

        loss_txt = init_output_file(base_path, "loss.tsv")
        with open(loss_txt, "a") as f:
            f.write(f"EPOCH\tTIMESTAMP\tBAD_EPOCHS\tLEARNING_RATE\tTRAIN_LOSS")

            dummy_result, _ = self.model.evaluate(
                [Sentence("d", labels=["0.1"])],
                eval_mini_batch_size,
                embeddings_in_memory,
            )
            if log_train:
                f.write(
                    "\tTRAIN_" + "\tTRAIN_".join(dummy_result.log_header.split("\t"))
                )
            if log_dev:
                f.write(
                    "\tDEV_LOSS\tDEV_"
                    + "\tDEV_".join(dummy_result.log_header.split("\t"))
                )
                if valid_with_misspellings:
                    suffix=log_suffix('DEV', eval_misspelling_rate, self.model.cmx_file_train, self.model.misspell_mode)
                    f.write(
                        f"\t{suffix}" + f"_LOSS\t{suffix})_" + f"\t{suffix}_".join(dummy_result.log_header.split("\t"))
                    )
                
            if log_test:
                f.write(
                    "\tTEST_LOSS\tTEST_"
                    + "\tTEST_".join(dummy_result.log_header.split("\t"))
                )
                if valid_with_misspellings:
                    suffix=log_suffix('TEST', eval_misspelling_rate, self.model.cmx_file_train, self.model.misspell_mode)
                    f.write(
                        f"\t{suffix}" + f"_LOSS\t{suffix})_" + f"\t{suffix}_".join(dummy_result.log_header.split("\t"))
                    )

            weight_extractor = WeightExtractor(base_path)

        optimizer = self.optimizer(self.model.parameters(), lr=learning_rate, **kwargs)
        if self.optimizer_state is not None:
            optimizer.load_state_dict(self.optimizer_state)

        # minimize training loss if training with dev data, else maximize dev score
        anneal_mode = "min" if train_with_dev else "max"

        if isinstance(optimizer, (AdamW, SGDW)):
            scheduler = ReduceLRWDOnPlateau(
                optimizer,
                factor=anneal_factor,
                patience=patience,
                mode=anneal_mode,
                verbose=True,
            )
        else:
            scheduler = ReduceLROnPlateau(
                optimizer,
                factor=anneal_factor,
                patience=patience,
                mode=anneal_mode,
                verbose=True,
            )
        if self.scheduler_state is not None:
            scheduler.load_state_dict(self.scheduler_state)

        train_data = self.corpus.train

        # if training also uses dev data, include in training set
        if train_with_dev:
            train_data = ConcatDataset([self.corpus.train, self.corpus.dev])

        dev_clean_score_history = []
        dev_noisy_score_history = []
        dev_clean_loss_history = []
        dev_noisy_loss_history = []
        train_loss_history = []

        complete_data = ConcatDataset([self.corpus.train, self.corpus.dev, self.corpus.test])
        char_vocab = make_char_vocab(complete_data)
        log.info(f"Vocabulary of the corpus (#{len(char_vocab)}): {char_vocab}")

        if self.model.misspell_mode == MisspellingMode.ConfusionMatrixBased:
            cmx, lut = load_confusion_matrix(self.model.cmx_file_train)
            cmx, lut = filter_cmx(cmx, lut, char_vocab)
        else:
            cmx, lut = None, {}

        loss_params = {}
        loss_params["verbose"] = False
        loss_params["char_vocab"] = char_vocab
        loss_params["cmx"] = cmx
        loss_params["lut"] = lut
        loss_params["embeddings_in_memory"] = embeddings_in_memory       
    
        # At any point you can hit Ctrl + C to break out of training early.
        try:
            previous_learning_rate = learning_rate

            for epoch in range(0 + self.epoch, max_epochs + self.epoch):
                log_line(log)
                try:
                    bad_epochs = scheduler.num_bad_epochs
                except:
                    bad_epochs = 0
                for group in optimizer.param_groups:
                    learning_rate = group["lr"]

                # reload last best model if annealing with restarts is enabled
                if (
                    learning_rate != previous_learning_rate
                    and anneal_with_restarts
                    and (base_path / "best-model.pt").exists()
                ):
                    log.info("resetting to best model")
                    self.model.load(base_path / "best-model.pt")

                previous_learning_rate = learning_rate

                # stop training if learning rate becomes too small
                if learning_rate < 0.0001:
                    log_line(log)
                    log.info("learning rate too small - quitting training!")
                    log_line(log)
                    break

                batch_loader = DataLoader(
                    train_data,
                    batch_size=mini_batch_size,
                    shuffle=shuffle,
                    num_workers=num_workers,
                )

                self.model.train()

                train_loss: float = 0
                train_auxilary_losses = {}
                seen_batches = 0
                total_number_of_batches = len(batch_loader)

                modulo = max(1, int(total_number_of_batches / 10))

                for batch_no, batch in enumerate(batch_loader):

                    loss, auxilary_losses = self.model.forward_loss(batch, params=loss_params)

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                    optimizer.step()

                    seen_batches += 1
                    train_loss += loss.item()

                    for k,v in auxilary_losses.items():
                        train_auxilary_losses[k] = train_auxilary_losses.get(k, 0) + v

                    clear_embeddings(
                        batch, also_clear_word_embeddings=not embeddings_in_memory
                    )

                    if batch_no % modulo == 0:
                        msg = f"epoch {epoch + 1} - iter {batch_no}/{total_number_of_batches} - loss {train_loss / seen_batches:.6f}"
                        
                        # note: this is the loss accumulated in the current epoch divided by the number of already seen batches

                        if len(train_auxilary_losses) > 0:
                            aux_losses_str = " ".join([f"{key}={value / seen_batches:.6f}" for (key, value) in train_auxilary_losses.items()])
                            msg += f" ({aux_losses_str})"
                        
                        log.info(msg)

                        iteration = epoch * total_number_of_batches + batch_no
                        if not param_selection_mode:
                            weight_extractor.extract_weights(
                                self.model.state_dict(), iteration
                            )

                train_loss /= seen_batches
                for k,v in auxilary_losses.items():
                    train_auxilary_losses[k] /= seen_batches

                self.model.eval()

                log_line(log)
                log.info(
                    f"EPOCH {epoch + 1} done: loss {train_loss:.6f} - lr {learning_rate:.4f} - bad epochs {bad_epochs}"
                )

                # anneal against train loss if training with dev, otherwise anneal against dev score
                current_score = train_loss

                with open(loss_txt, "a") as f:

                    f.write(
                        f"\n{epoch}\t{datetime.datetime.now():%H:%M:%S}\t{bad_epochs}\t{learning_rate:.4f}\t{train_loss}"
                    )

                    if log_train:
                        train_eval_result, train_loss = self.model.evaluate(
                            self.corpus.train,
                            eval_mini_batch_size,
                            embeddings_in_memory,
                            num_workers=num_workers,
                        )
                        f.write(f"\t{train_eval_result.log_line}")

                    if log_dev:
                        dev_eval_result_clean, dev_loss_clean = self.model.evaluate(
                            self.corpus.dev,
                            eval_mini_batch_size,
                            embeddings_in_memory,
                            num_workers=num_workers,
                        )
                        f.write(f"\t{dev_loss_clean}\t{dev_eval_result_clean.log_line}")
                        log.info(
                            f"DEV : loss {dev_loss_clean:.6f} - score {dev_eval_result_clean.main_score:.4f}"
                        )
                        # calculate scores using dev data if available
                        # append dev score to score history
                        dev_clean_score_history.append(dev_eval_result_clean.main_score)
                        dev_clean_loss_history.append(dev_loss_clean)

                        if valid_with_misspellings:
                            # evaluate on misspellings 
                            dev_eval_result_noisy, dev_loss_noisy = self.model.evaluate(
                                self.corpus.dev,
                                eval_mini_batch_size,
                                embeddings_in_memory,
                                num_workers=num_workers,
                                eval_mode=EvalMode.Misspellings,
                                misspell_mode=self.model.misspell_mode,
                                char_vocab=char_vocab,
                                cmx=cmx,
                                lut=lut,
                                misspelling_rate=eval_misspelling_rate,
                            )                            

                            f.write(f"\t{dev_loss_noisy}\t{dev_eval_result_noisy.log_line}")
                            
                            log.info(                                
                                f"{log_suffix('DEV', eval_misspelling_rate, self.model.cmx_file_train, self.model.misspell_mode)}"
                                + f" : loss {dev_loss_noisy:.6f} - score {dev_eval_result_noisy.main_score:.4f}"
                            )                        

                            # calculate scores using dev data if available
                            # append dev score to score history
                            dev_noisy_score_history.append(dev_eval_result_noisy)
                            dev_noisy_loss_history.append(dev_loss_noisy)
                        
                            current_score = (dev_eval_result_clean.main_score + dev_eval_result_noisy.main_score) / 2.0
                        else:
                            current_score = dev_eval_result_clean.main_score
    
                    if log_test:
                        test_eval_result_clean, test_loss_clean = self.model.evaluate(
                            self.corpus.test,
                            eval_mini_batch_size,
                            embeddings_in_memory,
                            base_path / f"test.tsv",
                            num_workers=num_workers,
                        )
                        f.write(f"\t{test_loss_clean}\t{test_eval_result_clean.log_line}")
                        log.info(
                            f"TEST : loss {test_loss_clean:.6f} - score {test_eval_result_clean.main_score:.4f}"
                        )    

                        if valid_with_misspellings:               
                            # evaluate on misspellings                       
                            test_eval_result_noisy, test_loss_noisy = self.model.evaluate(
                                self.corpus.test,
                                eval_mini_batch_size,
                                embeddings_in_memory,
                                base_path / f"test.tsv",
                                num_workers=num_workers,
                                eval_mode=EvalMode.Misspellings,
                                misspell_mode=self.model.misspell_mode,
                                char_vocab=char_vocab,
                                cmx=cmx,
                                lut=lut,
                                misspelling_rate=eval_misspelling_rate,
                            )
                            
                            f.write(f"\t{test_loss_noisy}\t{test_eval_result_noisy.log_line}")
                            log.info(
                                f"{log_suffix('TEST', eval_misspelling_rate, self.model.cmx_file_train, self.model.misspell_mode)}"
                                + f" : loss {test_loss_noisy:.6f} - score {test_eval_result_noisy.main_score:.4f}"
                                #f"TEST (misspell, rate={eval_misspelling_rate}) : loss {test_loss_noisy:.6f} - score {test_eval_result_noisy.main_score:.4f}"
                            )                                                                  

                scheduler.step(current_score)

                train_loss_history.append(train_loss)

                # if checkpoint is enable, save model at each epoch
                if checkpoint and not param_selection_mode:
                    self.model.save_checkpoint(
                        base_path / "checkpoint.pt",
                        optimizer.state_dict(),
                        scheduler.state_dict(),
                        epoch + 1,
                        train_loss,
                    )

                # if we use dev data, remember best model based on dev evaluation score
                if (
                    not train_with_dev
                    and not param_selection_mode
                    and current_score == scheduler.best
                ):
                    log.info("'best-model.pt' saved.")
                    self.model.save(base_path / "best-model.pt")

            # if we do not use dev data for model selection, save final model
            if save_final_model and not param_selection_mode:
                self.model.save(base_path / "final-model.pt")

        except KeyboardInterrupt:
            log_line(log)
            log.info("Exiting from training early.")
            if not param_selection_mode:
                log.info("Saving model ...")
                self.model.save(base_path / "final-model.pt")
                log.info("Done.")

        # test best model if test data is present
        if self.corpus.test:
            final_score_clean = self.final_test(
                base_path,
                embeddings_in_memory,
                evaluation_metric,
                eval_mini_batch_size,
                num_workers,
            )
            final_score_noisy = self.final_test(
                base_path,
                embeddings_in_memory,
                evaluation_metric,
                eval_mini_batch_size,
                num_workers,
                eval_mode=EvalMode.Misspellings,
                misspell_mode=self.model.misspell_mode,
                misspelling_rate=eval_misspelling_rate,
                char_vocab=char_vocab,
                cmx=cmx,
                lut=lut,
            )

        else:
            final_score_clean, final_score_noisy = 0, 0
            log.info("Test data not provided setting final score to 0")

        log.removeHandler(log_handler)

        return {
            "test_score_clean": final_score_clean,
            "test_score_noisy": final_score_noisy,
            "dev_clean_score_history": dev_clean_score_history,
            "dev_noisy_score_history": dev_noisy_score_history,
            "train_loss_history": train_loss_history,
            "dev_clean_loss_history": dev_clean_loss_history,
            "dev_noisy_loss_history": dev_noisy_loss_history,
        }

    def final_test(
        self,
        base_path: Path,
        embeddings_in_memory: bool,
        evaluation_metric: EvaluationMetric,
        eval_mini_batch_size: int,
        num_workers: int = 8,
        eval_mode: EvalMode = EvalMode.Standard,
        misspell_mode: MisspellingMode = MisspellingMode.Random,
        misspelling_rate: float = 0.0,
        char_vocab: set = {},
        cmx = None,
        lut = {},
    ):

        log_line(log)
        log.info("Testing using best model ...")

        self.model.eval()

        if (base_path / "best-model.pt").exists():
            self.model = self.model.load(base_path / "best-model.pt")

        test_results, test_loss = self.model.evaluate(
            self.corpus.test,
            eval_mini_batch_size=eval_mini_batch_size,
            embeddings_in_memory=embeddings_in_memory,
            out_path=base_path / "test.tsv",
            num_workers=num_workers,
            eval_mode=eval_mode,
            misspell_mode=misspell_mode,
            misspelling_rate=misspelling_rate,
            char_vocab=char_vocab,
            cmx=cmx,
            lut=lut,
        )

        test_results: Result = test_results
        log.info(test_results.log_line)
        log.info(test_results.detailed_results)
        log_line(log)

        # if we are training over multiple datasets, do evaluation for each
        if type(self.corpus) is MultiCorpus:
            for subcorpus in self.corpus.corpora:
                log_line(log)
                self.model.evaluate(
                    subcorpus.test,
                    eval_mini_batch_size,
                    embeddings_in_memory,
                    base_path / f"{subcorpus.name}-test.tsv",
                    eval_mode=eval_mode,
                    misspelling_rate=misspelling_rate,
                    char_vocab=char_vocab,
                )

        # get and return the final test score of best model
        final_score = test_results.main_score

        return final_score

    @classmethod
    def load_from_checkpoint(
        cls, checkpoint, corpus: Corpus, optimizer: Optimizer = SGD
    ):
        return ParameterizedModelTrainer(
            checkpoint["model"],
            corpus,
            optimizer,
            epoch=checkpoint["epoch"],
            loss=checkpoint["loss"],
            optimizer_state=checkpoint["optimizer_state_dict"],
            scheduler_state=checkpoint["scheduler_state_dict"],
        )