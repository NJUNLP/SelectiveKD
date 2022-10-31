# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
import json
import logging
import os
from argparse import Namespace

import torch
import numpy as np
from fairseq import metrics, utils
from fairseq.data import (
    LanguagePairDataset,
    data_utils,
    encoders,
)
from fairseq.tasks import FairseqTask, register_task
from fairseq.tasks.translation import load_langpair_dataset, TranslationConfig
from fairseq.models.transformer import TransformerModel
from plugins.data import MixedDataset
from fairseq.optim.amp_optimizer import AMPOptimizer

EVAL_BLEU_ORDER = 4


logger = logging.getLogger(__name__)

@dataclass
class TranslationSKDDSConfig(TranslationConfig):
    total_up: int = field(
        default=100000, metadata={"help": "total updates"}
    )
    evaluator: str = field(
        default="", metadata={"help": "name of evaluator model"}
    )
    kd_threshold: float = field(
        default=0.0, metadata={"help": "use kd if acc is below threshold"}
    )
    kd_threshold_offset: float = field(
        default=1.01, metadata={"help": "use kd if acc is below threshold"}
    )
    kd_data: str = field(
        default="", metadata={"help": "path of kd data"}
    )


@register_task("translation_selective_kd_deepshallow", dataclass=TranslationSKDDSConfig)
class TranslationSelectiveKDDeepShallowTask(FairseqTask):
    """
    Translate from one (source) language to another (target) language.

    Args:
        src_dict (~fairseq.data.Dictionary): dictionary for the source language
        tgt_dict (~fairseq.data.Dictionary): dictionary for the target language

    .. note::

        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.
    """

    cfg: TranslationConfig

    def __init__(self, cfg: TranslationConfig, src_dict, tgt_dict):
        super().__init__(cfg)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict

    @classmethod
    def setup_task(cls, cfg: TranslationConfig, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """

        paths = utils.split_paths(cfg.data)
        assert len(paths) > 0
        # find language pair automatically
        if cfg.source_lang is None or cfg.target_lang is None:
            cfg.source_lang, cfg.target_lang = data_utils.infer_language_pair(paths[0])
        if cfg.source_lang is None or cfg.target_lang is None:
            raise Exception(
                "Could not infer language pair, please provide it explicitly"
            )

        # load dictionaries
        src_dict = cls.load_dictionary(
            os.path.join(paths[0], "dict.{}.txt".format(cfg.source_lang))
        )
        tgt_dict = cls.load_dictionary(
            os.path.join(paths[0], "dict.{}.txt".format(cfg.target_lang))
        )
        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()
        logger.info("[{}] dictionary: {} types".format(cfg.source_lang, len(src_dict)))
        logger.info("[{}] dictionary: {} types".format(cfg.target_lang, len(tgt_dict)))

        return cls(cfg, src_dict, tgt_dict)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.cfg.data)
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]

        # infer langcode
        src, tgt = self.cfg.source_lang, self.cfg.target_lang

        if split == "train":
            assert self.cfg.kd_data != ""
            self.datasets[split] = MixedDataset(
                data_path, self.cfg.kd_data, src, tgt, self.src_dict, self.tgt_dict, combine, self.cfg
            )
        else:
            self.datasets[split] = load_langpair_dataset(
                data_path,
                split,
                src,
                self.src_dict,
                tgt,
                self.tgt_dict,
                combine=combine,
                dataset_impl=self.cfg.dataset_impl,
                upsample_primary=self.cfg.upsample_primary,
                left_pad_source=self.cfg.left_pad_source,
                left_pad_target=self.cfg.left_pad_target,
                max_source_positions=self.cfg.max_source_positions,
                max_target_positions=self.cfg.max_target_positions,
                load_alignments=self.cfg.load_alignments,
                truncate_source=self.cfg.truncate_source,
                num_buckets=self.cfg.num_batch_buckets,
                shuffle=(split != "test"),
                pad_to_multiple=self.cfg.required_seq_len_multiple,
            )

        # load evaluator
        if split == 'train':
            self.evaluator_name = self.cfg.evaluator
            state = torch.load(f'../checkpoints/{self.evaluator_name}/avg_last_5_checkpoint.pt')
            args = state['cfg']['model']
            self.evaluator = TransformerModel.build_model(args, self) # can be shared by CMLM and Vanilla
            self.evaluator.load_state_dict(state['model'], strict=True)
            print("Successfully loaded evaluator model!")

    def build_dataset_for_inference(self, src_tokens, src_lengths, constraints=None):
        return LanguagePairDataset(
            src_tokens,
            src_lengths,
            self.source_dictionary,
            tgt_dict=self.target_dictionary,
            constraints=constraints,
        )

    def build_model(self, cfg, from_checkpoint=False):
        model = super().build_model(cfg, from_checkpoint)
        if self.cfg.eval_bleu:
            detok_args = json.loads(self.cfg.eval_bleu_detok_args)
            self.tokenizer = encoders.build_tokenizer(
                Namespace(tokenizer=self.cfg.eval_bleu_detok, **detok_args)
            )

            gen_args = json.loads(self.cfg.eval_bleu_args)
            self.sequence_generator = self.build_generator(
                [model], Namespace(**gen_args)
            )
        return model

    def get_sample(self, sample, update_num):
        if "kd_target" not in sample:
            return

        def compute_acc(pred_tokens, target):
            seq_lens = (prev_output_tokens.ne(self.tgt_dict.pad())).sum(1)
            return (pred_tokens == target).sum(-1) / seq_lens

        self.evaluator.eval()
        src_tokens, src_lengths = (
            sample["net_input"]["src_tokens"],
            sample["net_input"]["src_lengths"],
        )

        if next(self.evaluator.parameters()) != sample["target"].device:
            self.evaluator = self.evaluator.to(sample["target"].device)

        prev_output_tokens = sample["net_input"]["prev_output_tokens"]

        with torch.no_grad():
            encoder_out = self.evaluator.encoder(src_tokens, src_lengths=src_lengths)
            attn_list, inner_states = self.evaluator.decoder.get_attention(
                prev_output_tokens=prev_output_tokens,
                encoder_out=encoder_out
            )
            word_ins_out = self.evaluator.decoder.output_layer(inner_states[-1].transpose(0, 1))
            _, output_tokens = word_ins_out.max(-1)
            # select sentences to kd
            acc = compute_acc(output_tokens, sample["target"])
            use_kd = acc < self.cfg.kd_threshold + (update_num / self.cfg.total_up) * self.cfg.kd_threshold_offset            
            # generate kd outputs
            raw_len, kd_len = sample["target"].shape[1], sample["kd_target"].shape[1]
            max_len = max(raw_len, kd_len)
            if kd_len > raw_len:
                new_tgt_tokens = torch.zeros_like(sample["kd_target"]).fill_(self.tgt_dict.pad())
                new_tgt_tokens[:, :raw_len] = sample["target"]
                sample["target"] = new_tgt_tokens
                new_prev = torch.zeros_like(sample["kd_target"]).fill_(self.tgt_dict.pad())
                new_prev[:, :raw_len] = sample["net_input"]["prev_output_tokens"]
                sample["net_input"]["prev_output_tokens"] = new_prev
            elif raw_len > kd_len:
                new_kd_target = torch.zeros_like(sample["target"]).fill_(self.tgt_dict.pad())
                new_kd_target[:, :kd_len] = sample["kd_target"]
                sample["kd_target"] = new_kd_target
                new_kd_prev = torch.zeros_like(sample["target"]).fill_(self.tgt_dict.pad())
                new_kd_prev[:, :kd_len] = sample["kd_prev_output_tokens"]
                sample["kd_prev_output_tokens"] = new_kd_prev
            kd_mask = use_kd.reshape(-1, 1).expand(-1, max_len)
            sample["target"][kd_mask] = sample["kd_target"][kd_mask]
            sample["net_input"]["prev_output_tokens"][kd_mask] = sample["kd_prev_output_tokens"][kd_mask]
            sample["kd_mask"] = kd_mask


    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        model.train()
        model.set_num_updates(update_num)

        self.get_sample(sample, update_num)

        with torch.autograd.profiler.record_function("forward"):
            with torch.cuda.amp.autocast(enabled=(isinstance(optimizer, AMPOptimizer))):
                loss, sample_size, logging_output = criterion(model, sample)
        if ignore_grad:
            loss *= 0
        with torch.autograd.profiler.record_function("backward"):
            optimizer.backward(loss)
        return loss, sample_size, logging_output

    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
        if self.cfg.eval_bleu:
            bleu = self._inference_with_bleu(self.sequence_generator, sample, model)
            logging_output["_bleu_sys_len"] = bleu.sys_len
            logging_output["_bleu_ref_len"] = bleu.ref_len
            # we split counts into separate entries so that they can be
            # summed efficiently across workers using fast-stat-sync
            assert len(bleu.counts) == EVAL_BLEU_ORDER
            for i in range(EVAL_BLEU_ORDER):
                logging_output["_bleu_counts_" + str(i)] = bleu.counts[i]
                logging_output["_bleu_totals_" + str(i)] = bleu.totals[i]
        return loss, sample_size, logging_output

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)
        if self.cfg.eval_bleu:

            def sum_logs(key):
                import torch

                result = sum(log.get(key, 0) for log in logging_outputs)
                if torch.is_tensor(result):
                    result = result.cpu()
                return result

            counts, totals = [], []
            for i in range(EVAL_BLEU_ORDER):
                counts.append(sum_logs("_bleu_counts_" + str(i)))
                totals.append(sum_logs("_bleu_totals_" + str(i)))

            if max(totals) > 0:
                # log counts as numpy arrays -- log_scalar will sum them correctly
                metrics.log_scalar("_bleu_counts", np.array(counts))
                metrics.log_scalar("_bleu_totals", np.array(totals))
                metrics.log_scalar("_bleu_sys_len", sum_logs("_bleu_sys_len"))
                metrics.log_scalar("_bleu_ref_len", sum_logs("_bleu_ref_len"))

                def compute_bleu(meters):
                    import inspect
                    import torch
                    try:
                        from sacrebleu.metrics import BLEU

                        comp_bleu = BLEU.compute_bleu
                    except ImportError:
                        # compatibility API for sacrebleu 1.x
                        import sacrebleu

                        comp_bleu = sacrebleu.compute_bleu

                    fn_sig = inspect.getfullargspec(comp_bleu)[0]
                    if "smooth_method" in fn_sig:
                        smooth = {"smooth_method": "exp"}
                    else:
                        smooth = {"smooth": "exp"}
                    bleu = comp_bleu(
                        correct=meters["_bleu_counts"].sum,
                        total=meters["_bleu_totals"].sum,
                        sys_len=meters["_bleu_sys_len"].sum if torch.is_tensor(meters["_bleu_sys_len"].sum) == False else meters["_bleu_sys_len"].sum.long().item(),
                        ref_len=meters["_bleu_ref_len"].sum if torch.is_tensor(meters["_bleu_ref_len"].sum) == False else meters["_bleu_ref_len"].sum.long().item(),
                        **smooth,
                    )
                    return round(bleu.score, 2)

                metrics.log_derived("bleu", compute_bleu)

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.cfg.max_source_positions, self.cfg.max_target_positions)

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.src_dict

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.tgt_dict

    def _inference_with_bleu(self, generator, sample, model):
        import sacrebleu

        def decode(toks, escape_unk=False):
            s = self.tgt_dict.string(
                toks.int().cpu(),
                self.cfg.eval_bleu_remove_bpe,
                # The default unknown string in fairseq is `<unk>`, but
                # this is tokenized by sacrebleu as `< unk >`, inflating
                # BLEU scores. Instead, we use a somewhat more verbose
                # alternative that is unlikely to appear in the real
                # reference, but doesn't get split into multiple tokens.
                unk_string=("UNKNOWNTOKENINREF" if escape_unk else "UNKNOWNTOKENINHYP"),
            )
            if self.tokenizer:
                s = self.tokenizer.decode(s)
            return s

        gen_out = self.inference_step(generator, [model], sample, prefix_tokens=None)
        hyps, refs = [], []
        for i in range(len(gen_out)):
            hyps.append(decode(gen_out[i][0]["tokens"]))
            refs.append(
                decode(
                    utils.strip_pad(sample["target"][i], self.tgt_dict.pad()),
                    escape_unk=True,  # don't count <unk> as matches to the hypo
                )
            )
        if self.cfg.eval_bleu_print_samples:
            logger.info("example hypothesis: " + hyps[0])
            logger.info("example reference: " + refs[0])
        if self.cfg.eval_tokenized_bleu:
            return sacrebleu.corpus_bleu(hyps, [refs], tokenize="none")
        else:
            return sacrebleu.corpus_bleu(hyps, [refs])
