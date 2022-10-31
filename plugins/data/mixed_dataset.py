# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from fairseq.data.fairseq_dataset import FairseqDataset
from fairseq.tasks.translation import load_langpair_dataset

class MixedDataset(FairseqDataset):
    def __init__(self, raw_data_path, kd_data_path, src, tgt, src_dict, tgt_dict, combine, cfg):
        self.raw_data = load_langpair_dataset(
            raw_data_path, "train",
            src, src_dict, tgt, tgt_dict,
            combine=combine,
            dataset_impl=cfg.dataset_impl,
            upsample_primary=cfg.upsample_primary,
            left_pad_source=cfg.left_pad_source,
            left_pad_target=cfg.left_pad_target,
            max_source_positions=cfg.max_source_positions,
            max_target_positions=cfg.max_target_positions,
            prepend_bos=True,
        )
        self.kd_data = load_langpair_dataset(
            kd_data_path, "train",
            src, src_dict, tgt, tgt_dict,
            combine=combine,
            dataset_impl=cfg.dataset_impl,
            upsample_primary=cfg.upsample_primary,
            left_pad_source=cfg.left_pad_source,
            left_pad_target=cfg.left_pad_target,
            max_source_positions=cfg.max_source_positions,
            max_target_positions=cfg.max_target_positions,
            prepend_bos=True,
        )
        assert len(self.raw_data) == len(self.kd_data)
        
    def __getitem__(self, index):
        raw = self.raw_data[index]
        kd = self.kd_data[index]
        return raw, kd
    
    def __len__(self):
        return len(self.raw_data)

    def collater(self, samples, pad_to_length=None):
        raw_sample = [sample[0] for sample in samples]
        kd_sample = [sample[1] for sample in samples]
        raw_res = self.raw_data.collater(raw_sample, pad_to_length)
        kd_res = self.kd_data.collater(kd_sample, pad_to_length)
        if "target" in kd_res:
            raw_res["kd_target"] = kd_res["target"]
            raw_res["kd_prev_output_tokens"] = kd_res["net_input"]["prev_output_tokens"]
        return raw_res
    
    def size(self, index):
        return self.raw_data.size(index)

    def num_tokens(self, index):
        return self.raw_data.num_tokens(index)

    def num_tokens_vec(self, indices):
        return self.raw_data.num_tokens_vec(indices)

    def get_batch_shapes(self):
        return self.raw_data.get_batch_shapes()

    def ordered_indices(self):
        return self.raw_data.ordered_indices()

    @property
    def supports_prefetch(self):
        return self.raw_data.supports_prefetch

    def prefetch(self, indices):
        self.raw_data.prefetch(indices)

    def filter_indices_by_size(self, indices, max_sizes):
        return self.raw_data.filter_indices_by_size(indices, max_sizes)
