#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
from pathlib import Path
from typing import Optional, Tuple

import csv
import pandas as pd
import torchaudio
from examples.speech_to_text.data_utils import (
    filter_manifest_df,
    gen_config_yaml,
    get_zip_manifest,
    save_df_to_tsv,
)
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm


log = logging.getLogger(__name__)

LANG_TOK = {
    "de": "LANG_TOK_DE",
    "en": "LANG_TOK_EN",
    "es": "LANG_TOK_ES",
    "fr": "LANG_TOK_FR",
    "it": "LANG_TOK_IT",
    "nl": "LANG_TOK_NL",
    "pl": "LANG_TOK_PL",
    "pt": "LANG_TOK_PT",
    "ro": "LANG_TOK_RO",
}


MANIFEST_COLUMNS = ["id", "audio", "n_frames", "tgt_text", "speaker"]


class Europarl_ST(Dataset):
    SPLITS = ["train", "dev", "test"]

    LANGUAGES = ["de", "en", "es", "fr", "it", "nl", "pl", "pt", "ro"]

    def __init__(
        self,
        root: str,
        split: str,
        source_language: str,
        target_language: Optional[str] = None,
    ) -> None:
        assert split in self.SPLITS
        assert source_language is not None
        assert source_language in self.LANGUAGES
        self.no_translation = target_language is None
        if not self.no_translation:
            assert target_language in self.LANGUAGES
        
        self.root = Path(root)

        def get_df(text_path, lang):
            df = pd.read_csv(
                text_path / "segments.lst",
                sep=" ",
                header=None,
                names=("path", "start", "end"),
                encoding="utf-8",
                escapechar="\\",
                quoting=csv.QUOTE_NONE,
                na_filter=False,
            )
            with open(text_path / f"segments.{lang}") as f:
                df["tgt_text"] = f.readlines()
            return df

        if not self.no_translation:     # ST
            df = get_df(self.root / target_language / split, target_language)
        else:                           # ASR
            df = []
            for lang in self.LANGUAGES:
                if lang != source_language:
                    df.append(get_df(self.root / lang / split, source_language))
            df = pd.concat(df, axis=0)
            df = df.drop_duplicates(subset=("path", "start", "end"))
            df.reset_index(drop=True, inplace=True)
        
        data = df.to_dict(orient="index").items()
        data = [v for k, v in sorted(data, key=lambda x: x[0])]
        self.data = []
        for e in data:
            try:
                path = self.root / "audios" / (e["path"] + ".mp3")
                _ = torchaudio.info(path.as_posix())
                self.data.append(e)
            except RuntimeError:
                pass

    def __getitem__(
        self, n: int
    ) -> Tuple[str, int, str]:
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            tuple: ``(tgt_text, speaker_id, sample_id)``
        """
        data = self.data[n]
        tgt_text = data["tgt_text"]
        _id = data["path"] + "_{:.2f}_{:.2f}".format(data["start"], data["end"])
        return tgt_text.strip(), -1, _id

    def __len__(self) -> int:
        return len(self.data)


def process(args):
    root = Path(args.data_root).absolute() / args.src_lang
    if not root.is_dir():
        raise NotADirectoryError(f"{root} does not exist")
    
    zip_path = root / f"fbank80.europarl-st.{args.src_lang}.zip"
    print("Fetching ZIP manifest...")
    audio_paths, audio_lengths = get_zip_manifest(zip_path)

    # Generate TSV manifest
    print("Generating manifest...")
    train_text = []
    task = f"asr_{args.src_lang}"
    if args.tgt_lang is not None:
        task = f"st_{args.src_lang}_{args.tgt_lang}"
    for split in Europarl_ST.SPLITS:
        manifest = {c: [] for c in MANIFEST_COLUMNS}
        dataset = Europarl_ST(root, split, args.src_lang, args.tgt_lang)
        for tgt_text, speaker_id, utt_id in tqdm(dataset):
            manifest["id"].append(utt_id)
            manifest["audio"].append(audio_paths[utt_id].replace(f"/root/fairseq/../datasets/europarl-st/{args.src_lang}", "../datasets/europarl-st"))
            manifest["n_frames"].append(audio_lengths[utt_id])
            manifest["tgt_text"].append(f"{LANG_TOK[args.src_lang]} "+tgt_text if args.tgt_lang is None else f"{LANG_TOK[args.tgt_lang]} "+tgt_text)
            manifest["speaker"].append(speaker_id)
        is_train_split = split.startswith("train")
        if is_train_split:
            train_text.extend(manifest["tgt_text"])
        df = pd.DataFrame.from_dict(manifest)
        df = filter_manifest_df(df, is_train_split=is_train_split)
        save_df_to_tsv(df, root / f"{split}_{task}.tsv")

    # Generate config YAML
    gen_config_yaml(
        root,
        spm_filename="spm_bpe64000.model",
        yaml_filename=f"config_{task}.yaml",
        specaugment_policy="lb",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root", "-d", required=True, type=str,
        help="data root with sub-folders for each language <root>/<src_lang>"
    )
    parser.add_argument("--src-lang", "-s", required=True, type=str)
    parser.add_argument("--tgt-lang", "-t", type=str)
    args = parser.parse_args()

    process(args)


if __name__ == "__main__":
    main()
