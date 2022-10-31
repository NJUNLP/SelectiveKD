#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
from os.path import defpath
from pathlib import Path
import shutil
from tempfile import NamedTemporaryFile
from typing import Optional, Tuple, List
import sentencepiece as sp
import os
import csv

import pandas as pd
import torchaudio
from examples.speech_to_text.data_utils import (
    create_zip,
    extract_fbank_features,
    filter_manifest_df,
    gen_config_yaml,
    get_zip_manifest,
    load_df_from_tsv,
    save_df_to_tsv,
)
from torch import Tensor
from torch.utils.data import Dataset
from torchaudio.datasets.utils import download_url, extract_archive
from tqdm import tqdm


log = logging.getLogger(__name__)

class CoVoST(Dataset):
    """Create a Dataset for CoVoST (https://github.com/facebookresearch/covost).

    Args:
        root (str): root path to the dataset and generated manifests/features
        source_language (str): source (audio) language
        target_language (str, optional): target (text) language,
        None for no translation (default: None)
        version (int, optional): CoVoST version. (default: 2)
        download (bool, optional): Whether to download the dataset if it is not
        found at root path. (default: ``False``).
    """

    LANGUAGES=["fr", "de", "es", "ca", "it", "ru", "zh-CN", "pt", "fa", "et", "mn", 
               "nl", "tr", "ar", "sv-SE", "lv", "sl", "ta", "ja", "id", "cy", "en"]

    def __init__(
        self,
        root: str,
        language: str
    ) -> None:
        assert language in self.LANGUAGES

        self.root = Path(root)
        cv_tsv_path = self.root / "validated.tsv"
        assert cv_tsv_path.is_file()
        df = load_df_from_tsv(cv_tsv_path)
        
        data = df.to_dict(orient="index").items()
        data = [v for k, v in sorted(data, key=lambda x: x[0])]
        self.data = []
        for e in data:
            try:
                path = self.root / "clips" / e["path"]
                _ = torchaudio.info(path.as_posix())
                self.data.append(e)
            except RuntimeError:
                pass

    def __getitem__(
        self, n: int
    ) -> Tuple[Tensor, int, str]:
        data = self.data[n]
        path = self.root / "clips" / data["path"]
        waveform, sample_rate = torchaudio.load(path)
        _id = data["path"].replace(".mp3", "")
        return waveform, sample_rate, _id

    def __len__(self) -> int:
        return len(self.data)

class Europarl_ST(Dataset):
    SPLITS = ["train", "dev", "test"]
    LANGUAGES = ["de", "en", "es", "fr", "it", "nl", "pl", "pt", "ro"]

    def __init__(
        self,
        root: str,
        lang: str
    ) -> None:
        assert lang in self.LANGUAGES
        
        self.root: Path = Path(root)
        self.audio_path = self.root / "audios"

        def get_df(lang, split):
            text_path = self.root / lang / split
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
            return df

        df = []
        for split in self.SPLITS:
            for lang2 in self.LANGUAGES:
                if lang2 != lang:
                    df.append(get_df(lang2, split))

        df = pd.concat(df, axis=0)
        df = df.drop_duplicates(subset=("path", "start", "end"))
        df.reset_index(drop=True, inplace=True)

        data = df.to_dict(orient="index").items()
        data = [v for k, v in sorted(data, key=lambda x: x[0])]
        self.data = []
        for e in data:
            try:
                path = self.audio_path / (e["path"] + ".mp3")
                _ = torchaudio.info(path.as_posix())
                self.data.append(e)
            except RuntimeError:
                pass


    def __getitem__(
        self, n: int
    ) -> Tuple[Tensor, int, str]:
        """Load the n-th sample from the dataset.
        Args:
            n (int): The index of the sample to be loaded
        Returns:
            tuple: ``(waveform, sample_rate, sample_id)``
        """
        data = self.data[n]
        path = self.audio_path / (data["path"] + ".mp3")
        waveform, sample_rate = torchaudio.load(path)
        wave_length = waveform.shape[-1]
        start = min(max(int(sample_rate * data["start"]), 0), wave_length - 1)
        end = min(max(int(sample_rate * data["end"]), 1), wave_length)
        waveform = waveform[:, start:end]
        _id = data["path"] + "_{:.2f}_{:.2f}".format(data["start"], data["end"])
        return waveform, sample_rate, _id

    def __len__(self) -> int:
        return len(self.data)


def extract_features(args):
    lang = args.lang

    root = Path(args.data_root) / lang
    zip_path = root / f"fbank80.{args.dataset}.{lang}.zip"
    feature_root = root / f"fbank80.{args.dataset}.{lang}"

    # Extract features
    feature_root.mkdir(exist_ok=True)

    if args.dataset == "covost2":
        dataset = CoVoST(root, lang)
    elif args.dataset == "europarl-st":
        dataset = Europarl_ST(root, lang)
    else:
        assert False, f"Dataset {args.dataset} not supported."
    print(f"Extracting log mel filter bank features for language {lang}...")
    for waveform, sample_rate, utt_id in tqdm(dataset):
        extract_fbank_features(
            waveform, sample_rate, feature_root / f"{utt_id}.npy"
        )

    # Pack features into ZIP
    print("ZIPing features...")
    create_zip(feature_root, zip_path)

    # Clean up
    if os.path.exists(feature_root):
        shutil.rmtree(feature_root)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root", "-d", required=True, type=str,
        help="data root with sub-folders for each language <root>/<src_lang>"
    )
    parser.add_argument(
        "--lang", "-s", required=True, type=str,
        help="language to process"
    )
    parser.add_argument(
        "--dataset", default="covost2", type=str
    )
    args = parser.parse_args()

    extract_features(args)


if __name__ == "__main__":
    main()


