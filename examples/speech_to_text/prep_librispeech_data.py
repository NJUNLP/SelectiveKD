#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
from pathlib import Path
import shutil
from tempfile import NamedTemporaryFile
import soundfile as sf

import pandas as pd
from examples.speech_to_text.data_utils import (
    create_zip,
    extract_fbank_features,
    gen_config_yaml,
    gen_vocab,
    get_zip_manifest,
    save_df_to_tsv,
)
from torchaudio.datasets import LIBRISPEECH
from tqdm import tqdm
from fairseq.data.audio.audio_utils import get_waveform, convert_waveform


log = logging.getLogger(__name__)

SPLITS = [
    "train-clean-100",
    "train-clean-360",
    "train-other-500",
    "dev-clean",
    "dev-other",
    "test-clean",
    "test-other",
]

MANIFEST_COLUMNS = ["id", "audio", "n_frames", "src_text", "speaker", "src_lang"]


def process(args):
    out_root = Path(args.output_root).absolute()
    out_root.mkdir(exist_ok=True)
    # Extract features
    audio_root = out_root / "flac"
    audio_root.mkdir(exist_ok=True)
    for split in SPLITS:
        print(f"Fetching split {split}...")
        dataset = LIBRISPEECH(out_root.as_posix(), url=split, download=True)
        print("Converting audios...")
        for wav, sample_rate, _, spk_id, chapter_no, utt_no in tqdm(dataset):
            sample_id = f"{spk_id}-{chapter_no}-{utt_no}"
            tgt_sample_rate = 16_000
            _wavform, _ = convert_waveform(
                wav, sample_rate, to_mono=True,
                to_sample_rate=tgt_sample_rate
            )
            sf.write(
                (audio_root / f"{sample_id}.flac").as_posix(),
                _wavform.T.numpy(), tgt_sample_rate
            )

    # # Pack features into ZIP
    zip_path = out_root / "flac.zip"
    print("ZIPing features...")
    create_zip(audio_root, zip_path)
    print("Fetching ZIP manifest...")
    audio_paths, audio_lengths = get_zip_manifest(zip_path, is_audio=True)
    # Generate TSV manifest
    print("Generating manifest...")
    train_text = []
    for split in SPLITS:
        manifest = {c: [] for c in MANIFEST_COLUMNS}
        dataset = LIBRISPEECH(out_root.as_posix(), url=split)
        for _, _, utt, spk_id, chapter_no, utt_no in tqdm(dataset):
            sample_id = f"{spk_id}-{chapter_no}-{utt_no}"
            manifest["id"].append(sample_id)
            manifest["audio"].append(audio_paths[sample_id].replace("/opt/tiger/fairseq", ".."))
            manifest["n_frames"].append(audio_lengths[sample_id])
            manifest["src_text"].append(utt.lower())
            manifest["speaker"].append(spk_id)
            manifest["src_lang"].append("en")
        save_df_to_tsv(
            pd.DataFrame.from_dict(manifest), out_root / f"{split}.tsv"
        )
        if split.startswith("train"):
            train_text.extend(manifest["src_text"])
    # Generate vocab
    vocab_size = "" if args.vocab_type == "char" else str(args.vocab_size)
    spm_filename_prefix = f"spm_{args.vocab_type}{vocab_size}"
    with NamedTemporaryFile(mode="w") as f:
        for t in train_text:
            f.write(t + "\n")
        gen_vocab(
            Path(f.name),
            out_root / spm_filename_prefix,
            args.vocab_type,
            args.vocab_size,
        )

    # Generate config YAML
    gen_config_yaml(
        out_root,
        spm_filename=spm_filename_prefix + ".model",
        yaml_filename=f"config.yaml",
        specaugment_policy=None,
        extra={"use_audio_input": True}
    )
    # Clean up
    shutil.rmtree(audio_root)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", "-o", required=True, type=str)
    parser.add_argument(
        "--vocab-type",
        default="unigram",
        required=True,
        type=str,
        choices=["bpe", "unigram", "char"],
    ),
    parser.add_argument("--vocab-size", default=10000, type=int)
    args = parser.parse_args()

    process(args)


if __name__ == "__main__":
    main()
