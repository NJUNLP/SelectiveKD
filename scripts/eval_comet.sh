#!/bin/bash
# example: bash scripts/eval_comet.sh en-de glat_ctc
# glat_ctc glat_ctc_skd; at ds; cmlm cmlm_skd; vanilla vanill_skd; 

COMET=./comet
RES=$COMET/generate-test.txt

rm $RES

if [ ! -f $COMET/generate-test.$1.$2.txt ]; then
    echo "Generating outputs..."
    # python3 fairseq_cli/generate.py ../datasets/wmt14.$1 \
    #     --user-dir plugins --gen-subset test \
    #     --task translation_selective_kd --remove-bpe --max-sentences 20 \
    #     --iter-decode-max-iter 0 --iter-decode-eos-penalty 0 --iter-decode-with-beam 1 \
    #     --path ../checkpoints/nat.glat_ctc_attn.wmt14.$1/avg_last_5_checkpoint.pt \
    #     --results-path $COMET

    python3 fairseq_cli/generate.py ../datasets/wmt14.$1 \
        --path ../checkpoints/at.ds_tiny.wmt14.$1/avg_last_5_checkpoint.pt \
        --batch-size 128 --beam 5 --remove-bpe --results-path $COMET

    # python3 fairseq_cli/generate.py ../datasets/wmt14.$1 \
    #     --user-dir plugins --gen-subset test \
    #     --task translation_selective_kd_standard --remove-bpe --max-sentences 20 \
    #     --results-path $COMET --iter-decode-max-iter 0 --iter-decode-eos-penalty 0 --iter-decode-with-beam 1 \
    #     --path ../checkpoints/nat.vanilla_skd.wmt14.$1/avg_last_5_checkpoint.pt

    # python3 fairseq_cli/generate.py ../datasets/wmt14.$1 \
    #     --user-dir plugins --gen-subset test \
    #     --task translation_selective_kd_standard --remove-bpe --max-sentences 20 \
    #     --results-path $COMET --iter-decode-max-iter 10 --iter-decode-eos-penalty 0 --iter-decode-with-beam 1 \
    #     --path ../checkpoints/nat.cmlm_skd.wmt14.$1/avg_last_5_checkpoint.pt

    cp $RES $COMET/generate-test.$1.$2.txt
fi

cat $COMET/generate-test.$1.$2.txt | grep '^T-[0-9]*' | sed -e 's/T-[0-9]*\s*//g' > $COMET/test.ref
cat $COMET/generate-test.$1.$2.txt | grep '^S-[0-9]*' | sed -e 's/S-[0-9]*\s*[-0-9.]*\s*//g' > $COMET/test.src
cat $COMET/generate-test.$1.$2.txt | grep '^D-[0-9]*' | sed -e 's/D-[0-9]*\s*[-0-9.]*\s*//g' > $COMET/test.tgt

cat $COMET/test.ref | wc -l
cat $COMET/test.src | wc -l
cat $COMET/test.tgt | wc -l

echo ">>> Evaluating..."
python3 scripts/eval_comet.py $COMET/test.src $COMET/test.tgt $COMET/test.ref

