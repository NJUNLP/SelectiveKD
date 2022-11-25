# Selective KD
Implementation for our AAAI-23 paper "Selective Knowledge Distillation for Non-Autoregressive Neural Machine Translation".

### Preparation
Train an autoregressive Transformer according to the instructions in [Fairseq](https://github.com/pytorch/fairseq).

Use the trained autoregressive Transformer to generate target sentences for the training set.

You should make sure that the order of source sentences in distilled data matches that in raw data.

Then binarize raw and distilled data respectively.

```
python3 fairseq_cli/preprocess.py \
    --source-lang ${SRC} --target-lang ${TGT} \
    --trainpref ${INPUT_DIR}/train --validpref ${INPUT_DIR}/valid --testpref ${INPUT_DIR}/test \
    --destdir ${DATA_DIR} \
    --srcdict ${INPUT_DIR}/dict.${SRC}.txt --tgtdict ${INPUT_DIR}/dict.${TGT}.txt
```

### Train an Evaluator
```
python3 fairseq_cli/train.py ${KD_DATA_DIR} \
    --arch glat_ctc --noise full_mask --share-all-embeddings \
    --criterion ctc_loss --label-smoothing 0.1 --lr 5e-4 --warmup-init-lr 1e-7 --stop-min-lr 1e-9 \
    --lr-scheduler inverse_sqrt --warmup-updates 4000 --optimizer adam --adam-betas '(0.9,0.999)' \
    --adam-eps 1e-6 --task translation_lev_modified --max-tokens 8192 --weight-decay 0.01 --dropout 0.1 \
    --encoder-layers 6 --encoder-embed-dim 512 --decoder-layers 6 --decoder-embed-dim 512 --fp16 \
    --max-source-positions 1000 --max-target-positions 1000 --max-update 100000 --seed 0 --clip-norm 2\
    --length-loss-factor 0 --log-interval 100 \
    --eval-bleu --eval-bleu-args '{"iter_decode_max_iter":0,"iter_decode_with_beam":1}' \
    --eval-tokenized-bleu --eval-bleu-remove-bpe --best-checkpoint-metric bleu \
    --maximize-best-checkpoint-metric --decoder-learned-pos --encoder-learned-pos \
    --apply-bert-init --activation-fn gelu --user-dir plugins --keep-best-checkpoints 5 \
    --save-dir ${EVALUATOR_DIR}
```

### Train a Model Using Selective KD
```
python3 fairseq_cli/train.py ${RAW_DATA_DIR} --kd-data ${KD_DATA_DIR} \
    --arch glat_ctc --noise full_mask --share-all-embeddings --num-workers 4 \
    --criterion ctc_loss --label-smoothing 0.1 --lr 5e-4 --warmup-init-lr 1e-7 --stop-min-lr 1e-9 \
    --lr-scheduler inverse_sqrt --warmup-updates 4000 --optimizer adam --adam-betas '(0.9,0.999)' \
    --adam-eps 1e-6 --task translation_selective_kd --max-tokens 8192 --weight-decay 0.01 --dropout 0.1 \
    --encoder-layers 6 --encoder-embed-dim 512 --decoder-layers 6 --decoder-embed-dim 512 --fp16 \
    --max-source-positions 1000 --max-target-positions 1000 --max-update 300000 --seed 0 --clip-norm 2\
    --length-loss-factor 0 --log-interval 100 \
    --eval-bleu --eval-bleu-args '{"iter_decode_max_iter":0,"iter_decode_with_beam":1}' \
    --eval-tokenized-bleu --eval-bleu-remove-bpe --best-checkpoint-metric bleu \
    --maximize-best-checkpoint-metric --decoder-learned-pos --encoder-learned-pos \
    --apply-bert-init --activation-fn gelu --user-dir plugins \
    --keep-best-checkpoints 5 --save-dir ${MODEL_DIR} \
    --evaluator ${EVALUATOR_PATH} --load-from-checkpoint ${INITIALIZATION_MODEL} \
    --kd-threshold 0.4 --kd-threshold-offset 0.6
```

### Inference
```
python3 fairseq_cli/generate.py ${RAW_DATA_DIR} \
    --user-dir plugins --gen-subset test \
    --task translation_selective_kd --remove-bpe --max-sentences 20 \
    --iter-decode-max-iter 0 --iter-decode-eos-penalty 0 --iter-decode-with-beam 1 \
    --path ${CHECKPOINT_PATH} --quiet
```

