#!/bin/bash

SPLIT="mmbench_dev_20230712"

python -m llava.eval.model_vqa_mmbench \
    --model-path liuhaotian/llava-v1.5-7b-lora \
    --model-base lmsys/vicuna-7b-v1.5 \
    --question-file /egr/research-optml/chenyiw9/datasets/llava-eval/mmbench/$SPLIT.tsv \
    --answers-file /egr/research-optml/chenyiw9/datasets/llava-eval/mmbench/answers/$SPLIT/llava-v1.5-7b-lora.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p playground/data/eval/mmbench/answers_upload/$SPLIT

python scripts/convert_mmbench_for_submission.py \
    --annotation-file /egr/research-optml/chenyiw9/datasets/llava-eval/mmbench/$SPLIT.tsv \
    --result-dir /egr/research-optml/chenyiw9/datasets/llava-eval/mmbench/answers/$SPLIT \
    --upload-dir /egr/research-optml/chenyiw9/datasets/llava-eval/mmbench/answers_upload/$SPLIT \
    --experiment llava-v1.5-7b-lora
