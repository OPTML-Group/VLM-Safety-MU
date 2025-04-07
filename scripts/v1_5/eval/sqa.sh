#!/bin/bash

python -m llava.eval.model_vqa_science \
    --model-path /egr/research-optml/chenyiw9/projects/LLaVA/checkpoints-img-token/llava-v1.5-7b-lora-v7 \
    --model-base lmsys/vicuna-7b-v1.5 \
    --question-file /egr/research-optml/chenyiw9/datasets/llava-eval/scienceqa/llava_test_CQM-A.json \
    --image-folder /egr/research-optml/chenyiw9/datasets/llava-eval/scienceqa/images/test \
    --answers-file /egr/research-optml/chenyiw9/datasets/llava-eval/scienceqa/answers-img-token/llava-v1.5-7b-lora-v7.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_science_qa.py \
    --base-dir /egr/research-optml/chenyiw9/datasets/llava-eval/scienceqa \
    --result-file /egr/research-optml/chenyiw9/datasets/llava-eval/scienceqa/answers-img-token/llava-v1.5-7b-lora-v7.jsonl \
    --output-file /egr/research-optml/chenyiw9/datasets/llava-eval/scienceqa/answers-img-token/llava-v1.5-7b-lora-v7_output.jsonl \
    --output-result /egr/research-optml/chenyiw9/datasets/llava-eval/scienceqa/answers-img-token/llava-v1.5-7b-lora-v7_result.json
