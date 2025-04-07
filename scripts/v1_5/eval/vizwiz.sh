#!/bin/bash

python -m llava.eval.model_vqa_loader \
    --model-path /egr/research-optml/chenyiw9/projects/LLaVA/checkpoints-img-token/llava-v1.5-7b-lora-v7 \
    --model-base lmsys/vicuna-7b-v1.5 \
    --question-file /egr/research-optml/chenyiw9/datasets/llava-eval/vizwiz/llava_test.jsonl \
    --image-folder /egr/research-optml/chenyiw9/datasets/llava-eval/vizwiz/test \
    --answers-file /egr/research-optml/chenyiw9/datasets/llava-eval/vizwiz/answers-img-token/llava-v1.5-7b-lora-v7.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python scripts/convert_vizwiz_for_submission.py \
    --annotation-file /egr/research-optml/chenyiw9/datasets/llava-eval/vizwiz/llava_test.jsonl \
    --result-file /egr/research-optml/chenyiw9/datasets/llava-eval/vizwiz/answers-img-token/llava-v1.5-7b-lora-v7.jsonl \
    --result-upload-file /egr/research-optml/chenyiw9/datasets/llava-eval/vizwiz/answers-img-token_upload/llava-v1.5-7b-lora-v7.json
