#!/bin/bash

python -m llava.eval.model_vqa_loader \
    --model-path /egr/research-optml/chenyiw9/projects/LLaVA/checkpoints-img-token/llava-v1.5-7b-lora-v7 \
    --model-base lmsys/vicuna-7b-v1.5 \
    --question-file /egr/research-optml/chenyiw9/datasets/llava-eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder /egr/research-optml/chenyiw9/datasets/llava-eval/textvqa/train_images \
    --answers-file /egr/research-optml/chenyiw9/datasets/llava-eval/textvqa/img-token-answer/llava-v1.5-7b-lora-v7.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python -m llava.eval.eval_textvqa \
    --annotation-file /egr/research-optml/chenyiw9/datasets/llava-eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file /egr/research-optml/chenyiw9/datasets/llava-eval/textvqa/img-token-answer/llava-v1.5-7b-lora-v7.jsonl 
