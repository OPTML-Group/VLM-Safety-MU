#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT="llava-v1.5-7b-lora-swin2"
SPLIT="llava_vqav2_mscoco_test-dev2015"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path /egr/research-optml/chenyiw9/projects/LLaVA/checkpoints-img-token/llava-v1.5-7b-lora-swin2 \
        --model-base lmsys/vicuna-7b-v1.5 \
        --question-file /egr/research-optml/chenyiw9/datasets/llava-eval/vqav2/$SPLIT.jsonl \
        --image-folder /egr/research-optml/chenyiw9/datasets/llava-eval/vqav2/test2015 \
        --answers-file /egr/research-optml/chenyiw9/datasets/llava-eval/vqav2/answers-img-token/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

output_file=/egr/research-optml/chenyiw9/datasets/llava-eval/vqav2/answers-img-token/$SPLIT/$CKPT/merge.jsonl

# Clear out the output file if it exists.
# > "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat /egr/research-optml/chenyiw9/datasets/llava-eval/vqav2/answers-img-token/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python scripts/convert_vqav2_for_submission.py --split $SPLIT --ckpt $CKPT

