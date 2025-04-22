python -m llava.eval.model_vqa_loader \
    --model-path /egr/research-optml/chenyiw9/projects/LLaVA/checkpoints/llava-v1.5-13b-lora-safe \
    --model-base lmsys/vicuna-13b-v1.5 \
    --question-file /egr/research-optml/chenyiw9/projects/SPA/data/help_share.jsonl \
    --image-folder  /egr/research-optml/chenyiw9/projects/SPA/ \
    --answers-file /egr/research-optml/chenyiw9/projects/SPA/llava_results/help_share/llava-v1.5-13b-lora-safe.json \
    --temperature 0 \
    --conv-mode vicuna_v1
