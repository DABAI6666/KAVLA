CKPT="KAVLA-LLaVA"
STEP="epoch"
SPLIT="twitter"

# 运行虚假新闻检测的模型推理脚本
python -m llava.eval.model_rumor_loader \
      --model-base lmsys/vicuna-7b-v1.5 \
      --model-path checkpoints/$CKPT/checkpoint-$STEP \
      --question-file ./playground/rumor_detection/eval/$SPLIT.jsonl \
      --image-folder / \
      --answers-file ./playground/rumor_detection/eval/answers/$SPLIT/$CKPT/$STEP.jsonl \
      --temperature 0 \
      --conv-mode plain &

wait


