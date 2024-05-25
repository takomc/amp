
export CUDA_VISIBLE_DEVICES=0
export CRYPTOGRAPHY_OPENSSL_NO_LEGACY=1
MODEL_BASE=[Path to Base Model]
MODEL_QLORA_BASE=[Path to QLORA Model]
MODEL_SUFFIX=[name of result file]

python eval/gen/eval_pope.py \
    --model-path ${MODEL_BASE}/ \
    --use-qlora True --qlora-path ${MODEL_QLORA_BASE} \
    --image-folder [Path to COCO Image] \   # e.g.: .../coco/val2014
    --question-file [Path to llava_pope_test.jsonl] \
    --image_aspect_ratio pad \
    --answers-file ./eval/results/gen/pope/${MODEL_SUFFIX}.jsonl \
    --temperature 0. \

python eval/eval/eval_pope.py \
    --annotation-dir [Path to POPE GT] \
    --question-file [Path to llava_pope_test.jsonl] \
    --result-file ./eval/results/gen/pope/${MODEL_SUFFIX}.jsonl \
