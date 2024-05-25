
export CUDA_VISIBLE_DEVICES=0
export CRYPTOGRAPHY_OPENSSL_NO_LEGACY=1

MODEL_BASE=[Path to Base Model]
MODEL_QLORA_BASE=[Path to QLORA Model]
MODEL_SUFFIX=[name of result file]

python eval/gen/eval_llava.py \
    --model-path ${MODEL_BASE}/ \
    --use-qlora True --qlora-path ${MODEL_QLORA_BASE} \
    --image-folder [Path to llava image folder] \
    --question-file [Path to qa90_questions.jsonl] \
    --image_aspect_ratio pad \
    --answers-file ./eval/results/gen/llavab/${MODEL_SUFFIX}.jsonl \
    --temperature 0.


python eval/eval/eval_gpt_llavab.py \
    --question [Path to qa90_questions.jsonl] \
    --context [Path to caps_boxes_coco2014_val_80.jsonl] \
    --rule [Path to rule.json] \
    --answer-list \
        [Path to qa90_gpt4_answer.jsonl] \
        ./eval/results/gen/llavab/${MODEL_SUFFIX}.jsonl \
    --output \
        ./eval/results/gpt/llavab/${MODEL_SUFFIX}.jsonl \
    --openai_key [Your GPT4 API Key]

python eval/eval/summarize_gpt_llavab.py -f ./eval/results/gpt/llavab/${MODEL_SUFFIX}.jsonl
