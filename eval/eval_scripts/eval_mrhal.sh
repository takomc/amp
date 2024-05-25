
export CUDA_VISIBLE_DEVICES=0
export CRYPTOGRAPHY_OPENSSL_NO_LEGACY=1

MODEL_BASE=[Path to Base Model]
MODEL_QLORA_BASE=[Path to QLORA Model]
MODEL_SUFFIX=[name of result file]
DATA_DIR=[Path to mmrhal image]

python eval/gen/eval_mrhal.py \
    --model-path ${MODEL_BASE}/ \
    --use-qlora True --qlora-path ${MODEL_QLORA_BASE} \
    --temperature 0.0 \
    --answers-file [Path to eval_gt.json] \
    --save-file ./eval/results/gen/mrhal/${MODEL_SUFFIX}.json \
    --image_aspect_ratio pad --test-prompt '' \
    --image_folder ${DATA_DIR} 

python eval/eval/eval_gpt_mrhal.py \
    --response ./eval/results/gen/mrhal/${MODEL_SUFFIX}.json \
    --evaluation ./eval/results/gpt/mrhal/${MODEL_BASE}.json \
    --api-key [Your GPT4 API Key] \
    --gpt-model gpt-4

python eval/eval/summarize_gpt_mrhal.py \
    --evaluation ./eval/results/gpt/mrhal/${MODEL_BASE}.json \