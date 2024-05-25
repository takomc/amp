
export CUDA_VISIBLE_DEVICES=0
export CRYPTOGRAPHY_OPENSSL_NO_LEGACY=1

MODEL_BASE=[Path to Base Model]
MODEL_QLORA_BASE=[Path to QLORA Model]
MODEL_SUFFIX=[name of result file]

python eval/gen/eval_mmhal.py \
    --model-path ${MODEL_BASE}/ \
    --use-qlora True --qlora-path ${MODEL_QLORA_BASE} \
    --temperature 0.0 \
    --answers-file \
    ./eval/results/gen/mmhal/${MODEL_SUFFIX}.json --image_aspect_ratio pad --test-prompt '' 

python eval/eval/eval_gpt_mmhal.py \
    --response ./eval/results/gen/mmhal/${MODEL_SUFFIX}.json \
    --evaluation ./eval/results/gpt/mmhal/${MODEL_SUFFIX}.json \
    --api-key [Your GPT4 API Key] \
    --gpt-model gpt-4

python eval/eval/summarize_gpt_mmhal.py \
    --evaluation ./eval/results/gpt/mmhal/${MODEL_SUFFIX}.json