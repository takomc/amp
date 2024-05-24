import copy
import os
from os.path import exists, join, isdir
from typing import Optional, Dict, Sequence
import numpy as np
import bitsandbytes as bnb
import pandas as pd
from packaging import version
from packaging.version import parse
import argparse
import logging

import torch
from torch.nn.utils.rnn import pad_sequence
import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
    Seq2SeqTrainer,
    BitsAndBytesConfig,
    LlamaTokenizer

)

from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    PeftModel
)
from peft.tuners.lora import LoraLayer

from llava.model import LlavaLlamaForCausalLM

class DisableLogger:
    def __enter__(self):
        logging.disable(logging.CRITICAL)

    def __exit__(self, exit_type, exit_value, exit_traceback):
        logging.disable(logging.NOTSET)

def get_accelerate_model(args, tokenizer, need_peft=False, checkpoint_dir=None):

    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        
   
    device_map = "auto"

    # if we are in a distributed setting, we need to set the device map and max memory per device
    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device_map = {'': local_rank}

    if args.full_finetune: 
        assert args.bits in [16, 32]

    print(f'loading base model {args.base_model_name}...')
    compute_dtype = (torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))
    with DisableLogger():
        model = LlavaLlamaForCausalLM.from_pretrained(
            args.base_model_name,
            cache_dir=args.cache_dir,
            load_in_4bit=args.bits == 4,
            load_in_8bit=args.bits == 8,
            device_map=device_map,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=args.bits == 4,
                load_in_8bit=args.bits == 8,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=args.double_quant,
                bnb_4bit_quant_type=args.quant_type,
            ),
            torch_dtype=(torch.float32 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)),
            trust_remote_code=args.trust_remote_code,
        )
    if compute_dtype == torch.float16 and args.bits == 4:
        if torch.cuda.is_bf16_supported():
            print('='*80)
            print('Your GPU supports bfloat16, you can accelerate training with the argument --bf16')
            print('='*80)


    setattr(model, 'model_parallel', True)
    setattr(model, 'is_parallelizable', True)

    model.config.torch_dtype=(torch.float32 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))

    # ref_model
    if need_peft:

        if args.vision_tower is not None:
            model.config.image_aspect_ratio = args.image_aspect_ratio
            model.config.image_grid_pinpoints = args.image_grid_pinpoints

            vision_tower = model.get_vision_tower()
            if not vision_tower.is_loaded:
                vision_tower.load_model()
            vision_tower.to(device="cuda", dtype=torch.bfloat16)
            vision_tower.requires_grad_(False)

            mm_projector = model.get_model().mm_projector
            mm_projector.to(device="cuda", dtype=torch.bfloat16)
            mm_projector.requires_grad_(False)
            model.initialize_vision_tokenizer(args, tokenizer=tokenizer)

        return model, tokenizer
    
    if not args.full_finetune:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)

    if not args.full_finetune:
        if checkpoint_dir is not None:
            print("Loading adapters from checkpoint.")
            model = PeftModel.from_pretrained(model, join(checkpoint_dir, 'adapter_model'), is_trainable=True)
        else:
            print(f'adding LoRA modules...')
            modules = ['gate_proj', 'up_proj', 'k_proj', 'down_proj', 'v_proj', 'q_proj', 'o_proj']
            print(f'found {modules} modules to add LoRA to')
            config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                target_modules=modules,
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, config)

    if args.vision_tower is not None:
        model.config.image_aspect_ratio = args.image_aspect_ratio
        model.config.image_grid_pinpoints = args.image_grid_pinpoints

        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        vision_tower.to(device="cuda", dtype=torch.bfloat16)
        vision_tower.requires_grad_(False)

        mm_projector = model.get_model().mm_projector
        mm_projector.to(device="cuda", dtype=torch.bfloat16)
        mm_projector.requires_grad_(False)

    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            if args.bf16:
                module = module.to(torch.bfloat16)
        if "lm_head" in name or "embed_tokens" in name:
            if hasattr(module, "weight"):
                if args.bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)
                    
    return model, tokenizer


def print_trainable_parameters(args, model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    if args.bits == 4: trainable_params /= 2
    print(
        f"trainable params: {trainable_params} || "
        f"all params: {all_param} || "
        f"trainable: {100 * trainable_params / all_param}"
    )

def make_models(
    args,
    tokenizer: transformers.PreTrainedTokenizer,
    accelerator,
    num_new_tokens: int = 0,
    resume_from_checkpoint: Optional[str] = None,
) -> dict:

    policy_resume_path = None

    model, tokenizer = get_accelerate_model(args, tokenizer)
    
    ref_model, _ = get_accelerate_model(args, tokenizer, need_peft=True)

    return dict(model=model, ref_model=ref_model, tokenizer=tokenizer)