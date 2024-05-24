# Copyright 2024 The AMP Team
# Inspired by LLaVA-RLHF, LLaVA-V1.5, and TRL

from torch.utils.data import Dataset
import torch

import dataclasses
from typing import Dict, List, Sequence
import logging, json
import transformers

from PIL import Image
import copy
import os

from llava.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava import conversation as conversation_lib
from llava.model import *
from llava.mm_utils import tokenizer_image_token
from data_utils.constants import *

logger = logging.getLogger(__name__)

# Datasets for muli-level preference learning
class LazySupervisedDataset(Dataset):

    def __init__(self,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args):
        super(LazySupervisedDataset, self).__init__()

        list_data_dict = json.load(open(data_args.data_path, "r"))

        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.preference_label = source_keys[data_args.num_levels]
        self.num_levels = data_args.num_levels

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if 'image' in sources[0]:
            image_file = self.list_data_dict[i]['image']
            image_folder = self.data_args.image_folder
            processor = self.data_args.image_processor
            image = Image.open(image_file).convert('RGB')
            if self.data_args.image_aspect_ratio == 'pad':
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result
                image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            else:
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                
            # add <image token> to str
            temp_sources = [[] for _ in range(self.num_levels)]
            for e in sources:
                for i in range(1, self.num_levels+1):
                    temp_sources[i-1].append(e[f"output_{i}"])

            sources = [preprocess_multimodal(copy.deepcopy(temp_source), self.data_args)   
                                for temp_source in temp_sources]

            _sources = {}
            for label, source in zip(self.preference_label, sources):
                _sources[label] = source
        else:
            temp_sources = []
            for e in sources:
                for i in range(1, self.num_levels+1):
                    temp_sources[i-1].append(e[f"output_{i}"])

            _sources = {}
            for label, source in zip(self.preference_label, temp_sources):
                _sources[label] = source

        data_dict_all = [
            preprocess_v1(
                _sources[key],
                self.tokenizer,
                has_image=('image' in self.list_data_dict[i])
                )
                for key in self.preference_label
            ]
        
        labels, input_idss, loss_turbos = {}, {}, {}

        for label, input_ids, loss_turbo, data_dict in \
            zip(source_labels[self.num_levels], source_input_idss[self.num_levels], source_loss_turbos[self.num_levels], data_dict_all):
            labels[label] = data_dict["labels"][0]
            input_idss[input_ids] = data_dict["input_ids"][0]
            loss_turbos[loss_turbo] = data_dict["loss_turbo"][0]
        # data_dict = {**labels, **input_idss, **loss_turbos, "input_ids": data_dict_all[0]["input_ids"][0]}
        data_dict = {**labels, **input_idss, **loss_turbos}

        # image exist in the data
        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
        return data_dict


@dataclasses.dataclass
class DataCollatorForQueryResponseDataset(object):
    
    tokenizer: transformers.PreTrainedTokenizer
    data_args: None
    def _right_pad_helper(self, instances: Sequence[dict], key: str):
        input_ids = [instance[key] for instance in instances] 
        if 'input_ids' in key:
            input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids,
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id,
            )
        else:
            input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids,
                batch_first=True,
                padding_value=IGNORE_INDEX,
            )

        return input_ids
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:

        all_labels = tuple(
            self._right_pad_helper(instances, key)
            for key in source_labels[self.data_args.num_levels]
        )

        all_input_idss = tuple(
            self._right_pad_helper(instances, key)
            for key in source_input_idss[self.data_args.num_levels]
        )

        all_loss_turbos = tuple(
            self._right_pad_helper(instances, key)
            for key in source_loss_turbos[self.data_args.num_levels]
        )
        
        all_attention_masks = tuple(
            all_input_ids.ne(self.tokenizer.pad_token_id).long()
            for all_input_ids in all_input_idss
        )
        # chosen_attention_mask = chosen_input_ids.ne(self.tokenizer.pad_token_id).long()
        # med_0_attention_mask = med_0_input_ids.ne(self.tokenizer.pad_token_id).long()
        # med_1_attention_mask = med_1_input_ids.ne(self.tokenizer.pad_token_id).long()
        # rejected_attention_mask = rejected_input_ids.ne(self.tokenizer.pad_token_id).long()
        dict_labels, dict_input_idss, dict_loss_turbos, dict_attention_masks = {}, {}, {}, {}

        for pre_label, pre_input_ids, pre_loss_turbo, pre_attention_mask, label, input_ids, loss_turbo, attention_mask in \
            zip(source_labels[self.data_args.num_levels], source_input_idss[self.data_args.num_levels], source_loss_turbos[self.data_args.num_levels], source_attention_masks[self.data_args.num_levels], \
                            all_labels, all_input_idss, all_loss_turbos, all_attention_masks):

            dict_labels[pre_label] = label
            dict_input_idss[pre_input_ids] = input_ids
            dict_loss_turbos[pre_loss_turbo] = loss_turbo
            dict_attention_masks[pre_attention_mask] = attention_mask

        data_dict = {**dict_labels, **dict_input_idss, **dict_loss_turbos, **dict_attention_masks}
        data_dict["images"] = torch.stack([instance['image'] for instance in instances])

        return data_dict

def make_rl_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_args
):

    train_dataset = LazySupervisedDataset(
        tokenizer=tokenizer,
        data_args=data_args,
    )
    
    return dict(
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=DataCollatorForQueryResponseDataset(tokenizer, data_args),
    )

def preprocess_v1(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()
    loss_turbo = torch.zeros(targets.shape, dtype=torch.float16, device=targets.device)

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())
        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        loss_index = 0
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            # Designed specificallyfor multi-round dialogues. We make the weights for latter rounds bigger.
            loss_turbo[0][cur_len + instruction_len : cur_len + round_len] = loss_index
            loss_index = loss_index + 0.05
            cur_len += round_len

        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
        loss_turbo=loss_turbo
    )

def preprocess_multimodal(
    sources: Sequence[str],
    data_args
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources
