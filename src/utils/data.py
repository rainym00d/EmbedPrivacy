from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from transformers import PreTrainedTokenizerBase


@dataclass
class DataCollatorWithEmbedding:
    tokenizer: PreTrainedTokenizerBase
    max_length: int
    label_padding_value: int = -100

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        embeddings = torch.tensor([f["embedding"] for f in features])
        labels = self.tokenize_label([f["text"] for f in features])

        batch = {"labels": labels, "inputs_embeds": embeddings}

        return batch

    def tokenize_label(self, text: List[str]):
        label_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False,
        )
        # * get labels from input ids
        labels = label_encoding.input_ids
        # * add eos token if length < max_length
        labels = add_eos_to_nested_lists(
            labels, self.max_length, self.tokenizer.eos_token_id
        )
        # * get cur_max_length to pad
        cur_max_length = get_max_length_in_nested_lists(labels)
        labels = pad_nested_lists(
            labels, cur_max_length, self.label_padding_value, "right"
        )

        return torch.tensor(labels)


def get_max_length_in_nested_lists(lst):
    if isinstance(lst[0], list):
        lengths = []
        for elem in lst:
            length = get_max_length_in_nested_lists(elem)
            lengths.append(length)
        max_length = max(lengths)
        return max_length
    else:
        return len(lst)


def add_eos_to_nested_lists(lst, max_length, eos_value):
    if isinstance(lst, list) and len(lst) and isinstance(lst[0], list):
        for i, elem in enumerate(lst):
            lst[i] = add_eos_to_nested_lists(elem, max_length, eos_value)
        return lst
    elif isinstance(lst, list):
        if len(lst) < max_length:
            return lst + [eos_value]
        return lst
    else:
        raise NotImplementedError(f"Unrecognized type {lst}")


def pad_nested_lists(lst, max_length, padding_value, padding_side):
    if isinstance(lst, list) and len(lst) and isinstance(lst[0], list):
        for i, elem in enumerate(lst):
            lst[i] = pad_nested_lists(elem, max_length, padding_value, padding_side)
        return lst
    elif isinstance(lst, list):
        if padding_side == "right":
            return lst + [padding_value for _ in range(max_length - len(lst))]
        else:
            return [padding_value for _ in range(max_length - len(lst))] + lst
    else:
        raise NotImplementedError(f"Unrecognized type {lst}")
