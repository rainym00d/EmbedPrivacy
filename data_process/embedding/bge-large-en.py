import argparse
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import datasets
import torch
from transformers import (
    AutoTokenizer,
    BertModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

from torch import Tensor
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

@dataclass
class MyDataCollator:
    tokenizer: PreTrainedTokenizerBase

    def __call__(self, features):
        text = [f["text"] for f in features]
        inputs = self.tokenizer(
            text, padding=True, truncation=True, max_length=256, return_tensors="pt"
        )
        return inputs

def preprocess_logits_for_metrics(logits, labels):
    # return torch.nn.functional.normalize(logits[0][:, 0], p=2, dim=1)
    return logits[0][:, 0]
    # return logits[1]


def main():
    # * parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dataset", type=str, required=True)
    parser.add_argument("--output_dataset", type=str, required=True)
    parser.add_argument("--train_size", type=int, required=True)
    parser.add_argument("--valid_size", type=int, required=True)
    parser.add_argument("--test_size", type=int, required=True)
    args, unknown = parser.parse_known_args()
    # * load dataset
    dataset = datasets.load_from_disk(args.input_dataset)
    total_size = args.train_size + args.valid_size + args.test_size
    if total_size > len(dataset):
        raise ValueError(
            f"The dataset only has {len(dataset)} samples, but passed arguments need train_size + valid_size + test_size = {total_size}!"
        )
    # * load model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-large-en')
    model = BertModel.from_pretrained('BAAI/bge-large-en')
    # * set trainer
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="./tmp",
            per_device_eval_batch_size=100,
            remove_unused_columns=False,
            dataloader_num_workers=32,
        ),
        data_collator=MyDataCollator(tokenizer),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )
    # * get embedding
    test_result = trainer.predict(dataset)
    embeddings = test_result.predictions
    # * create dataset
    dataset = dataset.map(
        lambda _, idx: {"embedding": embeddings[idx]}, with_indices=True, num_proc=32
    )
    train_dataset = dataset.select(range(args.train_size))
    valid_dataset = dataset.select(
        range(args.train_size, args.train_size + args.valid_size)
    )
    test_dataset = dataset.select(
        range(
            args.train_size + args.valid_size,
            args.train_size + args.valid_size + args.test_size,
        )
    )
    final_dataset = datasets.DatasetDict(
        {
            "train": train_dataset,
            "valid": valid_dataset,
            "test": test_dataset,
        }
    )
    final_dataset.save_to_disk(args.output_dataset)
    
if __name__ == "__main__":
    main()
