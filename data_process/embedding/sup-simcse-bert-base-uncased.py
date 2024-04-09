import argparse
from dataclasses import dataclass

import datasets
from transformers import (
    AutoTokenizer,
    BertModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)


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
    return logits[1]


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
    model = BertModel.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
    # * set trainer
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="./tmp",
            per_device_eval_batch_size=100,
            remove_unused_columns=False,
            dataloader_num_workers=16,
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
