import argparse
from dataclasses import dataclass
from collections import OrderedDict
import json
import datasets
import torch
import os
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
    # return logits[0][:, 0]
    return torch.nn.functional.normalize(logits[0][:, 0], p=2, dim=1)


def main():
    # * parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_list", type=str, required=True)
    parser.add_argument("--dataset_save_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args, unknown = parser.parse_known_args()
    
    dataset_list = [dataset.strip() for dataset in args.dataset_list.split(",")]
    
    dataset_dict = OrderedDict()
    for dataset_name in dataset_list:
        path = os.path.join(args.dataset_save_dir, f"attr_{dataset_name}.json")
        dataset_dict[dataset_name] = datasets.Dataset.from_json(path, cache_dir="cache")
    
    # * load model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-large-en')
    model = BertModel.from_pretrained('BAAI/bge-large-en')
    
    # * set trainer
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=args.output_dir,
            per_device_eval_batch_size=100,
            remove_unused_columns=False,
            dataloader_num_workers=32,
        ),
        data_collator=MyDataCollator(tokenizer),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )
    
    all_results = {}
    for dataset_name, test_dataset in dataset_dict.items():
        sample_size = len(test_dataset)
        # * sentence to embedding
        new_test_dataset = datasets.Dataset.from_dict({
            "text":sum(test_dataset["pred"], [])
        })
        test_result = trainer.predict(new_test_dataset)
        text_embeddings = test_result.predictions
        text_embeddings = text_embeddings.reshape(sample_size, 10, -1)

        # * test each attribute
        # * get all attribute in test dataset
        attribute_list = []
        for k in test_dataset.features.keys():
            if k not in ["text", "pred"] and "candidate_list" not in k:
                attribute_list.append(k)
        
        for attr in attribute_list:
            candidate_size = len(test_dataset[0][f"{attr}_candidate_list"])
            new_test_dataset = {"text": []}
            for example in test_dataset:
                for label in example[f"{attr}_candidate_list"]:
                    prompt = f"The {' '.join(attr.split('_'))} is {label}"
                    new_test_dataset["text"].append(prompt)
            new_test_dataset = datasets.Dataset.from_dict(new_test_dataset)
            
            # * attr to embedding
            test_result = trainer.predict(new_test_dataset)
            attr_embeddings = test_result.predictions
            attr_embeddings = attr_embeddings.reshape(sample_size, candidate_size, -1)
            
            if trainer.is_world_process_zero():
                metrics = [0] * 10
                for i in range(sample_size):
                    scores = text_embeddings[i] @ attr_embeddings[i].T
                    answers = scores.argmax(axis=-1)
                    for idx, answer in enumerate(answers):
                        if answer == 0:
                            metrics[idx] += 1
                metrics = [metric / sample_size for metric in metrics]
                all_results[f"{dataset_name}_{attr}"] = metrics

    if trainer.is_world_process_zero():
        output_file = os.path.join(args.output_dir, "attr_results.json")
        with open(output_file, "w") as f:
            json.dump(all_results, f, indent=4)
            
    
if __name__ == "__main__":
    main()
