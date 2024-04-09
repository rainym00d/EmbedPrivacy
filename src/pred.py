import json
import logging
import os
import sys
import warnings
from collections import OrderedDict

warnings.filterwarnings("ignore")

import datasets
import numpy as np
from transformers import (
    AutoConfig,
    AutoTokenizer,
    GenerationConfig,
    HfArgumentParser,
    set_seed,
)

from model import get_decrypt_model
from utils.args import DataArgs, GenerationArgs, ModelArgs, TrainingArgs
from utils.callback import PerplexityCallback
from utils.data import DataCollatorWithEmbedding
from utils.trainer import MyTrainer


def main():
    # * set parser
    parser = HfArgumentParser([ModelArgs, DataArgs, TrainingArgs, GenerationArgs])
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # * If we pass only one argument to the script and it's the path to a json file, let's parse it to get our arguments.
        model_args, data_args, training_args, generation_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        (
            model_args,
            data_args,
            training_args,
            generation_args,
        ) = parser.parse_args_into_dataclasses()

    # * set logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )
    # * set seed before initializing model.
    set_seed(training_args.seed)

    # * load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(
        pretrained_model_name_or_path=model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        trust_remote_code=True,
        local_files_only=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        trust_remote_code=True,
        local_files_only=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # * set pad token
    DecryptModel = get_decrypt_model(config.model_type)
    kwargs = {
        "embedding_dim": data_args.embedding_dim,
        "tokenizer": tokenizer,
    }

    model = DecryptModel.from_pretrained(
        pretrained_model_name_or_path=model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        **kwargs,
    )
    logging.info(f"Load model from pretrained ({model_args.model_name_or_path}).")
    
    # * load dataset
    test_dataset_dict = OrderedDict()
    for dataset_path in data_args.test_dataset_name_or_path:
        head, _ = os.path.split(dataset_path)
        dataset_name = os.path.split(head)[1]
        test_dataset_dict[dataset_name] = datasets.load_from_disk(dataset_path)
    # * set generation config to training args
    generation_args = generation_args.to_dict()
    generation_args["pad_token_id"] = tokenizer.pad_token_id
    generation_args["bos_token_id"] = tokenizer.bos_token_id
    generation_args["eos_token_id"] = tokenizer.eos_token_id
    model.generation_config = GenerationConfig(**generation_args)

    # * set trainer
    trainer = MyTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithEmbedding(tokenizer, model_args.model_max_length),
    )

    # * whether to test
    if not training_args.do_predict:
        return
    # * test each dataset
    for dataset_name, test_dataset in test_dataset_dict.items():
        all_decoded_preds = []
        for i in range(10):
            test_result = trainer.predict(test_dataset)
            
            if trainer.is_world_process_zero() and training_args.predict_with_generate:
                predictions = test_result.predictions
                predictions = np.where(
                    predictions != -100, predictions, tokenizer.pad_token_id
                )
                
                decoded_preds = tokenizer.batch_decode(
                    predictions, skip_special_tokens=True
                )
                decoded_preds = [decoded_pred.strip() for decoded_pred in decoded_preds]
                
                for j, decoded_pred in enumerate(decoded_preds):
                    if i == 0:
                        all_decoded_preds.append([decoded_pred])
                    else:
                        all_decoded_preds[j].append(decoded_pred)
        
        if trainer.is_world_process_zero() and training_args.predict_with_generate:
            output_prediction_path = os.path.join(
                training_args.output_dir, "generated_predictions"
            )
            if not os.path.exists(output_prediction_path):
                os.makedirs(output_prediction_path)
            output_prediction_file = os.path.join(
                output_prediction_path, f"attr_{dataset_name}.json"
            )
            
            generated_predictions = []
            for j in range(len(test_dataset)):
                data = test_dataset[j]
                data.pop("embedding")
                data["pred"] = all_decoded_preds[j]
                generated_predictions.append(data)
            
            with open(output_prediction_file, "w") as f:
                json.dump(generated_predictions, f, indent=4)


if __name__ == "__main__":
    main()
