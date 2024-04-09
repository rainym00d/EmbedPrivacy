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
from transformers.trainer_utils import get_last_checkpoint

from model import get_decrypt_model
from utils.args import DataArgs, GenerationArgs, ModelArgs, TrainingArgs
from utils.callback import PerplexityCallback
from utils.data import DataCollatorWithEmbedding
from utils.metric import get_compute_metrics
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
    # * detecting last checkpoint
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            logging.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change the `--output_dir` or add `--overwrite_output_dir` to train."
            )
    # * init model and tokenizer
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
    if (
        model_args.from_pretrained
        and training_args.resume_from_checkpoint is None
        and last_checkpoint is None
    ):
        model = DecryptModel.from_pretrained(
            pretrained_model_name_or_path=model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            **kwargs,
        )
        logging.info(f"Load model from pretrained ({model_args.model_name_or_path}).")
    else:
        model = DecryptModel(config, **kwargs)
        if training_args.resume_from_checkpoint is not None:
            logging.info(
                f"Init model from scratch. And then resume from {training_args.resume_from_checkpoint}. If you are confused, please check `training_args.resume_from_checkpoint`."
            )
        elif last_checkpoint is not None:
            logging.info(
                f"Init model from scratch. And then resume from {last_checkpoint}. If you are confused, please check `last_checkpoint`."
            )
        else:
            logging.info(
                "Init model from scratch. If you are confused, please check `model_args.from_pretrained`."
            )

    # * load dataset
    dataset = datasets.load_from_disk(data_args.dataset_name_or_path)
    train_dataset = dataset["train"].select(range(data_args.train_num))
    valid_dataset = dataset["valid"]
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

    # * prepare for trainer
    compute_metrics = get_compute_metrics(tokenizer)
    # * set trainer
    trainer = MyTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithEmbedding(tokenizer, model_args.model_max_length),
        compute_metrics=compute_metrics,
        callbacks=[
            PerplexityCallback(),
        ],
    )
    # * whether to train
    if training_args.do_train:
        # * whether to laod checkpoint
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
            logging.info(
                f"`training_args.resume_from_checkpoint` is not none. resuming training at {checkpoint}."
            )
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        # * train
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        best_model_dir = os.path.join(training_args.output_dir, "best_model")
        trainer.save_model(best_model_dir)  # Saves the tokenizer too for easy upload
        # * save some information
        metrics = {"train": train_result.metrics}
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # * whether to test
    if training_args.do_predict:
        # * test each dataset
        for dataset_name, test_dataset in test_dataset_dict.items():
            for _ in range(10):
                test_result = trainer.predict(test_dataset)
                # * log and save test metrics
                metrics = test_result.metrics
                metrics = {dataset_name: metrics}
                trainer.log_metrics("test", metrics, dataset_name=dataset_name)
                trainer.save_metrics("test", metrics, dataset_name=dataset_name)

                # * only main process generate text and write to file
                if trainer.is_world_process_zero() and training_args.predict_with_generate:
                    predictions = test_result.predictions
                    predictions = np.where(
                        predictions != -100, predictions, tokenizer.pad_token_id
                    )

                    decoded_preds = tokenizer.batch_decode(
                        predictions, skip_special_tokens=True
                    )
                    decoded_preds = [decoded_pred.strip() for decoded_pred in decoded_preds]
                    output_prediction_path = os.path.join(
                        training_args.output_dir, "generated_predictions"
                    )
                    if not os.path.exists(output_prediction_path):
                        os.makedirs(output_prediction_path)
                    output_prediction_file = os.path.join(
                        output_prediction_path, f"{dataset_name}.json"
                    )
                    generated_predictions = [
                        {"text": raw_input["text"], "pred": decoded_pred}
                        for raw_input, decoded_pred in zip(test_dataset, decoded_preds)
                    ]
                    with open(output_prediction_file, "w") as f:
                        json.dump(generated_predictions, f, indent=4)


if __name__ == "__main__":
    main()
