import os
from dataclasses import asdict, dataclass, field
from typing import Optional

from transformers import Seq2SeqTrainingArguments


@dataclass
class ModelArgs:
    # * base
    model_name_or_path: str = field(
        default="gpt2-xl",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    cache_dir: str = field(
        default="your_cache_dir",
        metadata={"help": "Path to save pretrain model."},
    )
    # * tokenizer parameter
    model_max_length: int = field(
        default=150,
        metadata={
            "help": "The maximum length (in number of tokens) for the inputs to the transformer model."
        },
    )
    # * other parameter
    use_special_token: bool = field(
        default=True,
        metadata={"help": "Whether to add special token to input."},
    )
    from_pretrained: bool = field(
        default=False,
        metadata={"help": "Whether to init model from pretrained."},
    )


@dataclass
class DataArgs:
    # * base
    dataset_name_or_path: str = field(
        default="wikipedia",
        metadata={"help": "Path of dataset"},
    )
    test_dataset_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Test dataset"},
    )
    dataset_save_dir: str = field(
        default="./data/EmbedPrivacy",
        metadata={"help": "The path to save dataset."},
    )
    train_num: int = field(
        default=-1,
        metadata={"help": "The number of training data."},
    )
    # * embedding
    embedding_model: str = field(
        default="gtr-t5-large",
        metadata={"help": "Embedding model."},
    )
    embedding_dim: int = field(
        default=768,
        metadata={"help": "Embdding dim of data."},
    )

    def __post_init__(self):
        # * handle dataset_name_or_path
        self.dataset_name_or_path = self._handle_dataset_path(self.dataset_name_or_path)
        # * handle test_dataset_name_or_path
        if self.test_dataset_name_or_path is None:
            self.test_dataset_name_or_path = self.dataset_name_or_path
        if isinstance(self.test_dataset_name_or_path, str):
            self.test_dataset_name_or_path = [
                s.strip() for s in self.test_dataset_name_or_path.split(",")
            ]
        if isinstance(self.test_dataset_name_or_path, list):
            self.test_dataset_name_or_path = [
                self._handle_dataset_path(s, True)
                for s in self.test_dataset_name_or_path
            ]

    def _handle_dataset_path(self, path, is_test=False):
        # * remove last slash
        path = path[:-1] if path.endswith("/") else path
        # * join dataset_save_dir and path
        # todo judge common path
        path = os.path.join(self.dataset_save_dir, self.embedding_model, path)

        # * if path is test_path, then join `test`
        last_folder = os.path.basename(path)
        if is_test and last_folder != "test":
            path = os.path.join(path, "test")

        return path


@dataclass
class TrainingArgs(Seq2SeqTrainingArguments):
    # * base
    do_train: bool = field(
        default=True, metadata={"help": "Whether to run training or not."}
    )
    do_predict: bool = field(
        default=True,
        metadata={"help": "Whether to run predictions on the test set or not."},
    )
    output_dir: str = field(
        default="outputs/0808-gpt2-xl-no-pretrained-layer-2",
        metadata={
            "help": "The output directory where the model predictions and checkpoints will be written."
        },
    )
    overwrite_output_dir: bool = field(
        default=False,
        metadata={"help": "If True, overwrite the content of the output directory."},
    )
    resume_from_checkpoint: Optional[str] = field(
        default=None,
        metadata={
            "help": "The path to a folder with a valid checkpoint for your model."
        },
    )
    # * basic train parameter
    num_train_epochs: float = field(
        default=10,
        metadata={"help": "Total number of training epochs to perform."},
    )
    max_steps: int = field(
        default=-1,
        metadata={
            "help": "If set to a positive number, the total number of training steps to perform. Overrides num_train_epochs."
        },
    )
    per_device_train_batch_size: int = field(
        default=20,
        metadata={"help": "The batch size per GPU/TPU core/CPU for training."},
    )
    per_device_eval_batch_size: int = field(
        default=100,
        metadata={"help": "The batch size per GPU/TPU core/CPU for evaluation."},
    )
    # * data parameter
    dataloader_num_workers: int = field(
        default=32,
        metadata={"help": "Number of subprocesses to use for data loading."},
    )
    remove_unused_columns: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to automatically remove the columns unused by the model forward method."
        },
    )
    # * eval parameter
    evaluation_strategy: str = field(
        default="steps",
        metadata={"help": "Evaluation strategy."},
    )
    eval_steps: int = field(
        default=1000,
        metadata={"help": "Evaluation frequency according to evaluation strategy."},
    )
    metric_for_best_model: str = field(
        default="eval_loss",
        metadata={"help": "Metric for selecting model."},
    )
    greater_is_better: bool = field(
        default=False,
        metadata={"help": "Bigger metric for better model."},
    )
    load_best_model_at_end: bool = field(
        default=True,
        metadata={
            "help": "Whether or not to load the best model found during training at the end of training."
        },
    )
    # * save & log parameter
    save_strategy: str = field(
        default="steps",
        metadata={"help": "The checkpoint save strategy to adopt during training."},
    )
    save_steps: int = field(
        default=1000,
        metadata={"help": "Saving frequency according to saving strategy"},
    )
    save_total_limit: int = field(
        default=2,
        metadata={"help": "How many checkpoints to keep in the output_dir."},
    )
    logging_steps: int = field(
        default=100,
        metadata={"help": "Logging frequency according to logging strategy."},
    )
    # * half precesion & ddp parameter
    fp16: bool = field(
        default=False,
        metadata={
            "help": "Whether to use fp16 16-bit (mixed) precision training instead of 32-bit training."
        },
    )
    fp16_full_eval: bool = field(
        default=False,
        metadata={"help": "Whether to use full float16 evaluation instead of 32-bit."},
    )
    bf16: bool = field(
        default=True,
        metadata={
            "help": "Whether to use bf16 16-bit (mixed) precision training instead of 32-bit training."
        },
    )
    bf16_full_eval: bool = field(
        default=False,
        metadata={"help": "Whether to use full bfloat16 evaluation instead of 32-bit."},
    )
    ddp_find_unused_parameters: bool = field(
        default=False,
        metadata={
            "help": "When using distributed training, the value of the flag find_unused_parameters passed to DistributedDataParallel."
        },
    )
    # * generate parameter
    predict_with_generate: bool = field(
        default=True,
        metadata={"help": "Call generate function when prediction."},
    )
    return_new_tokens_only: bool = field(
        default=True,
        metadata={"help": "Return only newly generated tokens when generation."},
    )
    # * wandb parameter
    use_wandb: bool = field(
        default=True,
        metadata={"help": "Whether to use wandb."},
    )
    wandb_id: Optional[str] = field(
        default=None,
        metadata={"help": "If wandb id is not None, will use this id to resume."},
    )
    wandb_project: Optional[str] = field(
        default="EmbedPrivacy",
        metadata={"help": "The name of wandb project, if None, use huggingface."},
    )
    run_name: Optional[str] = field(
        default="gpt2-xl-no-pretrained-layer-2",
        metadata={
            "help": "A descriptor for the run. Typically used for wandb logging."
        },
    )

    def __post_init__(self):
        if self.use_wandb:
            self.report_to = "wandb"
        else:
            self.report_to = "none"

        if self.fp16:
            self.fp16_full_eval = True
        if self.bf16:
            self.bf16_full_eval = True

        super().__post_init__()


@dataclass
class GenerationArgs:
    do_sample: bool = field(
        default=False,
        metadata={"help": "Sample when decoding?"},
    )
    num_return_sequences: int = field(
        default=1,
        metadata={"help": "How many sequences to generate?"},
    )
    temperature: float = field(
        default=0.6,
        metadata={"help": "Temperature for sampling."},
    )
    top_p: float = field(
        default=0.9,
        metadata={"help": "Top-p sampling value."},
    )
    max_new_tokens: int = field(
        default=128,
        metadata={"help": "Maximum new token number."},
    )

    def to_dict(self):
        return asdict(self)
