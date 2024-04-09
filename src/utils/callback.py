import math

from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import IntervalStrategy


class SaveAtLastStepCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        # * if save strategy is `step`, and training should end, then save the last model to resume
        if (
            args.save_strategy == IntervalStrategy.STEPS
            and state.global_step >= state.max_steps
        ):
            control.should_save = True

        return control


# * this metric will not show in log
class PerplexityCallback(TrainerCallback):
    def __init__(
        self, eval_metric_key_prefix: str = "eval", pred_metric_key_prefix: str = "test"
    ) -> None:
        super().__init__()
        self.eval_metric_key_prefix = eval_metric_key_prefix
        self.pred_metric_key_prefix = pred_metric_key_prefix

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        prefix = self.eval_metric_key_prefix
        assert f"{prefix}_loss" in metrics, f"Make sure `{prefix}_loss` in `metrics`"
        perplexity = math.exp(metrics[f"{prefix}_loss"])
        metrics[f"{prefix}_perplexity"] = perplexity

    def on_predict(self, args, state, control, metrics, **kwargs):
        prefix = self.pred_metric_key_prefix
        assert f"{prefix}_loss" in metrics, f"Make sure `{prefix}_loss` in `metrics`"
        perplexity = math.exp(metrics[f"{prefix}_loss"])
        metrics[f"{prefix}_perplexity"] = perplexity
