import json
import os

from transformers import Seq2SeqTrainer


class MyTrainer(Seq2SeqTrainer):
    def log_metrics(self, split, metrics, dataset_name=None):
        """
        Log metrics in a specially formatted way

        Under distributed environment this is done only for a process with rank 0.

        Args:
            split (`str`):
                Mode/split name: one of `train`, `eval`, `test`
            metrics (`Dict[str, float]`):
                The metrics returned from train/evaluate/predictmetrics: metrics dict

        Notes on memory reports:

        In order to get memory usage report you need to install `psutil`. You can do that with `pip install psutil`.

        Now when this method is run, you will see a report that will include: :

        ```
        init_mem_cpu_alloc_delta   =     1301MB
        init_mem_cpu_peaked_delta  =      154MB
        init_mem_gpu_alloc_delta   =      230MB
        init_mem_gpu_peaked_delta  =        0MB
        train_mem_cpu_alloc_delta  =     1345MB
        train_mem_cpu_peaked_delta =        0MB
        train_mem_gpu_alloc_delta  =      693MB
        train_mem_gpu_peaked_delta =        7MB
        ```

        **Understanding the reports:**

        - the first segment, e.g., `train__`, tells you which stage the metrics are for. Reports starting with `init_`
            will be added to the first stage that gets run. So that if only evaluation is run, the memory usage for the
            `__init__` will be reported along with the `eval_` metrics.
        - the third segment, is either `cpu` or `gpu`, tells you whether it's the general RAM or the gpu0 memory
            metric.
        - `*_alloc_delta` - is the difference in the used/allocated memory counter between the end and the start of the
            stage - it can be negative if a function released more memory than it allocated.
        - `*_peaked_delta` - is any extra memory that was consumed and then freed - relative to the current allocated
            memory counter - it is never negative. When you look at the metrics of any stage you add up `alloc_delta` +
            `peaked_delta` and you know how much memory was needed to complete that stage.

        The reporting happens only for process of rank 0 and gpu 0 (if there is a gpu). Typically this is enough since the
        main process does the bulk of work, but it could be not quite so if model parallel is used and then other GPUs may
        use a different amount of gpu memory. This is also not the same under DataParallel where gpu0 may require much more
        memory than the rest since it stores the gradient and optimizer states for all participating GPUS. Perhaps in the
        future these reports will evolve to measure those too.

        The CPU RAM metric measures RSS (Resident Set Size) includes both the memory which is unique to the process and the
        memory shared with other processes. It is important to note that it does not include swapped out memory, so the
        reports could be imprecise.

        The CPU peak memory is measured using a sampling thread. Due to python's GIL it may miss some of the peak memory if
        that thread didn't get a chance to run when the highest memory was used. Therefore this report can be less than
        reality. Using `tracemalloc` would have reported the exact peak memory, but it doesn't report memory allocations
        outside of python. So if some C++ CUDA extension allocated its own memory it won't be reported. And therefore it
        was dropped in favor of the memory sampling approach, which reads the current process memory usage.

        The GPU allocated and peak memory reporting is done with `torch.cuda.memory_allocated()` and
        `torch.cuda.max_memory_allocated()`. This metric reports only "deltas" for pytorch-specific allocations, as
        `torch.cuda` memory management system doesn't track any memory allocated outside of pytorch. For example, the very
        first cuda call typically loads CUDA kernels, which may take from 0.5 to 2GB of GPU memory.

        Note that this tracker doesn't account for memory allocations outside of [`Trainer`]'s `__init__`, `train`,
        `evaluate` and `predict` calls.

        Because `evaluation` calls may happen during `train`, we can't handle nested invocations because
        `torch.cuda.max_memory_allocated` is a single counter, so if it gets reset by a nested eval call, `train`'s tracker
        will report incorrect info. If this [pytorch issue](https://github.com/pytorch/pytorch/issues/16266) gets resolved
        it will be possible to change this class to be re-entrant. Until then we will only track the outer level of
        `train`, `evaluate` and `predict` methods. Which means that if `eval` is called during `train`, it's the latter
        that will account for its memory usage and that of the former.

        This also means that if any other tool that is used along the [`Trainer`] calls
        `torch.cuda.reset_peak_memory_stats`, the gpu peak memory stats could be invalid. And the [`Trainer`] will disrupt
        the normal behavior of any such tools that rely on calling `torch.cuda.reset_peak_memory_stats` themselves.

        For best performance you may want to consider turning the memory profiling off for production runs.
        """
        if not self.is_world_process_zero():
            return

        # *
        if dataset_name:
            print(f"***** {split} {dataset_name} metrics *****")
            metrics = metrics[dataset_name]
        else:
            print(f"***** {split} metrics *****")
            metrics = metrics[split]
        # *
        metrics_formatted = self.metrics_format(metrics)
        k_width = max(len(str(x)) for x in metrics_formatted.keys())
        v_width = max(len(str(x)) for x in metrics_formatted.values())
        for key in sorted(metrics_formatted.keys()):
            print(f"  {key: <{k_width}} = {metrics_formatted[key]:>{v_width}}")

    def save_metrics(self, split, metrics, combined=True, dataset_name=None):
        """
        Save metrics into a json file for that split, e.g. `train_results.json`.

        Under distributed environment this is done only for a process with rank 0.

        Args:
            split (`str`):
                Mode/split name: one of `train`, `eval`, `test`, `all`
            metrics (`Dict[str, float]`):
                The metrics returned from train/evaluate/predict
            combined (`bool`, *optional*, defaults to `True`):
                Creates combined metrics by updating `all_results.json` with metrics of this call

        To understand the metrics please read the docstring of [`~Trainer.log_metrics`]. The only difference is that raw
        unformatted numbers are saved in the current method.

        """
        if not self.is_world_process_zero():
            return
        # * modify the filename depend on dataset name
        if dataset_name:
            path = os.path.join(
                self.args.output_dir, f"{split}_results_{dataset_name}.json"
            )
        else:
            path = os.path.join(self.args.output_dir, f"{split}_results.json")
        # *
        new_metrics = {}
        for name, inner_dict in metrics.items():
            new_metrics[name] = {}
            for k, v in inner_dict.items():
                new_metrics[name][k] = [v]
        metrics = new_metrics

        if os.path.exists(path):
            with open(path, "r") as f:
                new_metrics = json.load(f)
                
            for name, inner_dict in metrics.items():
                for k, v in inner_dict.items():
                    new_metrics[name][k].append(v[0])
            metrics = new_metrics

        with open(path, "w") as f:
            json.dump(metrics, f, indent=4, sort_keys=True)

        if combined:
            path = os.path.join(self.args.output_dir, "all_results.json")
            if os.path.exists(path):
                with open(path, "r") as f:
                    all_metrics = json.load(f)
            else:
                all_metrics = {}

            all_metrics.update(metrics)
            with open(path, "w") as f:
                json.dump(all_metrics, f, indent=4, sort_keys=True)
