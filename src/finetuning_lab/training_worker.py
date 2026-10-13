"""Fresh-process MLX training instrumentation, shared by both methods."""
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.lora import CONFIG_DEFAULTS, train_model
from mlx_lm.tuner.datasets import load_dataset
from mlx_lm.tuner.callbacks import TrainingCallback
from .core import write_json

class Recorder(TrainingCallback):
    def __init__(self):
        self.train = []
        self.validation = []
    def on_train_loss_report(self, info):
        self.train.append(info)
    def on_val_loss_report(self, info):
        self.validation.append(info)

def main():
    config = json.loads(Path(sys.argv[1]).read_text())
    args = SimpleNamespace(**(CONFIG_DEFAULTS | config))
    np.random.seed(args.seed)
    mx.random.seed(args.seed)
    model, tokenizer = load(args.model)
    mx.eval(model.parameters())
    mx.synchronize()
    loaded_memory = mx.get_active_memory()
    train_set, valid_set, _ = load_dataset(args, tokenizer)
    recorder = Recorder()
    mx.reset_peak_memory()
    started = time.perf_counter()
    train_model(args, model, train_set, valid_set, recorder)
    mx.synchronize()
    elapsed = time.perf_counter() - started
    cumulative = 0
    compute_seconds = 0.
    for entry in recorder.train:
        interval_tokens = entry['trained_tokens'] - cumulative
        cumulative = entry['trained_tokens']
        if entry['tokens_per_second'] > 0:
            compute_seconds += interval_tokens / entry['tokens_per_second']
    parameters = {name:list(value.shape) for name,value in tree_flatten(model.trainable_parameters())}
    write_json(Path(args.adapter_path)/'training_metrics.json', {
        'loaded_base_active_bytes':loaded_memory,
        'training_peak_allocated_bytes':mx.get_peak_memory(),
        'training_wall_seconds':elapsed,
        'training_compute_seconds':compute_seconds,
        'loss_tokens_processed':cumulative,
        'loss_tokens_per_compute_second':cumulative/compute_seconds if compute_seconds else None,
        'trainable_parameter_shapes':parameters,
        'trainable_parameters':sum(value.size for _,value in tree_flatten(model.trainable_parameters())),
        'train_reports':recorder.train, 'validation_reports':recorder.validation,
        'measurement_scope':'Peak reset after base load and dataset setup; training includes adapter/optimizer creation, validation, and checkpoint saves. Wall time covers the same region. Throughput counts completion-loss tokens, not all input tokens.'})

if __name__ == '__main__': main()
