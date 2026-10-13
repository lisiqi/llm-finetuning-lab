"""Local experiment commands. No proprietary data or hosted generation calls."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from .core import (digest, flatten_squad, messages, prepare, read_jsonl,
                   score_predictions, write_json, write_jsonl)

URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/{}-v2.0.json"


def model_fingerprint(folder):
    files = sorted(p for p in Path(folder).rglob('*') if p.is_file() and '.cache' not in p.parts)
    if not files:
        raise ValueError("Model directory is empty")
    return {str(p.relative_to(folder)): digest(p) for p in files}


def metadata():
    packages = {}
    for package in ("mlx", "mlx-lm", "transformers", "huggingface-hub"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {"python": sys.version, "platform": platform.platform(), "packages": packages}


def download(args):
    folder = Path(args.output)
    folder.mkdir(parents=True, exist_ok=True)
    for split in ("train", "dev"):
        target = folder / f"{split}-v2.0.json"
        if not target.exists():
            temp = target.with_suffix(".download")
            try:
                with urllib.request.urlopen(URL.format(split), timeout=120) as response:
                    temp.write_bytes(response.read())
                flatten_squad(json.loads(temp.read_text()))
                temp.replace(target)
            finally:
                temp.unlink(missing_ok=True)
        flatten_squad(json.loads(target.read_text()))
        print(target, digest(target))
    write_json(folder / "manifest.json", {"source": URL, "sha256": {
        split: digest(folder / f"{split}-v2.0.json") for split in ("train", "dev")}})


def prepare_data(args):
    source = Path(args.source)
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output already contains data; choose a new directory")
    payloads = [json.loads((source / f"{s}-v2.0.json").read_text()) for s in ("train", "dev")]
    groups = prepare(*payloads, sizes=(args.train, args.valid, args.test), seed=args.seed,
                     max_chars=args.max_context_chars)
    for name, rows in groups.items():
        write_jsonl(output / f"{name}.jsonl", [{"messages": messages(r, True)} for r in rows])
        write_jsonl(output / f"{name}.reference.jsonl", rows)
    write_json(output / "manifest.json", {"seed": args.seed, "max_context_chars": args.max_context_chars,
        "source_hashes": {s: digest(source / f"{s}-v2.0.json") for s in ("train", "dev")},
        "splits": {name: {"n": len(rows), "unanswerable": sum(r['unanswerable'] for r in rows),
            "articles": sorted({r['title'] for r in rows}),
            "sha256": digest(output / f"{name}.reference.jsonl")} for name, rows in groups.items()}})
    print(f"Prepared {sum(map(len, groups.values()))} examples in {output}")


def tokenize_check(args):
    from mlx_lm import load
    _, tokenizer = load(args.model)
    folder = Path(args.data)
    report = {}
    for split in ("train", "valid", "test"):
        lengths = [len(tokenizer.apply_chat_template(row["messages"], tokenize=True))
                   for row in read_jsonl(folder / f"{split}.jsonl")]
        report[split] = {"n": len(lengths), "max_tokens": max(lengths),
                         "over_limit": sum(n > args.max_tokens for n in lengths)}
    print(json.dumps(report, indent=2))
    if any(row["over_limit"] for row in report.values()):
        raise ValueError("Examples exceed sequence budget. Reprepare with shorter contexts or increase budget; do not silently truncate answers.")


def train(args):
    import yaml
    config = json.loads(Path(args.config).read_text())
    if config.get("fine_tune_type") != "lora" or not config.get("mask_prompt"):
        raise ValueError("Expected LoRA with completion-only loss")
    # Verify sequences before starting expensive training.
    tokenize_check(argparse.Namespace(model=config["model"], data=config["data"],
                                      max_tokens=config["max_seq_length"]))
    folder = Path(config["adapter_path"])
    if folder.exists() and any(folder.iterdir()):
        raise ValueError("Adapter directory is nonempty; choose another path to preserve old results")
    folder.mkdir(parents=True, exist_ok=True)
    yaml_path = folder / "training.yaml"
    yaml_path.write_text(yaml.safe_dump(config))
    write_json(folder / "experiment.json", {"config":config, "environment":metadata(),
               "data_manifest_sha256":digest(Path(config['data']) / 'manifest.json'),
               "model_files_sha256":model_fingerprint(Path(config["model"])),
               "training_data_sha256":{split:digest(Path(config["data"])/f"{split}.jsonl") for split in ("train","valid")}})
    started = time.perf_counter()
    with (folder / "training.log").open("w") as log:
        process = subprocess.Popen([sys.executable, "-m", "finetuning_lab.training_worker", str(Path(args.config).resolve())],
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        code = process.wait()
    write_json(folder / "training_status.json", {"exit_code":code,
               "wall_seconds":time.perf_counter()-started})
    if code:
        raise RuntimeError(f"Training failed; see {folder / 'training.log'}")


def evaluate(args):
    import mlx.core as mx
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler
    if args.max_tokens <= 0 or (args.limit is not None and args.limit <= 0):
        raise ValueError("Token budget and limit must be positive")
    output = Path(args.output)
    if output.exists():
        raise ValueError("Run directory exists; choose a new output")
    if args.adapter and not (Path(args.adapter) / 'adapters.safetensors').exists():
        raise ValueError("Adapter weights missing")
    model, tokenizer = load(args.model, adapter_path=args.adapter)
    rows = read_jsonl(args.references)
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        raise ValueError("Empty evaluation set")
    output.mkdir(parents=True)
    mx.random.seed(args.seed)
    predictions = []
    started = time.perf_counter()
    for i,row in enumerate(rows):
        prompt = tokenizer.apply_chat_template(messages(row), tokenize=False, add_generation_prompt=True)
        text = generate(model, tokenizer, prompt=prompt, max_tokens=args.max_tokens,
                        sampler=make_sampler(temp=0.0), verbose=False)
        predictions.append({"id":row["id"], "prediction":text})
        write_jsonl(output / "predictions.jsonl", predictions)
        print(f"{i+1}/{len(rows)} {row['id']}", flush=True)
    metrics, details = score_predictions(rows, predictions)
    metrics["wall_seconds"] = time.perf_counter()-started
    peak = getattr(mx, "get_peak_memory", None)
    metrics["mlx_peak_memory_bytes"] = peak() if peak else None
    write_json(output / "metrics.json", metrics)
    write_jsonl(output / "details.jsonl", details)
    adapter_hash = digest(Path(args.adapter)/'adapters.safetensors') if args.adapter else None
    config_path = Path(args.model)/'config.json'
    write_json(output / "manifest.json", {"model":str(Path(args.model).resolve()),
        "model_config_sha256":digest(config_path) if config_path.exists() else None,
        "model_files_sha256":model_fingerprint(Path(args.model)),
        "adapter":args.adapter, "adapter_sha256":adapter_hash,
        "references_sha256":digest(args.references), "ids":[r['id'] for r in rows],
        "prompt_sha256":hashlib.sha256(json.dumps(messages(rows[0])[:1]).encode()).hexdigest(),
        "seed":args.seed, "max_tokens":args.max_tokens, "temperature":0., "environment":metadata()})
    print(json.dumps(metrics, indent=2))


def compare(args):
    folders = [Path(args.baseline), Path(args.adapted)]
    manifests = [json.loads((p / 'manifest.json').read_text()) for p in folders]
    for key in ('model', 'model_config_sha256', 'model_files_sha256', 'references_sha256', 'ids', 'prompt_sha256',
                'seed', 'max_tokens', 'temperature', 'environment'):
        if manifests[0][key] != manifests[1][key]:
            raise ValueError(f"Unmatched experiment setting: {key}")
    if manifests[0]['adapter'] is not None or not manifests[1]['adapter_sha256']:
        raise ValueError("Expected base-model baseline and trained adapter run")
    metrics = [json.loads((p / 'metrics.json').read_text()) for p in folders]
    lines = ['# Baseline vs adapter', '', '| Metric | Baseline | Adapter | Delta |', '|---|---:|---:|---:|']
    for key in ('answerable_exact_match','answerable_token_f1','overall_exact_match',
                'overall_token_f1','abstention_precision','abstention_recall'):
        a,b = [m[key] for m in metrics]
        values = [f'{v:.4f}' if v is not None else 'undefined' for v in (a,b)]
        delta = f'{b-a:+.4f}' if a is not None and b is not None else 'undefined'
        lines.append(f'| {key} | {values[0]} | {values[1]} | {delta} |')
    lines += ['', 'Scores are fractions. Balanced sampling differs from the natural dataset distribution.',
              'Inspect details.jsonl for both runs; no improvement is assumed.']
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('download'); p.add_argument('--output',default='data/raw'); p.set_defaults(fn=download)
    p = sub.add_parser('prepare'); p.add_argument('--source',default='data/raw'); p.add_argument('--output',default='data/prepared')
    for key,n in [('train',500),('valid',100),('test',200),('seed',42),('max-context-chars',1600)]:
        p.add_argument('--'+key,type=int,default=n)
    p.set_defaults(fn=prepare_data)
    p = sub.add_parser('check-tokens'); p.add_argument('--model',default='models/base'); p.add_argument('--data',default='data/prepared'); p.add_argument('--max-tokens',type=int,default=512); p.set_defaults(fn=tokenize_check)
    p = sub.add_parser('train'); p.add_argument('--config',default='configs/m1.json'); p.set_defaults(fn=train)
    p = sub.add_parser('evaluate'); p.add_argument('--model',default='models/base'); p.add_argument('--adapter'); p.add_argument('--references',default='data/prepared/test.reference.jsonl'); p.add_argument('--output',required=True)
    p.add_argument('--limit',type=int); p.add_argument('--max-tokens',type=int,default=64); p.add_argument('--seed',type=int,default=42); p.set_defaults(fn=evaluate)
    p = sub.add_parser('compare'); p.add_argument('--baseline',default='runs/baseline'); p.add_argument('--adapted',default='runs/adapted'); p.add_argument('--output',default='runs/comparison.md'); p.set_defaults(fn=compare)
    from .benchmark import setup_models, run_benchmark, report_benchmark
    p = sub.add_parser('setup-benchmark-models'); p.add_argument('--output',default='models/benchmark'); p.set_defaults(fn=setup_models)
    p = sub.add_parser('benchmark'); p.add_argument('--models',default='models/benchmark'); p.add_argument('--config',default='configs/m1.json'); p.add_argument('--output',required=True); p.add_argument('--steps',type=int); p.add_argument('--eval-limit',type=int); p.add_argument('--seed',type=int); p.add_argument('--order',choices=('lora-first','qlora-first'),default='lora-first'); p.set_defaults(fn=run_benchmark)
    p = sub.add_parser('benchmark-report'); p.add_argument('--output',required=True); p.set_defaults(fn=report_benchmark)
    args = parser.parse_args()
    try:
        args.fn(args)
    except (ValueError, RuntimeError) as error:
        parser.exit(1, f'Error: {error}\n')

if __name__ == '__main__':
    main()
