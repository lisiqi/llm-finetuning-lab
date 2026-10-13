"""Matched LoRA vs quantized-LoRA orchestration and report validation."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from .core import digest, write_json

MODEL = 'Qwen/Qwen2.5-0.5B-Instruct'
REVISION = '7ae557604adf67be50417f59c2c2f167def9a775'


def command(*args):
    subprocess.run([sys.executable, '-m', *map(str,args)], check=True)


def setup_models(args):
    from huggingface_hub import snapshot_download
    root=Path(args.output)
    for method in ('lora','qlora'):
        if (root/method).exists():
            raise ValueError('Model output exists; choose a fresh output directory')
    snapshot=snapshot_download(MODEL, revision=REVISION)
    for method in ('lora','qlora'):
        flags=['-q','--q-bits','4'] if method=='qlora' else []
        command('mlx_lm','convert','--hf-path',snapshot,'--mlx-path',root/method,*flags)
        write_json(root/method/'source.json',{'repo':MODEL,'revision':REVISION,'quantized':method=='qlora','bits':4 if method=='qlora' else None})


def validate_models(paths):
    sources=[json.loads((p/'source.json').read_text()) for p in paths]
    for key in ('repo','revision'):
        if sources[0][key]!=sources[1][key]: raise ValueError('Base model sources differ')
    if sources[0]['quantized'] or not sources[1]['quantized']:
        raise ValueError('Expected unquantized LoRA and quantized LoRA models')
    configs=[json.loads((p/'config.json').read_text()) for p in paths]
    if configs[0].get('quantization') or not configs[1].get('quantization'):
        raise ValueError('Model configs do not match claimed quantization')
    for key in ('model_type','hidden_size','intermediate_size','num_hidden_layers','num_attention_heads','vocab_size'):
        if configs[0].get(key)!=configs[1].get(key): raise ValueError(f'Mismatched model architecture: {key}')
    for name in ('tokenizer.json','tokenizer_config.json'):
        if digest(paths[0]/name)!=digest(paths[1]/name): raise ValueError('Tokenizer mismatch')
    return sources


def run_benchmark(args):
    root=Path(args.output)
    if root.exists(): raise ValueError('Benchmark directory exists; choose a fresh output')
    paths=[Path(args.models)/m for m in ('lora','qlora')]
    sources=validate_models(paths)
    config=json.loads(Path(args.config).read_text())
    if args.steps is not None:
        if args.steps<=0: raise ValueError('Steps must be positive')
        config['iters']=args.steps
    if args.eval_limit is not None and args.eval_limit<=0: raise ValueError('Evaluation limit must be positive')
    if args.seed is not None: config['seed']=args.seed
    order=['lora','qlora'] if args.order=='lora-first' else ['qlora','lora']
    root.mkdir(parents=True)
    write_json(root/'benchmark.json', {'model_sources':sources,'config':config,
        'eval_limit':args.eval_limit,'order':order,'smoke':args.steps is not None or args.eval_limit is not None})
    references=Path(config['data'])/'test.reference.jsonl'
    for method in order:
        model=Path(args.models)/method
        method_root=root/method
        adapter=method_root/'adapter'
        local=config | {'model':str(model),'adapter_path':str(adapter)}
        # Report every step so completion-token throughput has no partial-window ambiguity.
        local['steps_per_report']=1
        config_path=method_root/'config.json'
        write_json(config_path,local)
        eval_args=['--model',model,'--references',references,'--seed',str(config['seed'])]
        if args.eval_limit: eval_args+=['--limit',str(args.eval_limit)]
        command('finetuning_lab.cli','evaluate',*eval_args,'--output',method_root/'baseline')
        command('finetuning_lab.cli','train','--config',config_path)
        command('finetuning_lab.cli','evaluate',*eval_args,'--adapter',adapter,'--output',method_root/'adapted')
    report_benchmark(argparse.Namespace(output=str(root)))


def report_benchmark(args):
    from .cli import compare
    root=Path(args.output)
    validate_models([Path(json.loads((root/m/'config.json').read_text())['model']) for m in ('lora','qlora')])
    configs=[json.loads((root/m/'config.json').read_text()) for m in ('lora','qlora')]
    ignored={'model','adapter_path'}
    if {k:v for k,v in configs[0].items() if k not in ignored}!={k:v for k,v in configs[1].items() if k not in ignored}:
        raise ValueError('Training settings differ')
    manifests=[json.loads((root/m/'adapter/experiment.json').read_text()) for m in ('lora','qlora')]
    for key in ('data_manifest_sha256','training_data_sha256','environment'):
        if manifests[0][key]!=manifests[1][key]: raise ValueError(f'Training mismatch: {key}')
    for m in ('lora','qlora'):
        compare(argparse.Namespace(baseline=str(root/m/'baseline'),adapted=str(root/m/'adapted'),output=str(root/m/'quality-comparison.md')))
    evals=[json.loads((root/m/'baseline/manifest.json').read_text()) for m in ('lora','qlora')]
    for key in ('ids','references_sha256','prompt_sha256','seed','max_tokens','temperature','environment'):
        if evals[0][key]!=evals[1][key]: raise ValueError(f'Evaluation mismatch: {key}')
    for index,method in enumerate(('lora','qlora')):
        if manifests[index]['model_files_sha256'] != evals[index]['model_files_sha256']:
            raise ValueError('Training and evaluation model files differ')
    metrics=[json.loads((root/m/'adapter/training_metrics.json').read_text()) for m in ('lora','qlora')]
    if metrics[0]['trainable_parameter_shapes']!=metrics[1]['trainable_parameter_shapes']:
        raise ValueError('Adapted modules/shapes differ')
    peak=[m['training_peak_allocated_bytes'] for m in metrics]
    saving=(1-peak[1]/peak[0])*100
    lines=['# LoRA vs MLX quantized LoRA','', '| Training measure | LoRA | Quantized LoRA |','|---|---:|---:|']
    for key in ('loaded_base_active_bytes','training_peak_allocated_bytes','training_wall_seconds','loss_tokens_per_compute_second','trainable_parameters'):
        lines.append(f"| {key} | {metrics[0][key]:.3f} | {metrics[1][key]:.3f} |")
    for label,paths in [('base_storage_bytes',[Path(c['model']) for c in configs]),('adapter_weights_bytes',[root/m/'adapter' for m in ('lora','qlora')])]:
        sizes=[sum(p.stat().st_size for p in folder.glob('*.safetensors') if label=='base_storage_bytes' or p.name=='adapters.safetensors') for folder in paths]
        lines.append(f'| {label} | {sizes[0]} | {sizes[1]} |')
    lines+=['',f'Measured peak allocation reduction: {saving:.2f}%. Negative means quantized LoRA used more memory.', '',
        '## Four-way quality evaluation','', '| Configuration | Answer EM | Answer F1 | Abstention precision | Abstention recall |', '|---|---:|---:|---:|---:|']
    for method in ('lora','qlora'):
        for stage in ('baseline','adapted'):
            q=json.loads((root/method/stage/'metrics.json').read_text())
            vals=[q[k] for k in ('answerable_exact_match','answerable_token_f1','abstention_precision','abstention_recall')]
            lines.append('| '+method+' '+stage+' | '+' | '.join('undefined' if v is None else f'{v:.4f}' for v in vals)+' |')
    smoke=json.loads((root/'benchmark.json').read_text())['smoke']
    lines+=['', 'SMOKE TEST ONLY: do not use these numbers as a generalization claim.' if smoke else 'Single-seed benchmark: repeat with reversed order/multiple seeds before broad claims.',
        '',metrics[0]['measurement_scope'],
        'MLX allocation excludes system RAM and allocator cache. This tests MLX quantization, not original NF4/bitsandbytes QLoRA.',
        'Training order: '+', '.join(json.loads((root/'benchmark.json').read_text())['order'])+'. Thermal effects and warm caches can affect speed. Both run in separate processes.']
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    write_json(root/'summary.json', {'training':dict(zip(('lora','qlora'),metrics)),'peak_allocation_reduction_percent':saving,'smoke':smoke})
    print('\n'.join(lines))
