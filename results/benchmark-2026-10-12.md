# LoRA vs MLX quantized LoRA

| Training measure | LoRA | Quantized LoRA |
|---|---:|---:|
| loaded_base_active_bytes | 988065544.000 | 281161480.000 |
| training_peak_allocated_bytes | 1710174576.000 | 1085352052.000 |
| training_wall_seconds | 83.663 | 95.392 |
| loss_tokens_per_compute_second | 23.457 | 20.517 |
| trainable_parameters | 733184.000 | 733184.000 |
| base_storage_bytes | 988097730 | 278065225 |
| adapter_weights_bytes | 2938645 | 2938645 |

Measured peak allocation reduction: 36.54%. Negative means quantized LoRA used more memory.

## Four-way quality evaluation

| Configuration | Answer EM | Answer F1 | Abstention precision | Abstention recall |
|---|---:|---:|---:|---:|
| lora baseline | 0.5300 | 0.6706 | undefined | 0.0000 |
| lora adapted | 0.1200 | 0.1545 | 0.5366 | 0.8800 |
| qlora baseline | 0.2400 | 0.4824 | undefined | 0.0000 |
| qlora adapted | 0.0900 | 0.1030 | 0.5367 | 0.9500 |

Single-seed benchmark: repeat with reversed order/multiple seeds before broad claims.

Peak reset after base load and dataset setup; training includes adapter/optimizer creation, validation, and checkpoint saves. Wall time covers the same region. Throughput counts completion-loss tokens, not all input tokens.
MLX allocation excludes system RAM and allocator cache. This tests MLX quantization, not original NF4/bitsandbytes QLoRA.
Training order: lora, qlora. Thermal effects and warm caches can affect speed. Both run in separate processes.
