# Qwen3.5-9B Medical Reasoning SFT

This project fine-tunes a Qwen3.5-9B base model for Chinese medical reasoning using Transformers + TRL + PEFT (LoRA) + DeepSpeed.

## Features
- LoRA-based parameter-efficient fine-tuning for causal LM.
- Chinese CoT-style medical reasoning data processing pipeline.
- Multi-GPU training with DeepSpeed ZeRO-2.
- BF16 mixed-precision and gradient accumulation for memory efficiency.

## Main Script
- `train.py`: data preprocessing, LoRA wrapping, SFT training, and inference test.

## Environment
Key libraries used in this project:
- PyTorch
- transformers
- trl
- peft
- deepspeed
- datasets

## Quick Start
1. Configure model path in `train.py`.
2. Prepare DeepSpeed config (for example `ds_z2_offload_config1.json`).
3. Run training:

```bash
deepspeed --include 'localhost:0,1' train.py
```

## Notes
- Large local artifacts such as training checkpoints, logs, and datasets are excluded via `.gitignore`.
- If you need to publish model weights, consider using Hugging Face Hub or Git LFS.
