# FableForge Training Data & Notebooks

Training data and Unsloth Colab/Kaggle notebooks for fine-tuning the FableForge AI model family.

## Models

| Model | Base | Size | Training | Notebook |
|-------|------|------|----------|----------|
| ShellWhisperer-1.5B | Qwen2.5-Coder-1.5B-Instruct | 1.5B | SFT (shell/cmd) | [ShellWhisperer_1.5B_Unsloth_Optimized.ipynb](ShellWhisperer_1.5B_Unsloth_Optimized.ipynb) |
| ReasonCritic-7B | Qwen2.5-Coder-7B-Instruct | 7B | DPO (code critique) | [ReasonCritic_7B_finetune.ipynb](ReasonCritic_7B_finetune.ipynb) |
| FableForge-14B | Qwen2.5-Coder-14B-Instruct | 14B | SFT (agent) | [FableForge_14B_finetune.ipynb](FableForge_14B_finetune.ipynb) |

## Training Data

| File | Format | Examples | Description |
|------|--------|----------|-------------|
| `shellwhisperer_train.jsonl` | Alpaca + ChatML | 134 | Shell commands and multi-turn conversations |
| `shellwhisperer_val.jsonl` | Alpaca + ChatML | 10 | Validation set |
| `reasoncritic_dpo_train.jsonl` | DPO (prompt/chosen/rejected) | 28 | Code critique preference pairs |
| `reasoncritic_dpo_val.jsonl` | DPO | 7 | Validation set |
| `fableforge_sft_train.jsonl` | ChatML (messages) | 9 | Multi-turn coding agent conversations |
| `fableforge_sft_val.jsonl` | ChatML | 1 | Validation set |

## Quick Start (Kaggle/Colab)

1. Open `ShellWhisperer_1.5B_Unsloth_Optimized.ipynb` in Kaggle or Colab
2. Enable GPU (T4) and Internet access
3. Upload `shellwhisperer_train.jsonl` and `shellwhisperer_val.jsonl` to session storage
4. Run all cells — training takes ~15 min on T4

## Benchmark Results (MPS baseline)

First training run on Apple MPS (r=8, 2 epochs, fp16):
- Base model avg score: 1.93/2.50
- Fine-tuned avg score: 1.88/2.50 (-2.6%)
- The optimized Unsloth notebook uses r=16, 5 epochs, 4-bit QLoRA for better results

## HuggingFace

Models will be published to: https://huggingface.co/fableforge-ai
