# Lora-Low-Regret
Reproducing Lora Low Regret blog from scratch (no HF) and investigating the LR 10x ratio.




# LoRA Fine-tuning
---
## Setup
```bash
pip install -r requirements.txt
```

## Requirements
```
torch
transformers
datasets
scikit-learn
```
---
## Usage

### Basic run (LoRA with default params)
```bash
python main.py
```

### Full fine-tuning
```bash
python main.py --full-finetune
```

### LoRA vs Full Fine-tuning examples

LoRA (default):
```bash
python main.py --epochs 10 --rank 64 --alpha 128
```

Full Fine-tuning:
```bash
python main.py --full-finetune --epochs 10
```

## Parameters

- `--lrs`: Learning rates to sweep (default: 1e-4 5e-4 1e-3)
- `--epochs`: Number of training epochs (default: 5)
- `--rank`: LoRA rank (default: 128)
- `--alpha`: LoRA alpha scaling (default: 256)
- `--batch_size`: Batch size (default: 32)
- `--seed`: Random seed (default: 42)
- `--full-finetune`: Use full fine-tuning instead of LoRA

## Output

Results saved to `results.json`:
```json
{
  "0.0001": {
    "best_val_loss": 0.1989,
    "test_acc": 0.9234,
    "train_losses": [...],
    "val_losses": [...]
  }
}
```
