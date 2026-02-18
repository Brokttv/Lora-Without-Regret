# Lora-Without-Regret
Reproducing Lora Low Regret blog from scratch (no HF) and investigating the LR 10x ratio.


## Preliminary Investigation: Understanding the 10× Learning Rate Ratio

I conducted preliminary experiments to investigate the origin of the `10×` learning rate ratio observed in LoRA training.

### Background

It is important to note that the consistent ratio reported in the blog  stems from the `1/r` and `alpha` scaling factors. We know that `1/r` implicitly scales the learning rate by the layer width determined by rank `r`, ensuring updates' velocity remain invariant to width scaling as highlighted by Yang et al. in their [μP approach](https://blog.eleuther.ai/mutransfer/).

### Key Findings

The data shows that performance, as measured by either test accuracy or perplexity (depending on the task), is influenced by **adapters initialization**, **`alpha/r`** scale factor and **regime** defined by task nature.

**Standart configuration from Lora-Without-Regret blog:**
- 'A'initalized using uniform distributin and `B`is zero
- We use a constant `alpha` value of `32` and factor by `1/r`
- We set a fixed `lr` (no scheduler) used by both adapters
- We train  `Distil-bert-uncased` on a `10k` subset of AG-News (classifcation)

The results show that the optimal learning rate for all ranks is 10x higher than FullFT with test accuracy peaking at rank 32.


  <br>
  
<p align="center">
  <img src="assets/normal-lora-bert" width="750"/>
</p>
<br>

**Different regime:**

Now, we train `distilgpt2` on the `wikitext` dataset using the same configuration and data amount. We rely on test perplexity (`exp(NLL)`) for benchmarking.

We observe that performance here peaks at a higher rank `128` compared to previous setup, revealing that the optimal rank for is regime-dependent

  
  <br>
  <p align="center">
  <img src="assets/gpt3" width="700"/>
</p>
<br>

### Tweaking `alpha/r`

For the rest of experiments, We use `distilbert` and `ag-news` as they are base-line for all comparasions. 

Here, we increase the mutliplier `alpha/r` scaling adapters weights by a factor of 10  by setting `alpha = 10r`. 

<br>
  <p align="center">
  <img src="assets/bigger scale" width="750"/>
</p>
<br>

**We observe the following:**
- The optimal learning rate ratio is rank-dependent as it is `10x` at rank 16, `3x` for both rank 32 and 64, and only `1x` for rank 128
- Performance peaks at rank 16 as opposed to rank 32 shown previously
- The highest accuracy recorded is bigger than the one seen in the previous setup which uses `32`/r consistently across all ranks
  
Although the maximum accuracy belongs to rank 16 which happens to validate the 10x optimal ratio, we speculate that this is an **artifact** of the model and dataset used and **not linked to the ratio itself**. 

With that highlighted, we observe that a `lr` 10x bigger than FullFT's is not always linked to the best performance but rather is a result of particualr norm of the `AB` matrix impacted by **initialization distribution**, **`alpha/r`**, and finally **rank**.

**Important:**
From playing with `alpha`, it appears that when the the optimal learning rate is consistent across ranks, it is not due to `1/r` as Lora-Without-Regret blog claims but rather `alpha/r` with `alpha` being a constant.

**When we change the `alpha` dynamics, we observe two main trends:**
- Constant α → optimal LR ≈ invariant w.r.t. rank
- Constant α/r → optimal LR scales ↓ with rank
<br>


### Initialization

I change the standard initialization from `A` following an uniform distribution and `B` set to zero to both `A` and `B` following a **Gaussian** dsitribution.

As expected, not only the `10x` ratio falls, but training deteriorates drastically with a maximum accuracy capping around ~`25%`.

<br>
  <p align="center">
  <img src="assets/distrib-lora-bert" width="800"/>
</p>
<br>

### Conclusion
We emperically confirm that the `10x` ratio found repeatedly in Thinking Machines blog and elsewhere is controlled  by the initialization regime controlled by **`A` and `B` dsitributions**, **`alpha/r`**, and **rank**. We also show that  consistent optimal LR ratio across ranks is a result of `alpha` being a constant scaled by `1/r`.

**To try the experiemnts above, run the commands below**.




# LoRA Wihtout Regret Fine-tuning
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
---
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

## 📑 Citation

If you find this work useful, please cite it as:

```bibtex
@misc{brokttv2025vit,
  title        = {Lora-Without-Regret: Explaining the ratio between LoRA and FullFT learning rates},
  author       = {Brokttv},
  year         = {2025},
  howpublished = {\url{https://github.com/Brokttv/Lora-Without-Regret}},
}

