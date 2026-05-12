# Human Preference-Based RL — Tuning Guide

This document explains how to tune `train_human_rl.py` for optimal learning outcomes when fine-tuning the Gomoku AI from human feedback.

---

## Overview

The Human RL pipeline has two phases:

1. **Reward Model (RM) Training** — learns to rank actions so human-preferred moves score higher than rejected ones
2. **PPO Fine-Tuning** — updates the policy to favor high-reward actions without forgetting how to play

Each phase has different sensitivities. Tuning them correctly is the difference between a subtly improved AI and one that forgets how to play.

---

## Phase 1: Reward Model Training

### Goal
Learn to rank actions such that human-preferred moves score higher than rejected ones.

### Key Parameters

| Parameter | Default | What it controls | Tuning advice |
|---|---|---|---|
| `--epochs-rm` | 20 | How many passes over all preference pairs | **Data-dependent.** With <100 preferences, use 20–50 epochs. With 1,000+, 10–20 is enough. Watch for loss → 0 (overfitting). |
| `--lr-rm` | 1e-4 | Reward model learning rate | Use **1e-4 to 3e-4** for fast convergence. Lower (5e-5) if you see unstable RM loss jumps. |
| `--batch-size` | 64 | Preferences per RM gradient step | **Match to your data size.** With <200 preferences, drop to 16 or 32 so you get more gradient steps per epoch. With 2,000+, 64–128 is fine. |

### What Good RM Training Looks Like

```
[RM] Epoch 1/20  | Loss: 0.65
[RM] Epoch 5/20  | Loss: 0.15
[RM] Epoch 10/20 | Loss: 0.03
[RM] Epoch 20/20 | Loss: 0.002
```

- Loss should **monotonically decrease**.
- If loss plateaus early (e.g., epoch 3 at 0.5), your preferences may be noisy or contradictory.
- If loss hits exactly 0.0000 before epoch 10, you are **overfitting** — the RM memorized your small dataset. Reduce `--epochs-rm`.

### Overfitting Fix

```bash
python train_human_rl.py ... --epochs-rm 10 --lr-rm 5e-5
```

---

## Phase 2: PPO Fine-Tuning

### Goal
Update the policy to favor high-reward actions without forgetting how to play Gomoku.

### Key Parameters

| Parameter | Default | What it controls | Tuning advice |
|---|---|---|---|
| `--epochs-ppo` | 10 | Outer epochs (full passes over replay buffer steps) | **The most critical parameter for avoiding catastrophic forgetting.** Start with 5–10. Increase to 20 only if PPO loss is still improving and entropy hasn't collapsed. |
| `--ppo-steps` | 100 | Steps sampled from replay buffer per epoch | Total PPO updates = `epochs-ppo` × `ppo-steps` × `ppo-inner-epochs`. Default gives 4,000 updates. More updates = stronger RL effect but more forgetting. |
| `--lr-ppo` | 1e-5 | Policy learning rate | **Keep low.** 1e-5 is conservative. Use 5e-6 if you notice the policy degrading (loses to old checkpoint). Never go above 1e-4. |
| `--ppo-inner-epochs` | 4 | PPO updates per sampled batch | Standard PPO uses 3–10. Higher = more aggressive optimization per batch. Use 2–4 for conservative updates, 6–10 for stronger alignment. |
| `--ppo-clip` | 0.2 | PPO clipping epsilon | 0.1 = very conservative (small policy changes). 0.2 = standard. 0.3 = more aggressive. |
| `--ppo-entropy-coef` | 0.01 | Entropy bonus weight | Prevents policy collapse to deterministic play. Increase to **0.02–0.05** if entropy drops below 2.0. Decrease to 0.005 if you want more exploitation. |
| `--ppo-value-coef` | 0.5 | Value head loss weight | Usually leave at 0.5. Lower (0.1–0.3) if value loss dominates and policy barely changes. |

### What Good PPO Looks Like

```
[PPO] Epoch 1/10 | Policy: -0.011 | Value: 1.14 | Entropy: 3.70 | Ratio: 1.000
[PPO] Epoch 5/10 | Policy: -0.010 | Value: 0.98 | Entropy: 3.87 | Ratio: 1.000
[PPO] Epoch 10/10| Policy: -0.009 | Value: 0.78 | Entropy: 3.57 | Ratio: 0.999
```

**Watch these metrics:**
- **Entropy** dropping fast → policy is collapsing. Increase `--ppo-entropy-coef` or reduce `--epochs-ppo`.
- **Ratio** staying at 1.0 → PPO isn't updating. Try higher `--lr-ppo` or more `--ppo-steps`.
- **Policy loss** near 0 → no learning. Check that RM rewards have variance (not all the same).

---

## The Catastrophic Forgetting Problem

PPO can destroy your carefully-trained Gomoku policy if you train too hard.

**Symptoms:**
- Web UI: AI makes obviously bad moves (suicidal moves, ignores threats)
- Self-play win rate vs old checkpoint drops below 40%

**Prevention strategies:**

| Strategy | How |
|---|---|
| Very low LR | `--lr-ppo 5e-6` |
| Fewer updates | `--epochs-ppo 5 --ppo-steps 50` |
| Mix original value | `--original-value-weight 0.3` (keeps some original game outcome signal) |
| Conservative clip | `--ppo-clip 0.1` |

---

## The `--original-value-weight` Secret Weapon

This blends RM rewards with the original self-play game outcomes:

| Setting | Effect |
|---|---|
| `--original-value-weight 0.0` (default) | Pure human preference. Risky if preferences are sparse. |
| `--original-value-weight 0.3` | 70% RM + 30% original value. Safer; preserves base playing strength. |
| `--original-value-weight 0.5` | Balanced. Good when you have <100 preferences. |

**Recommendation:** Start with `0.2–0.3` for your first runs. Once you have 500+ preferences, drop to `0.0`.

---

## Data-Dependent Strategies

### You have < 50 preferences

```bash
python train_human_rl.py \
    --checkpoint data/gomoku_checkpoints/checkpoint_115012 \
    --epochs-rm 50 \
    --lr-rm 5e-5 \
    --batch-size 16 \
    --epochs-ppo 5 \
    --ppo-steps 30 \
    --lr-ppo 5e-6 \
    --original-value-weight 0.5
```

- RM will overfit; keep it simple.
- PPO should be very conservative — you're data-starved.

### You have 200–500 preferences

```bash
python train_human_rl.py \
    --checkpoint data/gomoku_checkpoints/checkpoint_115012 \
    --epochs-rm 20 \
    --batch-size 32 \
    --epochs-ppo 10 \
    --ppo-steps 100 \
    --lr-ppo 1e-5 \
    --ppo-entropy-coef 0.02
```

- The sweet spot. Standard settings work well.

### You have 1,000+ preferences

```bash
python train_human_rl.py \
    --checkpoint data/gomoku_checkpoints/checkpoint_115012 \
    --epochs-rm 15 \
    --batch-size 128 \
    --epochs-ppo 15 \
    --ppo-steps 200 \
    --lr-ppo 1e-5 \
    --ppo-clip 0.15 \
    --ppo-inner-epochs 6
```

- You can afford more aggressive PPO.
- Larger batch size for RM stabilizes training.

---

## Complete Tuning Recipes

### Conservative (safe, minimal forgetting)

```bash
python train_human_rl.py \
    --checkpoint data/gomoku_checkpoints/checkpoint_115012 \
    --epochs-rm 20 \
    --epochs-ppo 5 \
    --ppo-steps 50 \
    --lr-ppo 5e-6 \
    --original-value-weight 0.3 \
    --ppo-entropy-coef 0.03 \
    --batch-size 32
```

### Aggressive (strong alignment, higher forgetting risk)

```bash
python train_human_rl.py \
    --checkpoint data/gomoku_checkpoints/checkpoint_115012 \
    --epochs-rm 20 \
    --epochs-ppo 20 \
    --ppo-steps 200 \
    --lr-ppo 1e-5 \
    --original-value-weight 0.0 \
    --ppo-inner-epochs 8 \
    --batch-size 64
```

### Balanced (recommended default)

```bash
python train_human_rl.py \
    --checkpoint data/gomoku_checkpoints/checkpoint_115012 \
    --epochs-rm 20 \
    --epochs-ppo 10 \
    --ppo-steps 100 \
    --lr-ppo 1e-5 \
    --original-value-weight 0.2 \
    --ppo-entropy-coef 0.02 \
    --batch-size 64
```

---

## Validation Workflow

After each Human RL run, test before merging back into the main pipeline:

```bash
# 1. Load the Human RL model in the web UI
python -m uvicorn web.gomoku_server:app --host 0.0.0.0 --port 8188
# Load data/human_rl/best_model.pt, play 5 games against it
```

| Outcome | Action |
|---|---|
| Plays **worse** than before | PPO was too aggressive. Reduce `--epochs-ppo`, `--lr-ppo`, or increase `--original-value-weight`. |
| Plays **the same** | Preferences didn't affect policy. Increase `--epochs-ppo`, `--lr-ppo`, or reduce `--original-value-weight`. |
| Plays **better / more human-like** | Success! Merge with `apply_human_rl.py`. |

Only merge with `apply_human_rl.py` after it passes the eye test.

---

## Full Human RL Workflow

```bash
# 1. Collect preferences via web UI
python -m uvicorn web.gomoku_server:app --host 0.0.0.0 --port 8188

# 2. (Optional) Augment preferences by rotation
python augment_preferences.py

# 3. Train Human RL
python train_human_rl.py \
    --checkpoint data/gomoku_checkpoints/checkpoint_115012 \
    --epochs-rm 20 --epochs-ppo 10 --batch-size 64

# 4. Validate in web UI (load data/human_rl/best_model.pt)

# 5. Merge back to main training pipeline
python apply_human_rl.py \
    --source data/gomoku_checkpoints/ \
    --human-rl data/human_rl/best_model.pt \
    --output data/gomoku_checkpoints

# 6. Resume regular training
python train_gomoku.py --resume --steps 100000 --workers 128 --fp16
```
