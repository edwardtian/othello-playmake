# Othello AI Playground — User Manual

A self-learning Othello AI built with AlphaZero-style reinforcement learning (ResNet + MCTS), featuring a web-based playground for testing and monitoring.

---

## Table of Contents
1. [Quick Start](#quick-start)
2. [Project Structure](#project-structure)
3. [Installation](#installation)
4. [Web Playground](#web-playground)
5. [Training the AI](#training-the-ai)
6. [Resuming Training](#resuming-training)
7. [Loading Checkpoints](#loading-checkpoints)
8. [Testing](#testing)
9. [Troubleshooting](#troubleshooting)
10. [Hardware & Performance Notes](#hardware--performance-notes)

---

## Quick Start

```powershell
# 1. Navigate to project
cd F:\Workspace\flip-ai

# 2. Start the web server
python -m uvicorn web.server:app --host 0.0.0.0 --port 8000

# 3. Open browser
# http://localhost:8000

# 4. (Optional) Start training from command line
python train.py --steps 100000 --num-simulations 400
```

---

## Project Structure

```
F:\Workspace\flip-ai\
├── game\                    # Othello game engine
│   └── othello.py
├── ai\                      # AI core: model, MCTS, trainer
│   ├── model.py             # ResNet policy-value network
│   ├── mcts.py              # Monte Carlo Tree Search
│   ├── self_play.py         # Self-play game generation
│   ├── trainer.py           # Training loop & replay buffer
│   └── evaluate.py          # Champion vs Challenger arena
├── web\                     # Web application
│   ├── server.py            # FastAPI backend
│   ├── training_manager.py  # Training process control
│   └── static\              # Frontend (HTML/CSS/JS)
│       ├── index.html
│       ├── css\style.css
│       └── js\app.js
├── tests\                   # Test suite (86 tests)
├── data\                    # Checkpoints, replay buffer, logs
│   ├── checkpoints\
│   ├── replay_buffer\
│   └── logs\
├── train.py                 # CLI training script
├── resume_train.py          # Resume training from checkpoint
├── environment.yml          # Conda environment spec
└── SPEC.md                  # Full implementation specification
```

---

## Installation

### Prerequisites
- **OS**: Windows 10 Pro (tested)
- **Python**: 3.10+ (via Miniconda/Anaconda recommended)
- **Hardware**: CPU sufficient for training; GPU accelerates if compatible
- **Storage**: ~5GB for environment, ~20GB per checkpoint + replay buffer

### Step 1: Create Conda Environment

```powershell
# From project root
cd F:\Workspace\flip-ai

# Create environment from spec
conda env create -f environment.yml

# Or create manually
conda create -n othello-ai python=3.10
conda activate othello-ai
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install fastapi uvicorn websockets python-multipart pytest tqdm tensorboard
```

### Step 2: Verify Installation

```powershell
python -m pytest tests/ -q
# Should show: 86 passed
```

### Step 3: Configure Data Storage (Optional)

All large files default to `F:\Workspace\flip-ai\data\`. To change:

Edit `train.py` and `resume_train.py`:
```python
checkpoint_dir='F:\Your\Custom\Path\checkpoints'
log_dir='F:\Your\Custom\Path\logs'
```

---

## Web Playground

### Starting the Server

```powershell
conda activate othello-ai
cd F:\Workspace\flip-ai
python -m uvicorn web.server:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000` in your browser.

### Game Modes

| Mode | Description |
|---|---|
| **Play as Black** | You are Black (first player); AI is White |
| **Play as White** | You are White; AI is Black (moves first) |
| **Watch AI vs AI** | Both players are AI; auto-plays with delay |

### Making Moves

1. Select your game mode from the dropdown
2. Click **New Game** to start
3. Click a square on the board to place your piece
4. Valid moves are shown as translucent dots
5. If it's AI's turn, it will compute automatically

### AI Thinking Heatmap

After each AI move, a **thinking panel** appears showing:
- **Visit count heatmap**: Color-coded squares (red = high visits, green = low visits)
- **Position value**: A scalar from -1 to +1 estimating win probability

This reveals which moves the AI considered most promising.

### Training Dashboard

The bottom panel shows live training metrics:

| Control | Action |
|---|---|
| **Start Training** | Begins self-play + training in background |
| **Stop Training** | Pauses training and saves checkpoint |
| **Resume Training** | Continues from last checkpoint |

Charts update in real-time via WebSocket:
- **Loss Chart**: Policy loss, value loss, total loss
- **ELO Chart**: Champion model's ELO rating over time

### Loading a Checkpoint for Play

1. Train the AI for some steps (or use an existing checkpoint)
2. In the web UI, select a checkpoint from the **AI Model** dropdown
3. Click **Load**
4. The AI will now use that checkpoint's neural network for moves

---

## Training the AI

### From Command Line (Recommended for Long Runs)

```powershell
conda activate othello-ai
cd F:\Workspace\flip-ai

# Basic training
python train.py --steps 100000

# Full configuration
python train.py \
  --steps 100000 \
  --games-per-iter 10 \
  --checkpoint-interval 1000 \
  --eval-interval 5000 \
  --num-simulations 400 \
  --batch-size 512 \
  --buffer-capacity 500000 \
  --lr 0.001 \
  --num-blocks 20 \
  --num-channels 256 \
  --device auto \
  --checkpoint-dir data/checkpoints \
  --log-dir data/logs
```

### Training Parameters

| Parameter | Default | Description |
|---|---|---|
| `--steps` | 100000 | Total training steps |
| `--games-per-iter` | 10 | Self-play games per training iteration |
| `--checkpoint-interval` | 1000 | Save checkpoint every N steps |
| `--eval-interval` | 5000 | Evaluate vs champion every N steps |
| `--num-simulations` | 400 | MCTS simulations per move |
| `--batch-size` | 512 | Training batch size |
| `--buffer-capacity` | 500000 | Replay buffer size (positions) |
| `--lr` | 0.001 | Learning rate |
| `--num-blocks` | 20 | ResNet blocks |
| `--num-channels` | 256 | ResNet channels |
| `--device` | auto | `cuda`, `cpu`, or `auto` |

### Training Output

During training, you'll see logs like:

```
Step 100 | Loss: 2.3412 (P: 1.8234, V: 0.5178) | LR: 0.001000 | Buffer: 520
Step 200 | Loss: 2.1021 (P: 1.6543, V: 0.4478) | LR: 0.000999 | Buffer: 1040
...
Evaluation: Challenger win rate: 52.5% | Champion ELO: 1500 | Challenger ELO: 1500
```

Checkpoints are saved to `data/checkpoints/checkpoint_{step}/` containing:
- `model.pt` — Neural network weights
- `optimizer.pt` — Optimizer state
- `replay_buffer.pt` — Training data
- `meta.json` — Step count, games played, timestamp

---

## Resuming Training

If training is interrupted or you want to continue from a checkpoint:

```powershell
python resume_train.py --steps 200000
```

This automatically:
1. Finds the most recent checkpoint in `data/checkpoints/`
2. Loads model weights, optimizer state, and replay buffer
3. Continues training toward the new `--steps` target

You can also resume from the web UI by clicking **Resume Training**.

---

## Loading Checkpoints

### From Web UI

1. Start the server: `python -m uvicorn web.server:app --port 8000`
2. Open `http://localhost:8000`
3. Select checkpoint from **AI Model** dropdown
4. Click **Load**
5. Start a new game to play against the loaded model

### From Python API

```python
import torch
from ai.model import create_model

model = create_model(num_blocks=20, num_channels=256, device='cpu')
checkpoint = torch.load('data/checkpoints/checkpoint_1000/model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
```

### From CLI for Evaluation

```python
from ai.model import create_model
from ai.evaluate import evaluate_challenger

champion = create_model(device='cpu')
challenger = create_model(device='cpu')

# Load challenger checkpoint
checkpoint = torch.load('data/checkpoints/checkpoint_5000/model.pt')
challenger.load_state_dict(checkpoint['model_state_dict'])

# Evaluate
is_better, results = evaluate_challenger(champion, challenger, num_games=200)
print(f"Win rate: {results['challenger_win_rate']:.2%}")
```

---

## Testing

### Run All Tests

```powershell
python -m pytest tests/ -v
```

### Run Specific Test Files

```powershell
python -m pytest tests/test_othello.py -v      # Game engine
python -m pytest tests/test_model.py -v        # Neural network
python -m pytest tests/test_mcts.py -v         # MCTS
python -m pytest tests/test_trainer.py -v      # Training
python -m pytest tests/test_server.py -v       # Web API
python -m pytest tests/test_integration.py -v  # End-to-end
```

### Manual Model Test

```python
from ai.model import create_model
from game.othello import OthelloGame
import torch

model = create_model(device='cpu')
game = OthelloGame()

# Get AI move
state = torch.from_numpy(game.get_state_planes()).float().unsqueeze(0)
policy, value = model.predict(state)
print(f"Best move: {policy.argmax().item()}")
print(f"Position value: {value.item():.3f}")
```

---

## Troubleshooting

### PyTorch can't find CUDA / DLL errors

**Symptom**: `RuntimeError: CUDA error` or `DLL load failed`

**Solutions**:
1. Ensure NVIDIA drivers are up to date (check with `nvidia-smi`)
2. On Windows, PyTorch cu128 wheels may have compatibility issues; use cu121:
   ```powershell
   pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
   ```
3. Verify PATH includes cuDNN:
   ```powershell
   # Should show cudnn64_9.dll
   python -c "import ctypes; ctypes.CDLL('cudnn64_9.dll')"
   ```

### WebSocket disconnects during training

**Symptom**: Training dashboard stops updating

**Solution**: The frontend auto-reconnects after 3 seconds. If not:
1. Check server is running: `python -m uvicorn web.server:app --port 8000`
2. Refresh the browser page
3. Check server logs for errors

### Training is very slow

**Causes & Fixes**:

| Cause | Fix |
|---|---|
| CPU training | Expected on CPU. Consider reducing `--num-simulations` to 100 for faster iteration |
| Large replay buffer | Reduce `--buffer-capacity` to 100000 |
| Deep network | Use `--num-blocks 10 --num-channels 128` for faster but weaker play |
| MCTS bottleneck | Reduce `--num-simulations` or use batched inference (future optimization) |

### Checkpoint loading fails

**Symptom**: `FileNotFoundError` or `KeyError: 'model_state_dict'`

**Solutions**:
1. Verify checkpoint path exists
2. Check `meta.json` for corruption
3. Load just the state dict if full checkpoint is incompatible:
   ```python
   state_dict = torch.load('model.pt')
   if 'model_state_dict' in state_dict:
       model.load_state_dict(state_dict['model_state_dict'])
   else:
       model.load_state_dict(state_dict)
   ```

### Windows multiprocessing hangs

**Symptom**: Training freezes at startup

**Fix**: Ensure all entry points use `if __name__ == '__main__':` guard. This is already implemented in `train.py` and `resume_train.py`.

---

## Hardware & Performance Notes

### Recommended Configuration for Your Hardware

Your system: **Threadripper 3970X (32-core) + 256GB RAM + RTX Pro 6000 Blackwell**

| Setting | Recommended | Reason |
|---|---|---|
| `--num-blocks` | 20 | Deeper network leverages GPU |
| `--num-channels` | 256 | Balances capacity and speed |
| `--batch-size` | 2048–4096 | Fills GPU memory without OOM |
| `--num-simulations` | 400–800 | Strong play; scale with CPU cores |
| `--games-per-iter` | 10–20 | Sufficient data per training step |
| `--buffer-capacity` | 1,000,000–2,000,000 | Fits easily in 256GB RAM |

### Performance Expectations

| Hardware | Self-Play Games/hour | Training Steps/hour |
|---|---|---|
| CPU (3970X) | ~100–200 | ~500–1000 |
| GPU (with full sm_120 support) | ~500–1000 | ~3000–5000 |

> Note: Your GPU will be detected but may run via CPU fallback until a PyTorch build with native Windows sm_120 support is available. Training is still fully functional and will produce strong play.

### Storage Planning

| Checkpoint | Size |
|---|---|
| Model + Optimizer | ~400 MB |
| Replay Buffer (500K) | ~15 GB |
| Full Checkpoint | ~15–20 GB |

With `--checkpoint-interval 1000`, every 1000 steps consumes ~15–20GB. Manage disk space by archiving or deleting older checkpoints.

---

## Architecture Overview

For developers who want to understand or modify the system:

```
Self-Play Loop (Training)
=========================
1. MCTS searches current board state (800 simulations)
   - Neural net provides policy priors + position values
   - Dirichlet noise at root for exploration
2. Sample action from MCTS visit distribution
3. Play move, repeat until game over
4. Store all (state, policy, outcome) in replay buffer

Training Loop
=============
1. Sample batch from replay buffer
2. Forward pass: get policy logits + value
3. Compute loss: policy CE + value MSE
4. Backprop + AdamW optimizer step
5. Every 1000 steps: save checkpoint
6. Every 5000 steps: evaluate vs champion
   - If win rate > 55%: become new champion
   - Update ELO ratings
```

---

## License & Attribution

Built with:
- PyTorch (deep learning)
- FastAPI / Uvicorn (web backend)
- Chart.js (live charts)
- AlphaZero architecture inspiration

---

## Support

For issues or questions:
1. Check this manual's Troubleshooting section
2. Run tests: `python -m pytest tests/ -v`
3. Review logs in `data/logs/`
4. Consult `SPEC.md` for implementation details
