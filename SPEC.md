# Othello AI — Implementation Specification

## 1. Environment & Setup
- **Primary**: Conda environment (`othello-ai`) with Python 3.11, PyTorch 2.3+ (CUDA 12.1), FastAPI, Uvicorn, NumPy, Chart.js (frontend CDN).
- **Storage**: All checkpoints, replay buffers, and logs reside under **`F:\Workspace\flip-ai\data\`**.
- **Hardware Utilization**:
  - **GPU**: ResNet-20 (or deeper) with 256 channels. Batch training size 4096–8192. Batched MCTS leaf evaluation (gather ~256–512 leaf nodes, evaluate in one GPU batch).
  - **CPU**: 48–64 parallel self-play worker processes (leaving cores for OS/FastAPI). Each worker runs MCTS tree traversal on CPU and sends leaf nodes to a central GPU inference queue.
  - **RAM**: Replay buffer sized to ~2M games (~15–20GB in memory), well within 256GB.

## 2. Project Structure
```
F:\Workspace\flip-ai\
├── environment.yml
├── train.py                  # CLI entry point (also used by web backend)
├── resume_train.py           # CLI helper to resume from latest checkpoint
├── game\
│   └── othello.py            # 8x8 board logic, move validation, flipping, terminal detection
├── ai\
│   ├── model.py              # ResNet policy-value network
│   ├── mcts.py               # GPU-batched MCTS with UCT + Dirichlet noise
│   ├── self_play.py          # Parallel self-play game generation
│   ├── trainer.py            # Training loop, loss computation, LR scheduling
│   └── evaluate.py           # Champion vs Challenger arena (self-play ELO)
├── web\
│   ├── server.py             # FastAPI + WebSocket server
│   ├── training_manager.py   # Process wrapper for Start/Stop/Resume control
│   ├── static\
│   │   ├── index.html
│   │   ├── css\style.css
│   │   └── js\app.js         # Board renderer, heatmap, Chart.js dashboards
│   └── checkpoints\          # Symlinked to F:\Workspace\flip-ai\data\checkpoints\
└── data\                     # LARGE FILES
    ├── checkpoints\          # model.pt, optimizer.pt, meta.json
    ├── replay_buffer\        # buffer.pt (periodically serialized)
    └── logs\                 # training_log.jsonl, elo_history.json
```

## 3. Core AI Components

### A. Game Engine (`game/othello.py`)
- State representation: `np.int8` array `(8,8)` — `0=empty, 1=black, 2=white`.
- Action space: 64 squares + 1 pass (index 64). Only returns legal moves.
- Reward signal: Game engine only reports **legal/illegal move** and **game over + winner**. No heuristic hints.

### B. Neural Network (`ai/model.py`)
- **Input**: 3 planes of shape `(8,8)` — current player stones, opponent stones, color to move.
- **Backbone**: 20 residual blocks, 256 filters, 3×3 conv, BatchNorm, ReLU.
- **Policy Head**: Conv(1×1) → Flatten → Linear(65) → LogSoftmax.
- **Value Head**: Conv(1×1) → Flatten → Linear(256) → ReLU → Linear(1) → Tanh.
- Total parameters: ~40–50M. Fits easily in GPU memory with massive batch sizes.

### C. MCTS (`ai/mcts.py`)
- **Tree traversal**: Pure CPU in each worker.
- **Leaf evaluation**: Workers push leaf node batches to a central `torch.Tensor` queue. A dedicated GPU thread evaluates batches of 512 nodes at once and returns (policy_prior, value) to workers.
- **Dirichlet noise**: Added to root node priors during self-play (`alpha=0.3`, `epsilon=0.25`).
- **Temperature**: τ=1 for first 30 moves, τ→0 afterwards to sharpen move selection.

### D. Self-Play (`ai/self_play.py`)
- 48–64 parallel processes. Each process plays a full game, appending `(state, mcts_policy, outcome)` tuples to a shared `Manager().list()` or writes to a memory-mapped buffer.
- Target: ~50k–100k moves generated per minute.

### E. Training Loop (`ai/trainer.py`)
- **Replay Buffer**: Circular buffer of the last 2M games. Sample uniformly.
- **Loss**: `L = L_policy (cross-entropy) + L_value (MSE) + c||θ||²`.
- **Optimizer**: AdamW, initial LR 1e-3, cosine annealing to 1e-5 over 1M steps.
- **Checkpointing**: Every 1000 training steps, save:
  - `model.pt`, `optimizer.pt`, `scheduler.pt`
  - `replay_buffer.pt`
  - `meta.json` (step count, games played, best model path).

### F. Evaluation (`ai/evaluate.py`)
- **Arena**: New model (Challenger) plays 200 games vs current Champion.
- **Promotion**: If Challenger win rate > 55%, it becomes the new Champion.
- **ELO**: Track relative ELO in `data/logs/elo_history.json`. No random baseline — pure self-play improvement curve.

## 4. Web Playground Design

### FastAPI Backend (`web/server.py`)
| Endpoint / Socket | Purpose |
|---|---|
| `POST /api/game/new` | Start game (mode: human-black, human-white, ai-vs-ai) |
| `POST /api/game/move` | Human submits move; server validates and updates state |
| `POST /api/game/ai_move` | AI computes move via MCTS (800 simulations). Returns move + thinking data |
| `GET /api/game/state` | Current board, legal moves, turn, game over status |
| `POST /api/ai/load_checkpoint/{id}` | Load a specific checkpoint for testing |
| `WebSocket /ws/training` | Streams live metrics: policy loss, value loss, ELO, games/sec, GPU util |
| `POST /api/training/start` | Spawns/resumes training process |
| `POST /api/training/stop` | Gracefully pauses training (saves checkpoint) |
| `GET /api/training/status` | Returns `idle`, `running`, `paused` + current step & ELO |

### Frontend (`web/static/`)
- **Board**: HTML5 Canvas or CSS Grid. Click-to-move.
- **AI Thinking Heatmap**: After clicking "Get AI Move", squares highlight from red (low visit count) to green (high visit count) based on MCTS visit distribution before the AI commits.
- **Modes**:
  - *Play as Black* (AI is White)
  - *Play as White* (AI is Black)
  - *Spectate AI vs AI* (auto-play with delay, shows both sides' thinking)
- **Training Dashboard** (bottom panel):
  - Live line charts (Chart.js) for Policy Loss, Value Loss, ELO.
  - Current status: `Running` / `Paused` / `Idle`.
  - Buttons: **Start Training**, **Stop Training**, **Resume Training**.
  - Checkpoint selector dropdown to "Load model for play".

### Training Control (`web/training_manager.py`)
- Since the Python GIL would block FastAPI if training ran in the main thread, the backend spawns a dedicated `multiprocessing.Process` for the full self-play + training loop.
- Communication:
  - Main process → Training process: `multiprocessing.Queue` for commands (`START`, `STOP`, `RESUME`, `SHUTDOWN`).
  - Training process → Main process: `multiprocessing.Queue` for metrics (sent out via WebSocket).
- On **Stop**, the training process saves an emergency checkpoint and enters a paused state, ready to resume.

## 5. Data & Checkpointing Strategy (Drive F)
- **Path**: `F:\Workspace\flip-ai\data\`
- **Frequency**:
  - **Light save** (every 100 steps): `meta.json` + training stats only.
  - **Heavy save** (every 1000 steps): Full model + optimizer + replay buffer (~20GB). With NVMe/SSD on F:\, this takes ~30–60s.
  - **On Stop**: Immediate heavy save.
- **Resume Flow**: `resume_train.py` reads `data/checkpoints/latest/meta.json`, loads the associated `.pt` files, and continues.

## 6. Execution Roadmap

| Phase | Task | Deliverable |
|---|---|---|
| **1** | Conda env, project scaffold, Othello engine | Playable Python Othello with strict rules |
| **2** | ResNet policy-value network + MCTS (CPU/GPU batched) | Script that runs MCTS on a random board and prints best move |
| **3** | Self-play loop + replay buffer + trainer | `train.py` runs and prints decreasing loss |
| **4** | Checkpointing, evaluation arena, resume logic | `resume_train.py` works; ELO tracked |
| **5** | FastAPI backend + game API | Postman/curl can play a full game against AI |
| **6** | Frontend board + human play + AI move | Web page playable in browser |
| **7** | AI Thinking heatmap + Spectate mode | Visual MCTS overlay on board |
| **8** | Training dashboard + WebSocket live metrics | Charts update in real-time |
| **9** | Training control buttons (Start/Stop/Resume) + Checkpoint loader | Full control from web UI |
| **10** | Integration testing & Windows multiprocessing hardening | Stable long-run training |

## 7. Anticipated Issues & Mitigations
| Issue | Mitigation |
|---|---|
| **Windows `multiprocessing` spawn overhead** | Use `if __name__ == '__main__':` everywhere; limit workers to 48; use `torch.multiprocessing` where possible. |
| **Large replay buffer serialization time** | Save buffer asynchronously in a background thread; use `torch.save(..., pickle_protocol=5)` for speed. |
| **WebSocket disconnect during long training** | Frontend auto-reconnects; backend queues last 60 seconds of metrics for burst catch-up. |
| **CUDA OOM from overly large batches** | Start at batch size 4096; auto-detect and halve if `RuntimeError: CUDA out of memory`. |
