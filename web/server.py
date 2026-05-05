"""
FastAPI backend for Othello AI playground.

Provides:
  - Game API: create, move, AI move, state
  - Training control: start, stop, resume (via TrainingManager)
  - WebSocket: live training metrics stream
"""

import os
import json
import numpy as np
import torch
from typing import Optional, Dict
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from game.othello import OthelloGame, PASS_ACTION, BLACK, WHITE
from ai.model import OthelloNet, create_model
from ai.mcts import MCTS
from web.training_manager import training_manager

app = FastAPI(title="Othello AI Playground")

# Global AI model (lazy-loaded)
_model: Optional[OthelloNet] = None
_mcts: Optional[MCTS] = None
_device: str = 'cpu'

# Active game sessions
_sessions: Dict[str, 'GameSession'] = {}


def get_model() -> OthelloNet:
    """Get or initialize the global AI model with CPU fallback."""
    global _model, _mcts, _device
    if _model is None:
        _device = 'cpu'
        if torch.cuda.is_available():
            # Test if CUDA actually works (sm_120 compatibility check)
            try:
                test_tensor = torch.zeros(1, 3, 8, 8).cuda()
                test_model = OthelloNet(num_blocks=1, num_channels=8).cuda()
                with torch.no_grad():
                    _ = test_model(test_tensor)
                _device = 'cuda'
                del test_tensor, test_model
            except RuntimeError:
                print("CUDA available but incompatible with this GPU, using CPU")
                _device = 'cpu'

        _model = create_model(num_blocks=20, num_channels=256, device=_device)
        _mcts = MCTS(_model, num_simulations=800)
        print(f"Model loaded on {_device}")
    return _model


def get_mcts() -> MCTS:
    """Get or initialize MCTS."""
    if _mcts is None:
        get_model()
    return _mcts


class GameSession:
    """Manages a single game instance."""

    def __init__(self, session_id: str, mode: str = 'human-black'):
        self.session_id = session_id
        self.mode = mode  # human-black, human-white, ai-vs-ai
        self.game = OthelloGame()
        self.move_history = []
        self.thinking_data = None  # Last MCTS visit counts

    def get_state(self) -> dict:
        """Return current game state as JSON-serializable dict."""
        board = self.game.get_board_state().tolist()
        legal_moves = self.game.get_legal_moves()
        winner, black_count, white_count = self.game.get_winner()

        return {
            'session_id': self.session_id,
            'mode': self.mode,
            'board': board,
            'current_player': int(self.game.current_player),
            'legal_moves': legal_moves,
            'is_game_over': self.game.is_game_over(),
            'winner': int(winner) if winner is not None else None,
            'black_count': black_count,
            'white_count': white_count,
            'move_count': len(self.move_history),
        }

    def make_move(self, action: int) -> bool:
        """Make a move. Returns success."""
        success, msg = self.game.make_move(action)
        if success:
            self.move_history.append({
                'action': int(action),
                'player': int(self.game.get_opponent(self.game.current_player)),  # Player who just moved
                'message': msg,
            })
        return success

    def get_ai_move(self, temperature: float = 0.0) -> dict:
        """Get AI move with thinking data."""
        mcts = get_mcts()
        action_probs, value = mcts.search(self.game, temperature=temperature)
        action = int(np.argmax(action_probs))

        self.thinking_data = {
            'visit_counts': action_probs.tolist(),
            'value': float(value[0]),
        }

        success, msg = self.game.make_move(action)
        if success:
            self.move_history.append({
                'action': action,
                'player': int(self.game.current_player),  # Already switched
                'message': msg,
                'thinking': self.thinking_data,
            })

        return {
            'action': action,
            'success': success,
            'message': msg,
            'thinking': self.thinking_data,
        }


# Pydantic models for API
class NewGameRequest(BaseModel):
    mode: str = 'human-black'  # human-black, human-white, ai-vs-ai


class MoveRequest(BaseModel):
    action: int


class LoadCheckpointRequest(BaseModel):
    checkpoint_path: str


# API Endpoints
@app.post("/api/game/new")
def new_game(request: NewGameRequest):
    """Start a new game."""
    import uuid
    session_id = str(uuid.uuid4())[:8]
    session = GameSession(session_id, mode=request.mode)
    _sessions[session_id] = session
    return session.get_state()


@app.post("/api/game/{session_id}/move")
def make_move(session_id: str, request: MoveRequest):
    """Make a human move."""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = _sessions[session_id]
    if session.game.is_game_over():
        raise HTTPException(status_code=400, detail="Game is already over")

    if not session.game.is_valid_move(request.action):
        raise HTTPException(status_code=400, detail="Invalid move")

    success = session.make_move(request.action)
    if not success:
        raise HTTPException(status_code=400, detail="Move failed")

    return session.get_state()


@app.post("/api/game/{session_id}/ai_move")
def ai_move(session_id: str):
    """Request AI to make a move."""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = _sessions[session_id]
    if session.game.is_game_over():
        raise HTTPException(status_code=400, detail="Game is already over")

    result = session.get_ai_move(temperature=0.0)
    if not result['success']:
        raise HTTPException(status_code=400, detail="AI move failed")

    return {**session.get_state(), 'ai_move': result}


@app.get("/api/game/{session_id}/state")
def get_state(session_id: str):
    """Get current game state."""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return _sessions[session_id].get_state()


@app.post("/api/game/{session_id}/thinking")
def get_thinking(session_id: str):
    """Get last AI thinking data."""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = _sessions[session_id]
    if session.thinking_data is None:
        raise HTTPException(status_code=400, detail="No thinking data available")

    return session.thinking_data


@app.post("/api/ai/load_checkpoint")
def load_checkpoint(request: LoadCheckpointRequest):
    """Load a model checkpoint for play."""
    global _model, _mcts
    if not os.path.exists(request.checkpoint_path):
        raise HTTPException(status_code=404, detail="Checkpoint not found")

    try:
        checkpoint = torch.load(request.checkpoint_path, map_location=_device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        if _model is None:
            get_model()

        _model.load_state_dict(state_dict)
        _mcts = MCTS(_model, num_simulations=800)
        return {'status': 'success', 'message': 'Checkpoint loaded'}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/training/start")
def training_start():
    """Start training."""
    return training_manager.start(total_steps=100_000)


@app.post("/api/training/stop")
def training_stop():
    """Stop/pause training."""
    return training_manager.stop()


@app.post("/api/training/resume")
def training_resume():
    """Resume training."""
    return training_manager.resume()


@app.get("/api/training/status")
def training_status():
    """Get training status."""
    return training_manager.get_status()


# Training WebSocket
@app.websocket("/ws/training")
async def training_websocket(websocket: WebSocket):
    """WebSocket for live training metrics."""
    await websocket.accept()
    try:
        import asyncio
        while True:
            metrics = training_manager.get_metrics()
            if metrics:
                await websocket.send_json(metrics)
            await asyncio.sleep(1.0)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WebSocket error: {e}")


@app.get("/api/checkpoints")
def list_checkpoints():
    """List available checkpoints."""
    checkpoint_dir = 'data/checkpoints'
    checkpoints = []
    if os.path.exists(checkpoint_dir):
        for name in os.listdir(checkpoint_dir):
            path = os.path.join(checkpoint_dir, name)
            if os.path.isdir(path) and os.path.exists(os.path.join(path, 'model.pt')):
                checkpoints.append({
                    'name': name,
                    'path': path,
                })
    return {'checkpoints': checkpoints}


# Static files
app.mount("/", StaticFiles(directory="web/static", html=True), name="static")
