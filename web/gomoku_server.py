"""
FastAPI backend for Gomoku AI playground.

Provides:
  - Game API: create, move, AI move, state
  - Model loading from Gomoku checkpoints
"""

import os
import sys
# Add project root to path for imports when running from web/ directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uuid
import datetime
import json
import numpy as np
import torch
from typing import Optional, Dict
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from game.gomoku import GomokuGame, BLACK, WHITE
from ai.model import OthelloNet, create_model
from ai.mcts import MCTS

app = FastAPI(title="Gomoku AI Playground")

# Gomoku configuration
BOARD_SIZE = 15
ACTION_SIZE = BOARD_SIZE * BOARD_SIZE  # 225

# Global AI model (lazy-loaded)
_model: Optional[OthelloNet] = None
_mcts: Optional[MCTS] = None
_device: str = 'cpu'

# Active game sessions
_sessions: Dict[str, 'GameSession'] = {}

# Preference storage directory
PREFERENCE_DIR = 'data/human_preferences'


def get_model() -> OthelloNet:
    """Get or initialize the global AI model with CPU fallback."""
    global _model, _mcts, _device
    if _model is None:
        _device = 'cpu'
        if torch.cuda.is_available():
            try:
                test_tensor = torch.zeros(1, 3, BOARD_SIZE, BOARD_SIZE).cuda()
                test_model = OthelloNet(
                    num_blocks=1, num_channels=8,
                    board_size=BOARD_SIZE, action_size=ACTION_SIZE
                ).cuda()
                with torch.no_grad():
                    _ = test_model(test_tensor)
                _device = 'cuda'
                del test_tensor, test_model
            except RuntimeError:
                print("CUDA available but incompatible with this GPU, using CPU")
                _device = 'cpu'

        _model = create_model(
            num_blocks=10, num_channels=128,
            board_size=BOARD_SIZE, action_size=ACTION_SIZE,
            device=_device
        )
        _mcts = MCTS(_model, num_simulations=800)
        print(f"Gomoku model loaded on {_device}")
    return _model


def get_mcts() -> MCTS:
    """Get or initialize MCTS."""
    if _mcts is None:
        get_model()
    return _mcts


class GameSession:
    """Manages a single Gomoku game instance."""

    def __init__(self, session_id: str, mode: str = 'human-black'):
        self.session_id = session_id
        self.mode = mode  # human-black, human-white, ai-vs-ai
        self.game = GomokuGame()
        self.move_history = []
        self.thinking_data = None

    def get_state(self) -> dict:
        """Return current game state as JSON-serializable dict."""
        board = self.game.get_board_state().tolist()
        legal_moves = self.game.get_legal_moves()
        winner, black_count, white_count = self.game.get_winner()

        return {
            'session_id': self.session_id,
            'mode': self.mode,
            'board': board,
            'board_size': BOARD_SIZE,
            'current_player': int(self.game.current_player),
            'legal_moves': legal_moves,
            'is_game_over': self.game.is_game_over(),
            'winner': int(winner) if winner is not None else None,
            'black_count': int(black_count),
            'white_count': int(white_count),
            'move_count': len(self.move_history),
        }

    def make_move(self, action: int) -> bool:
        """Make a move. Returns success."""
        success, msg = self.game.make_move(action)
        if success:
            self.move_history.append({
                'action': int(action),
                'player': int(self.game.get_opponent(self.game.current_player)),
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
                'player': int(self.game.current_player),
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
    mode: str = 'human-black'


class MoveRequest(BaseModel):
    action: int


class LoadCheckpointRequest(BaseModel):
    checkpoint_path: str


class PreferenceRequest(BaseModel):
    ai_move: int
    preferred_move: Optional[int] = None
    move_number: int = 0
    type: str = 'good'  # 'good', 'suggest', 'bad'


def _save_preference(session_id: str, state: np.ndarray, ai_move: int, preferred_move: Optional[int], move_number: int = 0, pref_type: str = 'good'):
    """Save a single preference pair to disk."""
    os.makedirs(PREFERENCE_DIR, exist_ok=True)
    today = datetime.datetime.now().strftime('%Y%m%d')
    daily_dir = os.path.join(PREFERENCE_DIR, today)
    os.makedirs(daily_dir, exist_ok=True)

    pref_file = os.path.join(daily_dir, f'session_{session_id}.json')

    entry = {
        'state': state.tolist(),
        'preferred_action': preferred_move,
        'rejected_action': ai_move,
        'move_number': move_number,
        'type': pref_type,
        'timestamp': datetime.datetime.now().isoformat(),
    }

    # Append to existing file or create new
    if os.path.exists(pref_file):
        with open(pref_file, 'r') as f:
            data = json.load(f)
    else:
        data = {'session_id': session_id, 'preferences': []}

    data['preferences'].append(entry)

    with open(pref_file, 'w') as f:
        json.dump(data, f, indent=2)


# API Endpoints
@app.post("/api/game/new")
def new_game(request: NewGameRequest):
    """Start a new game."""
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


@app.post("/api/game/{session_id}/preference")
def submit_preference(session_id: str, request: PreferenceRequest):
    """Submit a human preference pair for an AI move."""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = _sessions[session_id]
    state = session.game.get_state_planes()

    _save_preference(
        session_id=session_id,
        state=state,
        ai_move=request.ai_move,
        preferred_move=request.preferred_move,
        move_number=request.move_number,
        pref_type=request.type,
    )

    return {'status': 'success', 'message': 'Preference saved'}


@app.get("/api/preferences/count")
def get_preference_count():
    """Get total number of saved preference pairs."""
    count = 0
    if os.path.exists(PREFERENCE_DIR):
        for root, dirs, files in os.walk(PREFERENCE_DIR):
            for f in files:
                if f.endswith('.json'):
                    try:
                        with open(os.path.join(root, f), 'r') as fp:
                            data = json.load(fp)
                            count += len(data.get('preferences', []))
                    except:
                        pass
    return {'total_preferences': count}


@app.post("/api/ai/load_checkpoint")
def load_checkpoint(request: LoadCheckpointRequest):
    """Load a model checkpoint for play."""
    global _model, _mcts
    checkpoint_path = request.checkpoint_path

    # If path is a directory, look for model.pt inside it
    if os.path.isdir(checkpoint_path):
        model_path = os.path.join(checkpoint_path, 'model.pt')
        if os.path.exists(model_path):
            checkpoint_path = model_path
        else:
            raise HTTPException(
                status_code=404,
                detail="Checkpoint directory does not contain model.pt"
            )
    elif not os.path.exists(checkpoint_path):
        raise HTTPException(status_code=404, detail="Checkpoint not found")

    try:
        checkpoint = torch.load(
            checkpoint_path, map_location=_device, weights_only=False
        )
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
        import traceback
        print(f"[Checkpoint Load Error] {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/checkpoints")
def list_checkpoints():
    """List available Gomoku checkpoints from all relevant directories."""
    checkpoint_dirs = [
        ('data/gomoku_checkpoints', 'Gomoku'),
        ('data/human_rl', 'Human RL'),
        ('data/checkpoints', 'Othello'),
    ]
    checkpoints = []

    for checkpoint_dir, source in checkpoint_dirs:
        if not os.path.exists(checkpoint_dir):
            continue
        for name in sorted(os.listdir(checkpoint_dir), reverse=True):
            path = os.path.join(checkpoint_dir, name)
            if os.path.isdir(path):
                if os.path.exists(os.path.join(path, 'model.pt')):
                    checkpoints.append({'name': f'[{source}] {name}', 'path': path})
            elif name.endswith('.pt') and os.path.isfile(path):
                # Direct .pt files (e.g., best_model.pt)
                checkpoints.append({'name': f'[{source}] {name}', 'path': path})

    return {'checkpoints': checkpoints}


# Serve Gomoku page at root
@app.get("/")
def serve_gomoku():
    return FileResponse("web/static/gomoku.html")

# Static files (shared CSS/JS)
app.mount("/css", StaticFiles(directory="web/static/css"), name="css")
app.mount("/js", StaticFiles(directory="web/static/js"), name="js")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8080)
