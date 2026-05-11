/**
 * Gomoku AI Playground - Frontend Logic
 */

const API_BASE = '/api';
const BOARD_SIZE = 15;
const BLACK = 1;
const WHITE = 2;

class GomokuApp {
    constructor() {
        this.sessionId = null;
        this.mode = 'human-black';
        this.isGameOver = false;
        this.humanColor = BLACK;
        this.isThinking = false;
        this.lastAiMove = null;          // Track last AI move for preferences
        this.isSuggestingMove = false;   // True when user is suggesting a better move
        this.moveCount = 0;

        this.initElements();
        this.initEventListeners();
        this.createBoard();
        this.refreshCheckpointList();
        this.refreshPreferenceCount();
        this.newGame();
    }

    initElements() {
        this.boardEl = document.getElementById('board');
        this.newGameBtn = document.getElementById('newGameBtn');
        this.aiMoveBtn = document.getElementById('aiMoveBtn');
        this.gameModeSelect = document.getElementById('gameMode');
        this.blackScoreEl = document.getElementById('blackScore');
        this.whiteScoreEl = document.getElementById('whiteScore');
        this.turnIndicatorEl = document.getElementById('turnIndicator');
        this.gameOverEl = document.getElementById('gameOver');
        this.winnerTextEl = document.getElementById('winnerText');
        this.thinkingPanelEl = document.getElementById('thinkingPanel');
        this.thinkingMapEl = document.getElementById('thinkingMap');
        this.thinkingValueEl = document.getElementById('thinkingValue');
        this.checkpointSelect = document.getElementById('checkpointSelect');
        this.loadCheckpointBtn = document.getElementById('loadCheckpointBtn');

        // Preference UI elements
        this.preferencePanelEl = document.getElementById('preferencePanel');
        this.preferenceTextEl = document.getElementById('preferenceText');
        this.thumbsUpBtn = document.getElementById('thumbsUpBtn');
        this.suggestMoveBtn = document.getElementById('suggestMoveBtn');
        this.badMoveBtn = document.getElementById('badMoveBtn');
        this.preferenceStatusEl = document.getElementById('preferenceStatus');
        this.prefCountEl = document.getElementById('prefCount');
    }

    initEventListeners() {
        this.newGameBtn.addEventListener('click', () => this.newGame());
        this.aiMoveBtn.addEventListener('click', () => this.requestAIMove());
        this.loadCheckpointBtn.addEventListener('click', () => this.loadCheckpoint());
        this.gameModeSelect.addEventListener('change', (e) => {
            this.mode = e.target.value;
            this.newGame();
        });

        // Preference buttons
        this.thumbsUpBtn.addEventListener('click', () => this.submitPreference(null));
        this.suggestMoveBtn.addEventListener('click', () => this.enterSuggestionMode());
        this.badMoveBtn.addEventListener('click', () => this.submitBadMove());

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

            const key = e.key.toLowerCase();

            // Escape = exit suggestion mode
            if (key === 'escape') {
                e.preventDefault();
                if (this.isSuggestingMove) {
                    this.exitSuggestionMode();
                }
                return;
            }

            // N = New Game (always works)
            if (key === 'n') {
                e.preventDefault();
                this.newGame();
                return;
            }

            // Preference shortcuts only when panel is visible and not in suggestion mode
            if (this.preferencePanelEl.classList.contains('hidden')) return;
            if (this.isSuggestingMove) return;

            if (key === 'g') {
                e.preventDefault();
                this.submitPreference(null);
            } else if (key === 's') {
                e.preventDefault();
                this.enterSuggestionMode();
            } else if (key === 'b') {
                e.preventDefault();
                this.submitBadMove();
            }
        });
    }

    createBoard() {
        this.boardEl.innerHTML = '';
        for (let row = 0; row < BOARD_SIZE; row++) {
            for (let col = 0; col < BOARD_SIZE; col++) {
                const cell = document.createElement('div');
                cell.className = 'cell';
                cell.dataset.row = row;
                cell.dataset.col = col;
                cell.addEventListener('click', () => this.handleCellClick(row, col));
                this.boardEl.appendChild(cell);
            }
        }
    }

    async newGame() {
        this.mode = this.gameModeSelect.value;
        this.humanColor = this.mode === 'human-white' ? WHITE : BLACK;
        this.isGameOver = false;
        this.isThinking = false;
        this.lastAiMove = null;
        this.isSuggestingMove = false;
        this.moveCount = 0;
        this.hideThinking();
        this.hideGameOver();
        this.hidePreferencePanel();
        this.boardEl.classList.remove('suggestion-mode');

        try {
            const response = await fetch(`${API_BASE}/game/new`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ mode: this.mode })
            });
            const data = await response.json();
            this.sessionId = data.session_id;
            this.updateUI(data);
            this.updateControls(data);

            // If AI vs AI or human is white, trigger AI move
            if (this.mode === 'ai-vs-ai' || this.mode === 'human-white') {
                setTimeout(() => this.requestAIMove(), 500);
            }
        } catch (err) {
            console.error('Failed to start new game:', err);
        }
    }

    async refreshCheckpointList() {
        try {
            const response = await fetch(`${API_BASE}/checkpoints`);
            const data = await response.json();
            const checkpoints = data.checkpoints || [];

            this.checkpointSelect.innerHTML = '<option value="">Default (untrained)</option>';
            checkpoints.forEach(ckpt => {
                const option = document.createElement('option');
                option.value = ckpt.path;
                option.textContent = ckpt.name;
                this.checkpointSelect.appendChild(option);
            });
        } catch (err) {
            console.error('Failed to load checkpoints:', err);
        }
    }

    async loadCheckpoint() {
        const path = this.checkpointSelect.value;
        if (!path) {
            console.log('Using default model');
            return;
        }

        try {
            const response = await fetch(`${API_BASE}/ai/load_checkpoint`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ checkpoint_path: path })
            });
            const data = await response.json();
            if (data.status === 'success') {
                alert('Checkpoint loaded: ' + this.checkpointSelect.options[this.checkpointSelect.selectedIndex].text);
            } else {
                alert('Failed to load checkpoint');
            }
        } catch (err) {
            console.error('Failed to load checkpoint:', err);
            alert('Error loading checkpoint');
        }
    }

    async handleCellClick(row, col) {
        if (this.isGameOver || this.isThinking) return;

        const action = row * BOARD_SIZE + col;

        // Suggestion mode: submit preference and STAY in suggestion mode
        // so user can suggest multiple better moves
        if (this.isSuggestingMove) {
            if (this.lastAiMove !== null && action !== this.lastAiMove) {
                await this.submitPreference(action);
                // Stay in suggestion mode — user can click another cell
            }
            return;
        }

        const currentPlayer = this.getCurrentPlayerFromUI();

        // Only allow moves on human's turn
        if (this.mode !== 'ai-vs-ai' && currentPlayer !== this.humanColor) {
            return;
        }

        await this.makeMove(action);
    }

    async makeMove(action) {
        try {
            const response = await fetch(`${API_BASE}/game/${this.sessionId}/move`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ action })
            });

            if (!response.ok) {
                console.warn('Invalid move');
                return;
            }

            const data = await response.json();
            this.updateUI(data);
            this.updateControls(data);

            if (data.is_game_over) {
                this.showGameOver(data);
            } else if (this.mode !== 'ai-vs-ai' && data.current_player !== this.humanColor) {
                // Trigger AI move after human move
                setTimeout(() => this.requestAIMove(), 300);
            } else if (this.mode === 'ai-vs-ai') {
                // Auto-play next AI move
                setTimeout(() => this.requestAIMove(), 500);
            }
        } catch (err) {
            console.error('Move failed:', err);
        }
    }

    async requestAIMove() {
        if (this.isGameOver || this.isThinking || !this.sessionId) return;

        this.isThinking = true;
        this.aiMoveBtn.disabled = true;

        try {
            const response = await fetch(`${API_BASE}/game/${this.sessionId}/ai_move`, {
                method: 'POST'
            });

            if (!response.ok) {
                console.warn('AI move failed');
                return;
            }

            const data = await response.json();
            this.updateUI(data);
            this.updateControls(data);

            if (data.ai_move && data.ai_move.thinking) {
                this.showThinking(data.ai_move.thinking);
            }

            // Track AI move for preference collection
            if (data.ai_move) {
                this.lastAiMove = data.ai_move.action;
                this.moveCount++;
                // Show preference panel for human-vs-AI modes
                if (this.mode !== 'ai-vs-ai' && !data.is_game_over) {
                    this.showPreferencePanel();
                }
            }

            if (data.is_game_over) {
                this.showGameOver(data);
            } else if (this.mode === 'ai-vs-ai') {
                setTimeout(() => this.requestAIMove(), 500);
            }
        } catch (err) {
            console.error('AI move failed:', err);
        } finally {
            this.isThinking = false;
            this.updateControls(this.lastState);
        }
    }

    // Preference collection methods
    showPreferencePanel() {
        if (!this.preferencePanelEl) return;
        this.preferencePanelEl.classList.remove('hidden');
        this.preferenceTextEl.textContent = 'Was this a good move?';
        this.preferenceStatusEl.textContent = '';
        this.preferenceStatusEl.classList.add('hidden');
        this.thumbsUpBtn.disabled = false;
        this.suggestMoveBtn.disabled = false;
        this.badMoveBtn.disabled = false;
    }

    hidePreferencePanel() {
        if (!this.preferencePanelEl) return;
        this.preferencePanelEl.classList.add('hidden');
    }

    enterSuggestionMode() {
        this.isSuggestingMove = true;
        this.preferenceTextEl.textContent = 'Click on the board where you would have played instead. You can suggest multiple moves. Press Escape when done.';
        this.boardEl.classList.add('suggestion-mode');
        this.thumbsUpBtn.disabled = true;
        this.suggestMoveBtn.disabled = true;
        this.badMoveBtn.disabled = true;
    }

    exitSuggestionMode() {
        this.isSuggestingMove = false;
        this.boardEl.classList.remove('suggestion-mode');
        this.hidePreferencePanel();
    }

    async submitPreference(preferredMove) {
        if (this.lastAiMove === null || !this.sessionId) return;

        const isThumbsUp = preferredMove === null;
        const prefMove = isThumbsUp ? this.lastAiMove : preferredMove;
        const prefType = isThumbsUp ? 'good' : 'suggest';

        try {
            const response = await fetch(`${API_BASE}/game/${this.sessionId}/preference`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    ai_move: this.lastAiMove,
                    preferred_move: prefMove,
                    move_number: this.moveCount,
                    type: prefType,
                })
            });

            if (response.ok) {
                this.preferenceStatusEl.classList.remove('hidden');
                this.refreshPreferenceCount();

                if (this.isSuggestingMove) {
                    // Stay in suggestion mode — show message encouraging more suggestions
                    this.preferenceStatusEl.textContent = 'Suggestion recorded! Click another cell to suggest more, or press Escape when done.';
                } else {
                    this.thumbsUpBtn.disabled = true;
                    this.suggestMoveBtn.disabled = true;
                    this.badMoveBtn.disabled = true;
                    this.preferenceStatusEl.textContent = isThumbsUp
                        ? 'Thanks! Good move recorded.'
                        : 'Thanks! Better move suggestion recorded.';
                    // Hide panel after a short delay
                    setTimeout(() => this.hidePreferencePanel(), 1500);
                }
            } else {
                console.warn('Failed to submit preference');
            }
        } catch (err) {
            console.error('Preference submission failed:', err);
        }
    }

    async submitBadMove() {
        if (this.lastAiMove === null || !this.sessionId) return;

        try {
            const response = await fetch(`${API_BASE}/game/${this.sessionId}/preference`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    ai_move: this.lastAiMove,
                    preferred_move: null,
                    move_number: this.moveCount,
                    type: 'bad',
                })
            });

            if (response.ok) {
                this.preferenceStatusEl.textContent = 'Thanks! Bad move recorded.';
                this.preferenceStatusEl.classList.remove('hidden');
                this.thumbsUpBtn.disabled = true;
                this.suggestMoveBtn.disabled = true;
                this.badMoveBtn.disabled = true;
                this.refreshPreferenceCount();

                setTimeout(() => this.hidePreferencePanel(), 1500);
            } else {
                console.warn('Failed to submit bad move');
            }
        } catch (err) {
            console.error('Bad move submission failed:', err);
        }
    }

    async refreshPreferenceCount() {
        try {
            const response = await fetch(`${API_BASE}/preferences/count`);
            const data = await response.json();
            this.prefCountEl.textContent = data.total_preferences || 0;
        } catch (err) {
            console.error('Failed to load preference count:', err);
        }
    }

    updateUI(state) {
        this.lastState = state;
        const board = state.board;
        const legalMoves = state.legal_moves || [];

        // Update board
        const cells = this.boardEl.querySelectorAll('.cell');
        cells.forEach((cell, index) => {
            const row = Math.floor(index / BOARD_SIZE);
            const col = index % BOARD_SIZE;
            const value = board[row][col];
            const action = row * BOARD_SIZE + col;

            cell.innerHTML = '';
            cell.classList.remove('legal-move');

            if (value === BLACK) {
                const piece = document.createElement('div');
                piece.className = 'piece black';
                cell.appendChild(piece);
            } else if (value === WHITE) {
                const piece = document.createElement('div');
                piece.className = 'piece white';
                cell.appendChild(piece);
            }

            if (legalMoves.includes(action)) {
                cell.classList.add('legal-move');
            }
        });

        // Update scores (piece counts)
        this.blackScoreEl.textContent = state.black_count;
        this.whiteScoreEl.textContent = state.white_count;

        // Update turn
        const turnText = state.current_player === BLACK ? 'Black' : 'White';
        this.turnIndicatorEl.textContent = turnText;
    }

    updateControls(state) {
        if (!state) return;
        this.aiMoveBtn.disabled = this.isThinking || state.is_game_over || this.mode === 'ai-vs-ai';
    }

    getCurrentPlayerFromUI() {
        return this.turnIndicatorEl.textContent === 'Black' ? BLACK : WHITE;
    }

    showGameOver(state) {
        this.isGameOver = true;
        this.gameOverEl.classList.remove('hidden');

        let text = '';
        if (state.winner === 0) {
            text = 'Game Over - Draw!';
        } else if (state.winner === BLACK) {
            text = 'Game Over - Black Wins!';
        } else {
            text = 'Game Over - White Wins!';
        }
        this.winnerTextEl.textContent = text;
    }

    hideGameOver() {
        this.gameOverEl.classList.add('hidden');
    }

    showThinking(thinking) {
        this.thinkingPanelEl.classList.remove('hidden');
        this.thinkingValueEl.textContent = thinking.value.toFixed(3);

        // Show visit count heatmap (15x15)
        const visits = thinking.visit_counts;
        const maxVisit = Math.max(...visits);
        this.thinkingMapEl.innerHTML = '';

        for (let i = 0; i < BOARD_SIZE * BOARD_SIZE; i++) {
            const cell = document.createElement('div');
            cell.className = 'thinking-cell';
            const visit = visits[i];
            const intensity = maxVisit > 0 ? visit / maxVisit : 0;
            const red = Math.round(255 * intensity);
            const green = Math.round(255 * (1 - intensity));
            cell.style.backgroundColor = `rgba(${red}, ${green}, 0, 0.6)`;
            cell.textContent = visit > 0 ? visit.toFixed(0) : '';
            this.thinkingMapEl.appendChild(cell);
        }
    }

    hideThinking() {
        this.thinkingPanelEl.classList.add('hidden');
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.app = new GomokuApp();
});
