/**
 * Othello AI Playground - Frontend Logic
 */

const API_BASE = '/api';

class OthelloApp {
    constructor() {
        this.sessionId = null;
        this.mode = 'human-black';
        this.isGameOver = false;
        this.humanColor = BLACK;
        this.isThinking = false;

        this.initElements();
        this.initEventListeners();
        this.createBoard();
        this.refreshCheckpointList();
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
    }

    initEventListeners() {
        this.newGameBtn.addEventListener('click', () => this.newGame());
        this.aiMoveBtn.addEventListener('click', () => this.requestAIMove());
        this.loadCheckpointBtn.addEventListener('click', () => this.loadCheckpoint());
        this.gameModeSelect.addEventListener('change', (e) => {
            this.mode = e.target.value;
            this.newGame();
        });
    }

    createBoard() {
        this.boardEl.innerHTML = '';
        for (let row = 0; row < 8; row++) {
            for (let col = 0; col < 8; col++) {
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
        this.hideThinking();
        this.hideGameOver();

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

        const action = row * 8 + col;
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

    updateUI(state) {
        this.lastState = state;
        const board = state.board;
        const legalMoves = state.legal_moves || [];

        // Update board
        const cells = this.boardEl.querySelectorAll('.cell');
        cells.forEach((cell, index) => {
            const row = Math.floor(index / 8);
            const col = index % 8;
            const value = board[row][col];
            const action = row * 8 + col;

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

        // Update scores
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
            text = `Game Over - Black Wins! (${state.black_count} - ${state.white_count})`;
        } else {
            text = `Game Over - White Wins! (${state.white_count} - ${state.black_count})`;
        }
        this.winnerTextEl.textContent = text;
    }

    hideGameOver() {
        this.gameOverEl.classList.add('hidden');
    }

    showThinking(thinking) {
        this.thinkingPanelEl.classList.remove('hidden');
        this.thinkingValueEl.textContent = thinking.value.toFixed(3);

        // Show visit count heatmap
        const visits = thinking.visit_counts;
        const maxVisit = Math.max(...visits);
        this.thinkingMapEl.innerHTML = '';

        for (let i = 0; i < 64; i++) {
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

// Constants
const BLACK = 1;
const WHITE = 2;

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.app = new OthelloApp();
    window.trainingDashboard = new TrainingDashboard();
});

class TrainingDashboard {
    constructor() {
        this.ws = null;
        this.reconnectInterval = 3000;
        this.maxDataPoints = 100;
        this.lossData = [];
        this.eloData = [];

        this.initElements();
        this.initCharts();
        this.initEventListeners();
        this.connectWebSocket();
    }

    initElements() {
        this.startBtn = document.getElementById('startTrainingBtn');
        this.stopBtn = document.getElementById('stopTrainingBtn');
        this.resumeBtn = document.getElementById('resumeTrainingBtn');
        this.statusEl = document.getElementById('trainingStatus');
        this.stepEl = document.getElementById('trainingStep');
        this.gamesEl = document.getElementById('trainingGames');
    }

    initCharts() {
        const lossCtx = document.getElementById('lossChart').getContext('2d');
        this.lossChart = new Chart(lossCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Total Loss',
                        data: [],
                        borderColor: '#00d4ff',
                        backgroundColor: 'rgba(0, 212, 255, 0.1)',
                        tension: 0.3,
                        pointRadius: 0,
                    },
                    {
                        label: 'Policy Loss',
                        data: [],
                        borderColor: '#ff6b6b',
                        backgroundColor: 'rgba(255, 107, 107, 0.1)',
                        tension: 0.3,
                        pointRadius: 0,
                    },
                    {
                        label: 'Value Loss',
                        data: [],
                        borderColor: '#51cf66',
                        backgroundColor: 'rgba(81, 207, 102, 0.1)',
                        tension: 0.3,
                        pointRadius: 0,
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { labels: { color: '#ccc' } }
                },
                scales: {
                    x: { ticks: { color: '#888' }, grid: { color: '#333' } },
                    y: { ticks: { color: '#888' }, grid: { color: '#333' } }
                }
            }
        });

        const eloCtx = document.getElementById('eloChart').getContext('2d');
        this.eloChart = new Chart(eloCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Champion ELO',
                        data: [],
                        borderColor: '#ffd43b',
                        backgroundColor: 'rgba(255, 212, 59, 0.1)',
                        tension: 0.3,
                        pointRadius: 2,
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { labels: { color: '#ccc' } }
                },
                scales: {
                    x: { ticks: { color: '#888' }, grid: { color: '#333' } },
                    y: { ticks: { color: '#888' }, grid: { color: '#333' } }
                }
            }
        });
    }

    initEventListeners() {
        this.startBtn.addEventListener('click', () => this.startTraining());
        this.stopBtn.addEventListener('click', () => this.stopTraining());
        this.resumeBtn.addEventListener('click', () => this.resumeTraining());
    }

    connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/training`;

        this.ws = new WebSocket(wsUrl);

        this.ws.onopen = () => {
            console.log('Training WebSocket connected');
        };

        this.ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                this.handleMetrics(data);
            } catch (e) {
                console.warn('Invalid WS message:', event.data);
            }
        };

        this.ws.onclose = () => {
            console.log('Training WebSocket disconnected, reconnecting...');
            setTimeout(() => this.connectWebSocket(), this.reconnectInterval);
        };

        this.ws.onerror = (err) => {
            console.error('WebSocket error:', err);
        };
    }

    async startTraining() {
        try {
            const response = await fetch(`${API_BASE}/training/start`, { method: 'POST' });
            const data = await response.json();
            this.updateStatus('Running');
            console.log('Training started:', data);
        } catch (err) {
            console.error('Failed to start training:', err);
        }
    }

    async stopTraining() {
        try {
            const response = await fetch(`${API_BASE}/training/stop`, { method: 'POST' });
            const data = await response.json();
            this.updateStatus('Paused');
            console.log('Training stopped:', data);
        } catch (err) {
            console.error('Failed to stop training:', err);
        }
    }

    async resumeTraining() {
        try {
            const response = await fetch(`${API_BASE}/training/resume`, { method: 'POST' });
            const data = await response.json();
            this.updateStatus('Running');
            console.log('Training resumed:', data);
        } catch (err) {
            console.error('Failed to resume training:', err);
        }
    }

    updateStatus(status) {
        this.statusEl.textContent = status;
        this.statusEl.style.color = status === 'Running' ? '#51cf66' : '#ff6b6b';
    }

    handleMetrics(data) {
        if (data.status) {
            this.updateStatus(data.status === 'running' ? 'Running' : 'Paused');
        }

        if (data.step !== undefined) {
            this.stepEl.textContent = data.step;
        }

        if (data.games_played !== undefined) {
            this.gamesEl.textContent = data.games_played;
        }

        // Update loss chart
        if (data.total_loss !== undefined && data.total_loss > 0) {
            this.lossData.push({
                step: data.step,
                total: data.total_loss,
                policy: data.policy_loss,
                value: data.value_loss,
            });

            if (this.lossData.length > this.maxDataPoints) {
                this.lossData.shift();
            }

            this.lossChart.data.labels = this.lossData.map(d => d.step);
            this.lossChart.data.datasets[0].data = this.lossData.map(d => d.total);
            this.lossChart.data.datasets[1].data = this.lossData.map(d => d.policy);
            this.lossChart.data.datasets[2].data = this.lossData.map(d => d.value);
            this.lossChart.update('none');
        }

        // Update ELO chart
        if (data.champion_elo !== undefined) {
            this.eloData.push({
                step: data.step,
                elo: data.champion_elo,
            });

            if (this.eloData.length > this.maxDataPoints) {
                this.eloData.shift();
            }

            this.eloChart.data.labels = this.eloData.map(d => d.step);
            this.eloChart.data.datasets[0].data = this.eloData.map(d => d.elo);
            this.eloChart.update('none');
        }
    }
}
