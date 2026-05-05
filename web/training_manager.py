"""
Training Manager for Othello AI.

Wraps the training loop in a subprocess and provides
Start / Stop / Resume control via queues.
"""

import os
import time
import json
import multiprocessing as mp
from typing import Optional, Dict, Any


class TrainingManager:
    """
    Manages training process lifecycle.

    Commands: START, STOP, RESUME, SHUTDOWN
    Metrics: streamed back via queue
    """

    def __init__(
        self,
        checkpoint_dir: str = 'data/checkpoints',
        log_dir: str = 'data/logs',
    ):
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        self.command_queue: Optional[mp.Queue] = None
        self.metrics_queue: Optional[mp.Queue] = None
        self.process: Optional[mp.Process] = None
        self.status = 'idle'  # idle, running, paused

    def start(self, total_steps: int = 100_000):
        """Start training in a subprocess."""
        if self.process is not None and self.process.is_alive():
            return {'status': 'already_running'}

        self.command_queue = mp.Queue()
        self.metrics_queue = mp.Queue()
        self.status = 'running'

        self.process = mp.Process(
            target=self._training_loop,
            args=(self.command_queue, self.metrics_queue, total_steps, self.checkpoint_dir, self.log_dir),
        )
        self.process.start()
        return {'status': 'started'}

    def stop(self):
        """Pause training."""
        if self.process is None or not self.process.is_alive():
            return {'status': 'not_running'}

        self.command_queue.put('STOP')
        self.status = 'paused'
        return {'status': 'stopping'}

    def resume(self):
        """Resume training if paused."""
        if self.status == 'paused' and self.process is not None and self.process.is_alive():
            self.command_queue.put('RESUME')
            self.status = 'running'
            return {'status': 'resumed'}
        else:
            # If process died, restart
            return self.start()

    def shutdown(self):
        """Terminate training process."""
        if self.process is not None and self.process.is_alive():
            self.command_queue.put('SHUTDOWN')
            self.process.join(timeout=10)
            if self.process.is_alive():
                self.process.terminate()
        self.status = 'idle'
        self.process = None
        return {'status': 'shutdown'}

    def get_metrics(self) -> Optional[Dict[str, Any]]:
        """Non-blocking read of latest metrics."""
        if self.metrics_queue is None:
            return None
        metrics = None
        while not self.metrics_queue.empty():
            try:
                metrics = self.metrics_queue.get_nowait()
            except:
                break
        return metrics

    def get_status(self) -> Dict[str, Any]:
        """Get current training status."""
        metrics = self.get_metrics()
        return {
            'status': self.status,
            'process_alive': self.process.is_alive() if self.process else False,
            'latest_metrics': metrics,
        }

    @staticmethod
    def _training_loop(
        command_queue: mp.Queue,
        metrics_queue: mp.Queue,
        total_steps: int,
        checkpoint_dir: str,
        log_dir: str,
    ):
        """Training subprocess entry point."""
        import torch
        from ai.model import create_model
        from ai.mcts import MCTS
        from ai.trainer import Trainer, ReplayBuffer

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = create_model(num_blocks=20, num_channels=256, device=device)
        mcts = MCTS(model, num_simulations=400)
        buffer = ReplayBuffer(capacity=500_000)
        trainer = Trainer(
            model=model,
            mcts=mcts,
            replay_buffer=buffer,
            device=device,
            checkpoint_dir=checkpoint_dir,
            log_dir=log_dir,
        )

        paused = False

        while trainer.training_step < total_steps:
            # Check commands
            while not command_queue.empty():
                try:
                    cmd = command_queue.get_nowait()
                    if cmd == 'STOP':
                        paused = True
                        trainer.save_checkpoint(f"paused_{trainer.training_step}")
                        metrics_queue.put({'status': 'paused', 'step': trainer.training_step})
                    elif cmd == 'RESUME':
                        paused = False
                        metrics_queue.put({'status': 'resumed', 'step': trainer.training_step})
                    elif cmd == 'SHUTDOWN':
                        trainer.save_checkpoint(f"shutdown_{trainer.training_step}")
                        return
                except:
                    break

            if paused:
                time.sleep(0.5)
                continue

            # Generate self-play data and train
            positions = trainer.generate_self_play_data(num_games=10)
            metrics = trainer.train_step()
            metrics['buffer_size'] = len(buffer)
            metrics['positions_added'] = positions
            metrics['games_played'] = trainer.games_played
            metrics['status'] = 'running'
            metrics['step'] = trainer.training_step

            metrics_queue.put(metrics)

            # Checkpointing
            if trainer.training_step % 1000 == 0:
                trainer.save_checkpoint(f"checkpoint_{trainer.training_step}")

        # Training complete
        trainer.save_checkpoint("final")
        metrics_queue.put({'status': 'completed', 'step': trainer.training_step})


# Global training manager instance
training_manager = TrainingManager()
