"""
Training Manager for Othello AI.

Wraps the fast async training loop in a subprocess and provides
Start / Stop / Resume control via queues.
"""

import os
import time
import json
import multiprocessing as mp
from typing import Optional, Dict, Any


class TrainingManager:
    """
    Manages fast async training process lifecycle.

    Commands: START, STOP, RESUME, SHUTDOWN
    Metrics: streamed back via queue including worker count, games/sec, inference batch size
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

    def start(
        self,
        total_steps: int = 100_000,
        num_workers: int = 4,
        num_simulations: int = 100,
        mcts_batch_size: int = 16,
        inference_batch_size: int = 64,
        training_batch_size: int = 256,
        buffer_capacity: int = 500_000,
        lr: float = 1e-3,
        num_blocks: int = 20,
        num_channels: int = 256,
        use_fp16: bool = False,
    ):
        """Start fast async training in a subprocess."""
        if self.process is not None and self.process.is_alive():
            return {'status': 'already_running'}

        self.command_queue = mp.Queue()
        self.metrics_queue = mp.Queue()
        self.status = 'running'

        self.process = mp.Process(
            target=self._training_loop,
            args=(
                self.command_queue,
                self.metrics_queue,
                total_steps,
                self.checkpoint_dir,
                self.log_dir,
                num_workers,
                num_simulations,
                mcts_batch_size,
                inference_batch_size,
                training_batch_size,
                buffer_capacity,
                lr,
                num_blocks,
                num_channels,
                use_fp16,
            ),
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
        num_workers: int,
        num_simulations: int,
        mcts_batch_size: int,
        inference_batch_size: int,
        training_batch_size: int,
        buffer_capacity: int,
        lr: float,
        num_blocks: int,
        num_channels: int,
        use_fp16: bool,
    ):
        """Fast async training subprocess entry point."""
        import torch
        import time
        import torch.multiprocessing as mp
        from ai.model import create_model
        from ai.trainer import Trainer, ReplayBuffer
        from ai.evaluate import evaluate_challenger, update_elo, save_elo_history, load_champion
        from ai.inference_server import start_inference_server
        from ai.worker import start_worker_pool

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model_config = {'num_blocks': num_blocks, 'num_channels': num_channels}

        model = create_model(**model_config, device=device)
        replay_buffer = ReplayBuffer(capacity=buffer_capacity)
        trainer = Trainer(
            model=model,
            mcts=None,
            replay_buffer=replay_buffer,
            device=device,
            lr=lr,
            batch_size=training_batch_size,
            checkpoint_dir=checkpoint_dir,
            log_dir=log_dir,
        )

        # Load champion if exists
        champion_model = create_model(**model_config, device=device)
        if load_champion(checkpoint_dir, champion_model, device):
            model.load_state_dict(champion_model.state_dict())

        # Start inference server
        server_process, request_queue, result_queues, control_queue = start_inference_server(
            model_config=model_config,
            num_workers=num_workers,
            device=device,
            max_batch_size=inference_batch_size,
            use_fp16=use_fp16,
        )
        control_queue.put({
            'type': 'update_weights',
            'state_dict': model.state_dict(),
        })
        time.sleep(1)

        # Start worker pool
        ctx = mp.get_context('spawn')
        game_queue = ctx.Queue(maxsize=1000)
        mcts_config = {
            'num_simulations': num_simulations,
            'batch_size': mcts_batch_size,
            'c_puct': 1.5,
            'dirichlet_alpha': 0.3,
            'dirichlet_epsilon': 0.25,
        }
        worker_processes = start_worker_pool(
            num_workers=num_workers,
            request_queue=request_queue,
            result_queues=result_queues,
            game_queue=game_queue,
            mcts_config=mcts_config,
        )

        paused = False
        elo_history = []
        champion_elo = 1500.0
        challenger_elo = 1500.0
        last_weight_sync = 0
        last_checkpoint = 0
        last_eval = 0
        games_consumed = 0
        start_time = time.time()
        last_metrics_time = start_time

        try:
            while trainer.training_step < total_steps:
                # Check commands
                while not command_queue.empty():
                    try:
                        cmd = command_queue.get_nowait()
                        if cmd == 'STOP':
                            paused = True
                            trainer.save_checkpoint(f"paused_{trainer.training_step}")
                            metrics_queue.put_nowait({
                                'status': 'paused',
                                'step': trainer.training_step,
                                'worker_count': num_workers,
                            })
                        elif cmd == 'RESUME':
                            paused = False
                            metrics_queue.put_nowait({
                                'status': 'resumed',
                                'step': trainer.training_step,
                                'worker_count': num_workers,
                            })
                        elif cmd == 'SHUTDOWN':
                            trainer.save_checkpoint(f"shutdown_{trainer.training_step}")
                            raise SystemExit
                    except:
                        break

                if paused:
                    time.sleep(0.5)
                    continue

                # Consume games from queue
                games_this_step = 0
                while not game_queue.empty() and games_this_step < 20:
                    try:
                        game_data = game_queue.get_nowait()
                        trainer.replay_buffer.add_game(game_data)
                        games_this_step += 1
                        games_consumed += 1
                    except:
                        break

                # Train if we have enough data
                if len(replay_buffer) >= training_batch_size:
                    metrics = trainer.train_step()
                    metrics['buffer_size'] = len(replay_buffer)
                    metrics['games_consumed'] = games_consumed
                    metrics['status'] = 'running'
                    metrics['step'] = trainer.training_step
                    metrics['worker_count'] = num_workers

                    # Calculate games/sec
                    elapsed = time.time() - start_time
                    metrics['games_per_hour'] = games_consumed / (elapsed / 3600) if elapsed > 0 else 0
                    metrics['inference_batch_size'] = inference_batch_size

                    # Send metrics to queue periodically (every ~1 second)
                    if time.time() - last_metrics_time >= 1.0:
                        try:
                            metrics_queue.put_nowait(metrics)
                        except:
                            pass  # Queue full, drop metric and keep training
                        last_metrics_time = time.time()

                    # Weight sync to inference server
                    if trainer.training_step - last_weight_sync >= 100:
                        control_queue.put({
                            'type': 'update_weights',
                            'state_dict': model.state_dict(),
                        })
                        last_weight_sync = trainer.training_step

                    # Checkpointing
                    if trainer.training_step - last_checkpoint >= 5000:
                        try:
                            metrics_queue.put_nowait({
                                'status': 'saving_checkpoint',
                                'step': trainer.training_step,
                                'worker_count': num_workers,
                            })
                        except:
                            pass
                        trainer.save_checkpoint(f"checkpoint_{trainer.training_step}")
                        last_checkpoint = trainer.training_step

                    # Evaluation
                    if trainer.training_step > 0 and trainer.training_step - last_eval >= 10000:
                        try:
                            metrics_queue.put_nowait({
                                'status': 'evaluating',
                                'step': trainer.training_step,
                                'worker_count': num_workers,
                            })
                        except:
                            pass
                        is_better, eval_results = evaluate_challenger(
                            champion_model,
                            model,
                            num_games=100,
                            num_simulations=num_simulations,
                            device=device,
                        )
                        challenger_score = eval_results['challenger_win_rate']
                        champion_elo, challenger_elo = update_elo(
                            champion_elo, challenger_elo, challenger_score
                        )
                        elo_history.append({
                            'step': trainer.training_step,
                            'champion_elo': champion_elo,
                            'challenger_elo': challenger_elo,
                            **eval_results,
                        })
                        save_elo_history(elo_history, log_dir)

                        if is_better:
                            champion_model.load_state_dict(model.state_dict())
                            trainer.save_checkpoint(f"best_{trainer.training_step}", is_best=True)
                            champion_elo = challenger_elo
                            challenger_elo = 1500.0

                        last_eval = trainer.training_step
                else:
                    # Not enough data yet, wait for workers
                    time.sleep(0.1)

        except SystemExit:
            pass
        except KeyboardInterrupt:
            pass
        finally:
            # Cleanup
            trainer.save_checkpoint("final")
            for p in worker_processes:
                if p.is_alive():
                    p.terminate()
            control_queue.put({'type': 'shutdown'})
            try:
                metrics_queue.put_nowait({
                    'status': 'completed',
                    'step': trainer.training_step,
                    'worker_count': num_workers,
                })
            except:
                pass


# Global training manager instance
training_manager = TrainingManager()
