"""
Centralized Inference Server for Othello AI.

Runs in a dedicated process, holding the neural network model.
Workers send batches of board states via Queue and receive (policy, value) back.
Each worker has its own dedicated result queue to prevent cross-contamination.
"""

import time
import numpy as np
import torch
import torch.multiprocessing as mp
from typing import Tuple, Dict, Any, List
from ai.model import OthelloNet, create_model


class InferenceServer:
    """
    Centralized model inference process.
    
    Receives evaluation requests from workers via request_queue,
    batches them, runs model.forward(), and returns results via per-worker result queues.
    """

    def __init__(
        self,
        model_config: Dict[str, Any],
        device: str = 'cpu',
        max_batch_size: int = 64,
        max_wait_ms: float = 5.0,
        use_fp16: bool = False,
    ):
        self.model_config = model_config
        self.device = device
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.use_fp16 = use_fp16 and device.startswith('cuda')
        self.model = None
        self.stats = {
            'batches_processed': 0,
            'positions_evaluated': 0,
            'avg_batch_size': 0.0,
        }

    def _load_model(self):
        """Load model into memory."""
        if self.device.startswith('cuda'):
            torch.set_float32_matmul_precision('high')
        self.model = create_model(**self.model_config, device=self.device)
        self.model.eval()
        print(f"[InferenceServer] Model loaded on {self.device}")
        print(f"[InferenceServer] Parameters: {self.model.count_parameters():,}")
        if self.use_fp16:
            print("[InferenceServer] FP16 mixed-precision inference enabled")

    @staticmethod
    def _extract_worker_id(request_id: str) -> int:
        """Extract worker id from request_id like 'w0_r123'."""
        # Format: w{worker_id}_r{counter}
        parts = request_id.split('_')
        if len(parts) >= 1 and parts[0].startswith('w'):
            return int(parts[0][1:])
        return 0

    def _evaluate_and_respond(self, batch, result_queues: List[mp.Queue]):
        """Evaluate a batch and send results back to per-worker queues."""
        try:
            states = np.stack([req['state'] for req in batch])
            states_tensor = torch.from_numpy(states).float().to(self.device)
            
            with torch.no_grad():
                if self.use_fp16:
                    with torch.amp.autocast('cuda'):
                        policy_logits, values = self.model(states_tensor)
                else:
                    policy_logits, values = self.model(states_tensor)
                policies = torch.softmax(policy_logits, dim=-1).cpu().numpy()
                values = values.squeeze(-1).cpu().numpy()

            for i, req in enumerate(batch):
                worker_id = self._extract_worker_id(req['request_id'])
                result_queue = result_queues[worker_id] if worker_id < len(result_queues) else result_queues[0]
                result_queue.put({
                    'request_id': req['request_id'],
                    'policy': policies[i],
                    'value': values[i],
                })

            self.stats['batches_processed'] += 1
            self.stats['positions_evaluated'] += len(batch)
            self.stats['avg_batch_size'] = (
                0.9 * self.stats['avg_batch_size'] + 0.1 * len(batch)
            )

        except Exception as e:
            print(f"[InferenceServer] Error evaluating batch: {e}")
            for req in batch:
                worker_id = self._extract_worker_id(req['request_id'])
                result_queue = result_queues[worker_id] if worker_id < len(result_queues) else result_queues[0]
                result_queue.put({
                    'request_id': req['request_id'],
                    'policy': np.ones(65, dtype=np.float32) / 65,
                    'value': 0.0,
                })

    def run(self, request_queue: mp.Queue, result_queues: List[mp.Queue], control_queue: mp.Queue):
        """
        Main server loop.
        
        Args:
            request_queue: Workers put {'state': np.array, 'request_id': str} here
            result_queues: List of queues, one per worker. Server puts results in the appropriate queue.
            control_queue: For weight updates and shutdown signals
        """
        self._load_model()
        running = True

        while running:
            # Check for control signals (non-blocking)
            while not control_queue.empty():
                try:
                    cmd = control_queue.get_nowait()
                    if cmd['type'] == 'shutdown':
                        running = False
                        break
                    elif cmd['type'] == 'update_weights':
                        state_dict = cmd['state_dict']
                        self.model.load_state_dict(state_dict)
                        print("[InferenceServer] Weights updated")
                except:
                    break

            if not running:
                # Process any remaining requests before shutdown
                batch = []
                while not request_queue.empty():
                    try:
                        req = request_queue.get_nowait()
                        if req is not None:
                            batch.append(req)
                    except:
                        break
                if batch:
                    self._evaluate_and_respond(batch, result_queues)
                break

            # Collect batch of requests
            batch = []
            start_time = time.time()
            
            while len(batch) < self.max_batch_size:
                elapsed_ms = (time.time() - start_time) * 1000
                if elapsed_ms >= self.max_wait_ms and len(batch) > 0:
                    break
                
                try:
                    req = request_queue.get(timeout=0.001)
                    if req is None:
                        running = False
                        break
                    batch.append(req)
                except:
                    if len(batch) > 0:
                        break
                    continue

            if not batch:
                continue

            self._evaluate_and_respond(batch, result_queues)

        print(f"[InferenceServer] Shutting down. Stats: {self.stats}")


def start_inference_server(
    model_config: Dict[str, Any],
    num_workers: int,
    device: str = 'cpu',
    max_batch_size: int = 64,
    use_fp16: bool = False,
) -> Tuple[mp.Process, mp.Queue, List[mp.Queue], mp.Queue]:
    """
    Start inference server in a background process.
    
    Args:
        model_config: Model configuration dict
        num_workers: Number of workers (determines number of result queues)
        device: Device to run model on
        max_batch_size: Maximum batch size for inference
        use_fp16: Use FP16 mixed-precision for inference (GPU only)
    
    Returns:
        (process, request_queue, result_queues, control_queue)
        result_queues is a list of mp.Queue, one per worker.
    """
    ctx = mp.get_context('spawn')
    request_queue = ctx.Queue(maxsize=1024)
    result_queues = [ctx.Queue(maxsize=1024) for _ in range(num_workers)]
    control_queue = ctx.Queue(maxsize=16)

    server = InferenceServer(model_config, device, max_batch_size, use_fp16=use_fp16)
    process = ctx.Process(
        target=server.run,
        args=(request_queue, result_queues, control_queue),
        daemon=True,
    )
    process.start()
    
    # Give server time to load model
    time.sleep(2)
    
    return process, request_queue, result_queues, control_queue
