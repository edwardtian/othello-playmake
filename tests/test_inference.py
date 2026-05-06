"""Integration tests for inference server + worker pool."""

import pytest
import time
import numpy as np
import torch
import torch.multiprocessing as mp

from ai.model import OthelloNet
from ai.inference_server import InferenceServer, start_inference_server
from ai.worker import InferenceClient, SelfPlayWorker, start_worker_pool
from game.othello import OthelloGame


class TestInferenceServer:
    def test_server_evaluates_batch(self):
        """Test that inference server can evaluate a batch of states."""
        model_config = {'num_blocks': 2, 'num_channels': 64}
        ctx = mp.get_context('spawn')
        request_queue = ctx.Queue()
        result_queue = ctx.Queue()
        control_queue = ctx.Queue()

        server = InferenceServer(model_config, device='cpu', max_batch_size=8)
        p = ctx.Process(target=server.run, args=(request_queue, [result_queue], control_queue))
        p.start()
        time.sleep(2)

        # Send batch of requests
        for i in range(4):
            state = np.random.randn(3, 8, 8).astype(np.float32)
            request_queue.put({
                'request_id': f'w0_test_{i}',
                'state': state,
            })

        # Wait for server to process
        time.sleep(3)

        # Collect results
        results = {}
        for _ in range(20):
            try:
                result = result_queue.get(timeout=0.5)
                results[result['request_id']] = result
            except:
                break

        control_queue.put({'type': 'shutdown'})
        p.join(timeout=10)

        assert len(results) == 4
        for i in range(4):
            assert f'w0_test_{i}' in results
            assert results[f'w0_test_{i}']['policy'].shape == (65,)
            assert np.isclose(results[f'w0_test_{i}']['policy'].sum(), 1.0, atol=1e-5)

    def test_server_weight_update(self):
        """Test that server can receive weight updates."""
        model_config = {'num_blocks': 2, 'num_channels': 64}
        ctx = mp.get_context('spawn')
        request_queue = ctx.Queue()
        result_queue = ctx.Queue()
        control_queue = ctx.Queue()

        server = InferenceServer(model_config, device='cpu', max_batch_size=8)
        p = ctx.Process(target=server.run, args=(request_queue, [result_queue], control_queue))
        p.start()
        time.sleep(2)

        # Create a model and send its weights
        model = OthelloNet(**model_config)
        control_queue.put({
            'type': 'update_weights',
            'state_dict': model.state_dict(),
        })
        time.sleep(1)

        # Verify it still works after weight update
        state = np.random.randn(3, 8, 8).astype(np.float32)
        request_queue.put({
            'request_id': 'w0_after_update',
            'state': state,
        })

        result = result_queue.get(timeout=5)
        assert result['request_id'] == 'w0_after_update'

        control_queue.put({'type': 'shutdown'})
        p.join(timeout=10)


class TestInferenceClient:
    def test_client_evaluates_state(self):
        """Test that client can evaluate a state through the server."""
        model_config = {'num_blocks': 2, 'num_channels': 64}
        ctx = mp.get_context('spawn')
        request_queue = ctx.Queue()
        result_queue = ctx.Queue()
        control_queue = ctx.Queue()

        server = InferenceServer(model_config, device='cpu', max_batch_size=8)
        p = ctx.Process(target=server.run, args=(request_queue, [result_queue], control_queue))
        p.start()
        time.sleep(2)

        client = InferenceClient(request_queue, result_queue, worker_id=0)
        state = np.random.randn(3, 8, 8).astype(np.float32)
        policy, value = client.evaluate(state)

        assert policy.shape == (65,)
        assert np.isclose(policy.sum(), 1.0, atol=1e-5)
        assert -1.0 <= value <= 1.0

        control_queue.put({'type': 'shutdown'})
        p.join(timeout=10)

    def test_client_batch_evaluate(self):
        """Test batch evaluation through client."""
        model_config = {'num_blocks': 2, 'num_channels': 64}
        ctx = mp.get_context('spawn')
        request_queue = ctx.Queue()
        result_queue = ctx.Queue()
        control_queue = ctx.Queue()

        server = InferenceServer(model_config, device='cpu', max_batch_size=16)
        p = ctx.Process(target=server.run, args=(request_queue, [result_queue], control_queue))
        p.start()
        time.sleep(2)

        client = InferenceClient(request_queue, result_queue, worker_id=0)
        states = np.random.randn(8, 3, 8, 8).astype(np.float32)
        policies, values = client.evaluate_batch(states)

        assert policies.shape == (8, 65)
        assert values.shape == (8,)
        for i in range(8):
            assert np.isclose(policies[i].sum(), 1.0, atol=1e-5)
            assert -1.0 <= values[i] <= 1.0

        control_queue.put({'type': 'shutdown'})
        p.join(timeout=10)


class TestWorkerIntegration:
    def test_worker_plays_game(self):
        """Test that a worker can play a complete game via inference server."""
        model_config = {'num_blocks': 2, 'num_channels': 64}
        ctx = mp.get_context('spawn')
        request_queue = ctx.Queue()
        result_queue = ctx.Queue()
        control_queue = ctx.Queue()
        game_queue = ctx.Queue()

        # Start server
        server = InferenceServer(model_config, device='cpu', max_batch_size=16)
        server_p = ctx.Process(target=server.run, args=(request_queue, [result_queue], control_queue))
        server_p.start()
        time.sleep(2)

        # Start worker
        mcts_config = {
            'model': None,
            'evaluator': None,
            'num_simulations': 16,
            'batch_size': 4,
        }
        worker = SelfPlayWorker(
            worker_id=0,
            request_queue=request_queue,
            result_queue=result_queue,
            game_queue=game_queue,
            mcts_config=mcts_config,
        )
        worker_p = ctx.Process(target=worker.run, args=(1,))  # Play 1 game
        worker_p.start()

        # Wait for game
        game_data = game_queue.get(timeout=60)
        assert len(game_data) > 0

        worker_p.join(timeout=10)
        control_queue.put({'type': 'shutdown'})
        server_p.join(timeout=10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
