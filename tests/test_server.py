"""Tests for FastAPI backend."""

import pytest
from fastapi.testclient import TestClient

from web.server import app, _sessions

client = TestClient(app)


class TestGameAPI:
    def test_new_game(self):
        response = client.post("/api/game/new", json={"mode": "human-black"})
        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert data["mode"] == "human-black"
        assert data["current_player"] == 1  # Black starts
        assert not data["is_game_over"]

    def test_make_valid_move(self):
        # Create game
        response = client.post("/api/game/new", json={"mode": "human-black"})
        session_id = response.json()["session_id"]

        # Make a valid move (19 = (2,3))
        response = client.post(f"/api/game/{session_id}/move", json={"action": 19})
        assert response.status_code == 200
        data = response.json()
        assert data["current_player"] == 2  # Now white's turn
        assert data["move_count"] == 1

    def test_make_invalid_move(self):
        response = client.post("/api/game/new", json={"mode": "human-black"})
        session_id = response.json()["session_id"]

        # Invalid move (0 = corner)
        response = client.post(f"/api/game/{session_id}/move", json={"action": 0})
        assert response.status_code == 400

    def test_get_state(self):
        response = client.post("/api/game/new", json={"mode": "human-white"})
        session_id = response.json()["session_id"]

        response = client.get(f"/api/game/{session_id}/state")
        assert response.status_code == 200
        data = response.json()
        assert data["mode"] == "human-white"

    def test_ai_move(self):
        response = client.post("/api/game/new", json={"mode": "human-black"})
        session_id = response.json()["session_id"]

        response = client.post(f"/api/game/{session_id}/ai_move")
        assert response.status_code == 200
        data = response.json()
        assert "ai_move" in data
        assert data["ai_move"]["success"] is True

    def test_session_not_found(self):
        response = client.get("/api/game/invalid/state")
        assert response.status_code == 404

    def test_thinking_data(self):
        response = client.post("/api/game/new", json={"mode": "human-black"})
        session_id = response.json()["session_id"]

        # No thinking data before AI move
        response = client.post(f"/api/game/{session_id}/thinking")
        assert response.status_code == 400

        # Make AI move
        client.post(f"/api/game/{session_id}/ai_move")
        response = client.post(f"/api/game/{session_id}/thinking")
        assert response.status_code == 200
        data = response.json()
        assert "visit_counts" in data
        assert len(data["visit_counts"]) == 65


class TestCheckpointAPI:
    def test_load_missing_checkpoint(self):
        response = client.post("/api/ai/load_checkpoint", json={"checkpoint_path": "nonexistent.pt"})
        assert response.status_code == 404


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
