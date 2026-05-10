#!/bin/bash

python -m uvicorn web.gomoku_server:app --host 0.0.0.0 --port 8188
