#!/bin/bash --login

srun --pty --partition agent-dev,agent-dev-15 --gres=gpu:1 -t 04:00:00 bash
