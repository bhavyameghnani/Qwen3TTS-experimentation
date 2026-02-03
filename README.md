# Qwen3TTS Experimentation

This repository contains local experiments for running **Qwen3 Text-to-Speech (TTS)** models using PyTorch.  
It is intended for experimentation and research, while keeping large assets and local paths out of version control.

---

## Overview

- Experiments with Qwen3 TTS models
- CPU-based setup
- No model weights committed to the repository
- Local paths and configs are kept private

---

## Model Path Configuration

Model paths are not hardcoded in the source code.

Create a local configuration file:
 local_config.py (not committed)
'''
MODEL_PATH = "/absolute/path/to/qwen3-tts-model"
This file is excluded via .gitignore.
'''

A template is provided for reference:

local_config.example.py
MODEL_PATH = "/path/to/your/local/qwen3-tts-model"
Example Usage
from local_config import MODEL_PATH
import torch
from qwen3_tts import Qwen3TTSModel
```
model = Qwen3TTSModel.from_pretrained(
    MODEL_PATH,
    device_map="cpu",
    dtype=torch.float32,
)
```
# Environment Setup

Recommended:
Python 3.11

PyTorch

Virtual environment
