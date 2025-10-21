#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  4 10:48:31 2025

@author: nca
"""

# stt_service.py
from faster_whisper import WhisperModel
from pathlib import Path
from typing import Tuple

# Choose size: tiny, base, small, medium, large-v2, large-v3
# You can also set this with env WHISPER_MODEL_SIZE
import os
MODEL_SIZE = os.environ.get("WHISPER_MODEL_SIZE", "large-v2")

# device="cpu" for AWS CPU instances; set compute_type to int8 for speed/ram
# If you have a GPU, set device="cuda" and compute_type="float16"
model = WhisperModel(
    MODEL_SIZE,
    device=os.environ.get("WHISPER_DEVICE", "cpu"),
    compute_type=os.environ.get("WHISPER_COMPUTE_TYPE", "int8")
)

def convert_speech_to_text(audio_file: str) -> str:
    """
    Transcribe a WAV/MP3/M4A/etc. file to text using faster-whisper.
    """
    # vad_filter helps with long pauses/noise; adjust beam_size for accuracy/speed
    segments, info = model.transcribe(
        audio_file,
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
    )
    out = []
    for seg in segments:
        out.append(seg.text.strip())
    return " ".join(out).strip()

