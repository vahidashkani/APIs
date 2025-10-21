#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  4 10:50:34 2025

@author: nca
"""

# main.py
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import shutil
import uuid
from pathlib import Path
from stt_service import convert_speech_to_text

app = FastAPI()

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

@app.post("/start-session")
async def start_session(file: UploadFile = File(...)):
    try:
        # Save file
        session_id = str(uuid.uuid4())
        file_path = UPLOAD_DIR / f"{session_id}_{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Run STT
        transcription = convert_speech_to_text(str(file_path))

        return JSONResponse({
            "session_id": session_id,
            "transcription": transcription
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/chat")
async def chat(session_id: str = Form(...), question: str = Form(...)):
    # For now, just echo back
    return {"response": f"You asked: {question} for session {session_id}"}


