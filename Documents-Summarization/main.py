# main.py
import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional

from speech_service import SpeechService

app = FastAPI(
    title="Speech-to-English + Free Summarization API",
    version="1.0.0",
    description="Upload speech audio → English transcript + free summary (no API key needed).",
)

# Single shared instance
service = SpeechService()


class SummarizeRequest(BaseModel):
    text: str
    max_words: Optional[int] = 120

class SummarizeResponse(BaseModel):
    summary: str


@app.get("/health")
def health():
    return {
        "status": "ok",
        "whisper_model": service.whisper_model_size,
        "summarizer": "facebook/bart-large-cnn",
    }


@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(..., description="Audio file"),
):
    try:
        audio_bytes = await file.read()
        tmp_path = _save_temp(audio_bytes, file.filename or "audio.wav")
        transcript, duration = service.transcribe(tmp_path)
        os.remove(tmp_path)
        return JSONResponse({"duration_sec": duration, "transcript": transcript})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")


def _save_temp(b: bytes, fname_hint: str) -> str:
    import tempfile
    suffix = os.path.splitext(fname_hint)[-1] or ".wav"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.write(b)
    tmp.flush()
    tmp.close()
    return tmp.name


@app.post("/summarize", response_model=SummarizeResponse)
async def summarize_text(req: SummarizeRequest):
    try:
        summary = service.summarize(req.text, max_words=req.max_words)
        return {"summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summarization failed: {e}")


@app.post("/analyze")
async def analyze_audio(
    file: UploadFile = File(..., description="Audio file"),
    max_words: Optional[int] = 120,
):
    try:
        audio_bytes = await file.read()
        result = service.transcribe_and_summarize(
            audio_bytes=audio_bytes,
            filename_hint=file.filename or "audio.wav",
            max_words=max_words,
        )
        return JSONResponse(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analyze failed: {e}")

