# app/main.py
import io
import os
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
import soundfile as sf  # import once

from app.model_service import EnhancerService

# -------- Config via env (override in docker run) --------
MODEL_PATH = os.environ.get("MODEL_PATH", "./best_ckpt/ckpt")
N_FFT = int(os.environ.get("N_FFT", "400"))
HOP = int(os.environ.get("HOP", "100"))
SR = int(os.environ.get("SAMPLE_RATE", "16000"))
CUT_LEN = int(os.environ.get("CUT_LEN", str(16000 * 16)))

app = FastAPI(title="Noisy-to-Enhanced API (TSCNet)")

# load model once
try:
    svc = EnhancerService(
        model_path=MODEL_PATH,
        n_fft=N_FFT,
        hop=HOP,
        sample_rate=SR,
        num_channel=64,
    )
except Exception as e:
    raise RuntimeError(f"Failed to initialize model: {e}")

@app.get("/")
def root():
    return {
        "status": "ok",
        "device": str(svc.device),
        "model_path": MODEL_PATH,
        "sr": SR,
        "n_fft": N_FFT,
        "hop": HOP,
    }

@app.post("/enhance")
async def enhance_endpoint(
    file: UploadFile = File(...),
    return_wav: bool = Form(True),
):
    """
    Enhance a single noisy file.
    - If return_wav=True (default): returns enhanced audio as audio/wav stream.
    - If return_wav=False: returns JSON with length/sample_rate.
    """
    try:
        data = await file.read()
        tmp_in = Path("/tmp") / file.filename
        tmp_in.write_bytes(data)

        enhanced = svc.enhance_file(str(tmp_in), cut_len=CUT_LEN)

        if return_wav:
            buf = io.BytesIO()
            sf.write(buf, enhanced, svc.sample_rate, format="WAV")
            buf.seek(0)
            headers = {
                "Content-Disposition": f'attachment; filename="{Path(file.filename).stem}_enhanced.wav"'
            }
            return StreamingResponse(buf, media_type="audio/wav", headers=headers)
        else:
            return {"length": int(enhanced.shape[0]), "sample_rate": svc.sample_rate}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Enhance failed: {e}")

@app.post("/enhance-zip")
async def enhance_zip_endpoint(noisy_zip: UploadFile = File(...)):
    """
    Upload a ZIP of WAVs → get a ZIP back with enhanced WAVs (same names).
    """
    try:
        nz = await noisy_zip.read()
        out_zip = svc.enhance_zip(nz, cut_len=CUT_LEN)
        headers = {"Content-Disposition": 'attachment; filename="enhanced.zip"'}
        return StreamingResponse(io.BytesIO(out_zip), media_type="application/zip", headers=headers)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Enhance-zip failed: {e}")

