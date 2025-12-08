#!/usr/bin/env python3
# runs parakeet in memory, transcribes, returns to hyprwhspr
import io
import os
import tempfile
from typing import Optional, List

import nemo.collections.asr as nemo_asr
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Header
from pydantic import BaseModel
import soundfile as sf
import uvicorn


MODEL_NAME = "nvidia/parakeet-tdt-0.6b-v3"
USE_GPU = True  # flip to False if you want to force CPU


class TranscriptionResponse(BaseModel):
    text: str


app = FastAPI(title="Hyprwhspr Parakeet Backend")

print(f"[PARAKEET] Loading model {MODEL_NAME}...")
asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=MODEL_NAME)

if USE_GPU:
    try:
        asr_model = asr_model.to("cuda")
        print("[PARAKEET] Moved model to CUDA")
    except Exception as e:
        print(f"[PARAKEET] Could not move model to CUDA: {e}")

asr_model.eval()
print("[PARAKEET] Model ready.")


def _ensure_mono_16k(tmp_path: str) -> str:
    """
    Ensure audio is mono 16kHz, converting if necessary.
    For v1, we assume hyprwhspr is already sending mono 16k wav,
    and just validate; you can add resampling here if needed.
    """
    data, sr = sf.read(tmp_path, dtype="float32", always_2d=True)
    # data: shape (num_samples, num_channels)

    if data.shape[1] != 1:
        raise ValueError(f"Expected mono audio, got {data.shape[1]} channels")

    if sr != 16000:
        # Fail fast!!
        raise ValueError(f"Expected 16kHz audio, got {sr} Hz")

    return tmp_path


@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe(
    file: UploadFile = File(...),
    language: Optional[str] = Form(default=None),
    authorization: Optional[str] = Header(default=None),
):
    # Auth here?

    if file.content_type not in ("audio/wav", "audio/x-wav", "audio/flac"):
        raise HTTPException(status_code=400, detail=f"Unsupported content_type {file.content_type}")

    # Read bytes
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio payload")

    # Persist to a temporary file – matches NeMo's file-path-based transcribe API
    try:
        suffix = ".wav" if file.filename.endswith(".wav") else ".flac"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        # Validate/inforce mono 16k
        try:
            _ensure_mono_16k(tmp_path)
        except ValueError as e:
            os.unlink(tmp_path)
            raise HTTPException(status_code=400, detail=str(e))

        # Call Parakeet
        # NeMo example: output = asr_model.transcribe(['2086-149220-0033.wav'])
        # So we follow that exactly:
        outputs: List = asr_model.transcribe([tmp_path])

        if not outputs:
            os.unlink(tmp_path)
            raise HTTPException(status_code=500, detail="No output from ASR model")

        result = outputs[0]

        # The example uses output[0].text
        if hasattr(result, "text"):
            text = result.text
        elif isinstance(result, str):
            text = result
        else:
            # Fallback for unexpected object types
            text = str(result)

        os.unlink(tmp_path)
        return TranscriptionResponse(text=text.strip())

    except HTTPException:
        # Propagate HTTPException as-is
        raise
    except Exception as e:
        # Hard failure from ASR / decoding
        if "tmp_path" in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise HTTPException(status_code=500, detail=f"ASR error: {e}") from e


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)


