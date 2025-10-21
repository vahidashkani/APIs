# speech_service.py
import os
import tempfile
from typing import Tuple

from faster_whisper import WhisperModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


class SpeechService:
    """
    Speech-to-text (Whisper) + summarization with Mistral (Hugging Face).
    """

    def __init__(
        self,
        whisper_model_size: str = "large-v2",
        whisper_device: str = "cpu",
        whisper_compute_type: str = "int8",
        llm_model: str = "mistralai/Mistral-7B-Instruct-v0.2",
    ):
        # ---------- Whisper (STT) ----------
        self._whisper = WhisperModel(
            whisper_model_size,
            device=whisper_device,
            compute_type=whisper_compute_type,
        )

        # ---------- Mistral LLM ----------
        print("Loading Mistral model (this may take a while the first time)...")
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model)
        self.model = AutoModelForCausalLM.from_pretrained(
            llm_model,
            device_map="auto",     # GPU if available
            torch_dtype="auto"
        )
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )

    def transcribe(self, audio_path: str) -> Tuple[str, float]:
        """
        Transcribe audio and translate to English.
        """
        segments, info = self._whisper.transcribe(
            audio_path,
            task="translate",   # always produce English
            language=None,      # auto-detect input
        )
        transcript = " ".join(seg.text.strip() for seg in segments if seg.text)
        return transcript, getattr(info, "duration", 0.0) or 0.0

    def summarize(self, text: str, max_words: int = 500) -> str:
        """
        Summarize using Mistral (prompt-based).
        max_words ~ desired length of summary (words).
        We map words → tokens: tokens ≈ words * 1.4
        """
        approx_tokens = int(max_words * 1.4)
        prompt = f"Summarize the following text in about {max_words} words:\n\n{text}\n\nSummary:"
        outputs = self.generator(
            prompt,
            max_new_tokens=approx_tokens,
            do_sample=False,
            temperature=0.2,
        )
        return outputs[0]["generated_text"].split("Summary:")[-1].strip()

    def transcribe_and_summarize(self, audio_bytes: bytes, filename_hint: str = "audio", max_words: int = 500) -> dict:
        suffix = os.path.splitext(filename_hint)[-1] or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
            tmp.write(audio_bytes)
            tmp.flush()
            transcript, duration = self.transcribe(tmp.name)

        summary = self.summarize(transcript, max_words=max_words)

        return {
            "duration_sec": duration,
            "transcript": transcript,
            "summary": summary,
            "llm_model": "Mistral",
            "target_summary_words": max_words,
        }

