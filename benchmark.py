"""Transcribe each clip with a set of speech recognition models, one pass per clip.

Usage: python benchmark.py corpus.json audio/ model_results.json [engine ...]

The gemini_* engines need GEMINI_API_KEY. The Vosk model is expected in vosk-models/:
    https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import whisper
from google import genai
from google.genai import types
from pydub import AudioSegment
from transformers import AutoModelForCTC, AutoModelForMultimodalLM, AutoProcessor, pipeline
from vosk import KaldiRecognizer, Model as VoskModel

WHISPER_PROMPT = "Transcribe only the clear foreground voice, ignoring background chatter."

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
GEMINI_PROMPTS = {
    "naive": "Transcribe this audio. Output only the transcript text, nothing else.",
    "instructed": (
        "This recording contains one main speaker reading a single sentence, with several "
        "other people chattering in the background. Transcribe only the main speaker's "
        "sentence, word for word, and ignore the background voices. Output only the "
        "transcript text, with no commentary or labels."
    ),
}

VOSK_MODEL_DIR = Path("vosk-models/vosk-model-small-en-us-0.15")


def to_16k_mono_pcm(clip_path: Path) -> bytes:
    audio = AudioSegment.from_wav(clip_path).set_frame_rate(16000).set_channels(1).set_sample_width(2)
    return audio.raw_data


def to_16k_float_array(clip_path: Path) -> np.ndarray:
    raw = to_16k_mono_pcm(clip_path)
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def whisper_engine(size: str):
    model = whisper.load_model(size)

    def transcribe(clip_path: Path) -> str:
        return model.transcribe(str(clip_path), initial_prompt=WHISPER_PROMPT)["text"]

    return transcribe


def vosk_engine():
    model = VoskModel(str(VOSK_MODEL_DIR))

    def transcribe(clip_path: Path) -> str:
        rec = KaldiRecognizer(model, 16000)
        rec.AcceptWaveform(to_16k_mono_pcm(clip_path))
        return json.loads(rec.FinalResult()).get("text", "")

    return transcribe


def wav2vec2_engine():
    asr = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h")

    def transcribe(clip_path: Path) -> str:
        arr = to_16k_float_array(clip_path)
        return asr({"array": arr, "sampling_rate": 16000})["text"]

    return transcribe


def qwen3_asr_engine(model_id: str = "Qwen/Qwen3-ASR-0.6B-hf"):
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForMultimodalLM.from_pretrained(model_id, dtype=torch.float32)

    def transcribe(clip_path: Path) -> str:
        inputs = processor.apply_transcription_request(audio=str(clip_path)).to(model.device, model.dtype)
        output_ids = model.generate(**inputs, max_new_tokens=256)
        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        return processor.decode(generated_ids, return_format="transcription_only")[0]

    return transcribe


def parakeet_ctc_engine(model_id: str = "nvidia/parakeet-ctc-0.6b"):
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForCTC.from_pretrained(model_id, dtype="auto")

    def transcribe(clip_path: Path) -> str:
        arr = to_16k_float_array(clip_path)
        inputs = processor(arr, sampling_rate=16000)
        inputs.to(model.device, dtype=model.dtype)
        outputs = model.generate(**inputs)
        return processor.batch_decode(outputs)[0]

    return transcribe


def gemini_engine(variant: str):
    client = genai.Client()
    prompt = GEMINI_PROMPTS[variant]
    config = types.GenerateContentConfig(temperature=0)

    def transcribe(clip_path: Path) -> str:
        audio = types.Part.from_bytes(data=clip_path.read_bytes(), mime_type="audio/wav")
        resp = client.models.generate_content(model=GEMINI_MODEL, contents=[prompt, audio], config=config)
        return (resp.text or "").strip()

    return transcribe


ENGINES = {
    "whisper_base": lambda: whisper_engine("base"),
    "whisper_small": lambda: whisper_engine("small"),
    "vosk_small": vosk_engine,
    "wav2vec2_base": wav2vec2_engine,
    "qwen3_asr_0.6b": qwen3_asr_engine,
    "parakeet_ctc_0.6b": parakeet_ctc_engine,
    "gemini_naive": lambda: gemini_engine("naive"),
    "gemini_instructed": lambda: gemini_engine("instructed"),
}


def main(corpus_path: Path, audio_dir: Path, results_path: Path, engines: list) -> None:
    clips = [c["id"] for c in json.loads(corpus_path.read_text())["clips"]]
    results = {"models": {}}
    for name in engines or ENGINES:
        transcribe = ENGINES[name]()
        results["models"][name] = {"per_clip": {clip: {"predicted": transcribe(audio_dir / clip)} for clip in clips}}
        results_path.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4:])
