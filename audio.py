"""Generate overlapping-speech clips: one Piper TTS voice reads a target sentence while five
other voices read random filler fragments in the background.

Usage: python audio.py corpus.json audio/

Voice models are expected in voices/:
    python -m piper.download_voices <voice> ... --download-dir voices
"""
import json
import random
import subprocess
import sys
from pathlib import Path

from pydub import AudioSegment

VOICES_DIR = Path("voices")

TARGET_VOICE = "en_US-amy-medium"
BABBLE_VOICES = [
    "en_US-ryan-medium",
    "en_US-danny-low",
    "en_US-kathleen-low",
    "en_US-joe-medium",
    "en_US-hfc_female-medium",
]

BABBLE_ATTENUATION_DB = 4
ONSET_JITTER_MS = 800

random.seed(42)


def speak(text: str, voice: str, out_path: Path) -> AudioSegment:
    model = VOICES_DIR / f"{voice}.onnx"
    config = VOICES_DIR / f"{voice}.onnx.json"
    subprocess.run(
        ["python", "-m", "piper", "-m", str(model), "-c", str(config), "-f", str(out_path)],
        input=text,
        text=True,
        check=True,
        capture_output=True,
    )
    audio = AudioSegment.from_wav(out_path)
    out_path.unlink()
    return audio


def make_babble_track(voice: str, filler_pool: list, min_duration_ms: int, tmp_path: Path) -> AudioSegment:
    track = AudioSegment.empty()
    while len(track) < min_duration_ms:
        sentence = random.choice(filler_pool)
        track += speak(sentence, voice, tmp_path) + AudioSegment.silent(duration=200)

    offset = random.randint(0, ONSET_JITTER_MS)
    track = AudioSegment.silent(duration=offset) + track
    return track[:min_duration_ms]


def make_clip(target_text: str, filler_pool: list, out_path: Path) -> None:
    tmp_path = out_path.with_suffix(".tmp.wav")

    target = speak(target_text, TARGET_VOICE, tmp_path)

    babble_tracks = [make_babble_track(voice, filler_pool, len(target), tmp_path) for voice in BABBLE_VOICES]
    babble = babble_tracks[0]
    for track in babble_tracks[1:]:
        babble = babble.overlay(track)
    babble = babble - BABBLE_ATTENUATION_DB

    mixed = target.overlay(babble)
    mixed.export(out_path, format="wav")


def main(corpus_path: Path, audio_dir: Path) -> None:
    corpus = json.loads(corpus_path.read_text())
    audio_dir.mkdir(exist_ok=True)
    for i, clip in enumerate(corpus["clips"]):
        make_clip(clip["target"], corpus["filler"], audio_dir / f"clip_{i:02d}.wav")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
