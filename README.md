# Crosstalk

Code and data for **[Can Machines Follow One Voice Through a Crowded Room?](https://www.kelvinjiang.com/2026/10/can-machines-follow-voice-through-crowded-room.html)**,
a small study testing whether "cocktail party" audio can work as a reverse Turing test: one speaker talking over
background chatter, transcribed by people and by speech recognition models.

**Short answer: yes, well enough.** Humans still lead (0.71 vs. 0.54 mean word accuracy against the best model,
Gemini 3.8 Flash), but the gap is too narrow to tell people and machines apart, and models are only getting better. So
cocktail-party audio doesn't work as a reverse Turing test. Check out the article for the method, the results, and
what didn't work:

[![Mean word accuracy: humans lead, followed by each model from best to worst](https://www.kelvinjiang.com/assets/img/crosstalk_chart_leaderboard.svg)](https://www.kelvinjiang.com/2026/10/can-machines-follow-voice-through-crowded-room.html)

## Code & data

| File | Contents |
|---|---|
| `audio/` | The 40 test clips exactly as used in the study |
| `corpus.json` | Target sentences, the background filler fragments, and each clip's SHA-256 |
| `model_results.json` | Transcripts from 8 model configurations (Whisper, Vosk, Wav2Vec2, Qwen3-ASR, Parakeet, Gemini) |
| `human_results.jsonl` | 146 human transcriptions from 18 sessions |
| `audio.py` | Clip generation: Piper TTS voices mixed with pydub |
| `benchmark.py` | Runs each model over the clips |
| `scoring.py` | Word accuracy (1 − WER), pass rates, the human-model gap, AUC, and bootstrap CIs |

Notes:
* Human responses are anonymized: each row has a random session id, the clip, the typed answer, and the number of
  plays. No other participant data is included.
* The first 30 answers have no `plays` field indicating play counts as those started getting recorded at a later point

## Running

Scoring:

```
python3 scoring.py corpus.json model_results.json human_results.jsonl
```

Rerunning the models requires `GEMINI_API_KEY` for the Gemini engines (see `benchmark.py` for model setup):

```
python3 benchmark.py corpus.json audio/ model_results.json [engine ...]
```

`audio.py` regenerates clips with `piper-tts`, `pydub` and ffmpeg. Piper's output isn't bit-for-bit deterministic,
so the files in `audio/` (matching the hashes in `corpus.json`) are reference copies.

## License

- **Code**: MIT, see [LICENSE](LICENSE).
- **Data**: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Please credit and link the
  [article](https://www.kelvinjiang.com/2026/10/can-machines-follow-voice-through-crowded-room.html).
- **Audio** (`audio/`): [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). The clips are synthesized
  with [Piper voices](https://huggingface.co/rhasspy/piper-voices) whose training data carries its own terms, including
  CC BY-NC-SA 4.0 for the `ryan` and `hfc_female` voices used in every clip.
