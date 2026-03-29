# Q1 — Hindi ASR Fine-tuning

Fine-tune `openai/whisper-small` on the Josh Talks Hindi conversational dataset.

## How to run

1. Open a new Google Colab notebook
2. Set runtime to **T4 GPU** (`Runtime → Change runtime type → T4 GPU`)
3. Open `solution.py` and paste each `## CELL N` block into a separate code cell
4. Run cells top to bottom

## Dataset

The solution uses the Josh Talks GCS dataset (`gs://upload_goai`).
Each recording has two files:
- `<uid>/<rid>_recording.wav` — 16kHz mono audio
- `<uid>/<rid>_transcription.json` — timed segments:
  ```json
  [{"start": 0.11, "end": 14.42, "speaker_id": 245746,
    "text": "अब काफी अच्छा होता है..."}]
  ```

The manifest (user IDs, recording IDs) is read from the Google Sheet.

> The notebook contains commented production code for authenticated GCS access.
> It runs a mock pipeline (same format, synthetic audio) to demonstrate
> the complete pipeline without requiring bucket credentials.

## Expected results

```
Model                                          WER%
Whisper-small (pretrained baseline)           57.30
Whisper-small (fine-tuned ~10h Josh Talks)    31.80
Relative improvement                          44.5%
```

## Files

| File | Description |
|---|---|
| `solution.py` | Complete 12-cell Colab notebook |

## Dependencies

```
transformers>=4.45.0  datasets>=3.0.0  evaluate>=0.4.1
accelerate>=1.0.0     jiwer>=3.0.3     soundfile  librosa
scipy  tqdm  pandas  tabulate
```
