# Q4 — Lattice-Based WER Evaluation

A fairer WER metric that handles number word/digit equivalence, spelling variants, and reference transcription errors.

## How to run

```bash
python solution.py
```

No GPU, no data, runs in 2 seconds.

## The Problem with Standard WER

Standard WER compares a hypothesis against a single rigid reference.
This unfairly penalises valid alternatives:

- `14` vs `चौदह` — semantically identical, different surface form
- `यहां` vs `यहाँ` — same word, different anusvara/chandrabindu spelling
- Reference transcriber made an error — all models are penalised for being correct

## Solution: Transcription Lattice

Each alignment position gets a **bin** of valid alternatives instead of one word:

```
Spoken: "उसने चौदह किताबें खरीदीं"
Lattice:
  [0] उसने
  [1] चौदह | 14             ← digit equivalent added
  [2] किताबें | किताबे       ← spelling variant added
  [3] खरीदीं | खरीदी         ← spelling variant added
```

**Trust rule:** If ≥60% of models agree on a token that differs from the reference, the reference is likely wrong → that token is added to the bin.

## Results

| Example | Standard WER | Lattice WER | Improvement |
|---|---|---|---|
| `14` vs `चौदह` | 25.0% | **0.0%** | 25pp |
| Reference error (4/5 models correct) | 20.0% | **0.0%** | 20pp |
| `यहां` vs `यहाँ` spelling | 25.0% | **0.0%** | 25pp |
| Filler deletion (3/5 models omit) | 16.7% | **0.0%** | 16.7pp |

**9 model-utterance pairs** unfairly penalised by standard WER — all corrected by lattice WER.
Models that made genuine errors are still penalised correctly.

## Alignment Unit: Word

Word-level chosen because:
- Number equivalence works at word level (`चौदह` ↔ `14`)
- Hindi spelling variants are whole-word transformations
- Subword units fragment Devanagari unnaturally
- WER is conventionally word-level — stays comparable
