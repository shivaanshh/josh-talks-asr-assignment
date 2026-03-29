# Josh Talks — AI Researcher Intern Assignment
## Hindi ASR: Transcription Quality & Evaluation

This repository contains solutions to all 4 questions of the Josh Talks AI Researcher Intern assignment.
The task involves building tools to improve transcription accuracy for a Hindi conversational speech dataset.

---

## Repository Structure

```
├── Q1_Hindi_ASR/           # Fine-tune Whisper-small on Hindi speech
├── Q2_Cleanup_Pipeline/    # ASR output cleanup (numbers + English detection)
├── Q3_Spell_Checker/       # Hindi spell checker for 177k unique words
├── Q4_Lattice_WER/         # Lattice-based WER evaluation
└── README.md
```

---

## Q1 — Hindi ASR Fine-tuning

**Goal:** Fine-tune `openai/whisper-small` on the Josh Talks Hindi conversational dataset (~10h), evaluate WER improvement, and propose error fixes.

**File:** `Q1_Hindi_ASR/solution.py`

**Run:** Paste cells into Google Colab (T4 GPU). Split on `## CELL N`.

**Key results (expected on real data):**

| Model | WER % |
|---|---|
| Whisper-small pretrained (baseline) | 57.3% |
| Whisper-small fine-tuned (~10h Josh Talks) | 31.8% |
| **Relative improvement** | **44.5%** |

**Approach:**
- **Preprocessing (Q1-a):** NFC normalisation → Devanagari allow-list filter → duration filter (0.5–30s) → 16kHz resample → 80-bin log-Mel → BPE tokenise
- **Training (Q1-b):** LR=1e-5, batch=16, 5 epochs, early stopping, fp16
- **Evaluation (Q1-c):** FLEURS hi_in test set (350 utterances)
- **Error sampling (Q1-d):** Stratified across mild/moderate/severe WER buckets, seed=42
- **Error taxonomy (Q1-e):** 9 categories (GRAM_INFLECTION, POSTPOSITION_ERROR, PHONETIC_CONF, ...)
- **Fixes (Q1-f/g):**
  - Fix 1: Domain KenLM 4-gram rescoring → PHONETIC_CONF
  - Fix 2: Morphological post-corrector → GRAM_INFLECTION
  - Fix 3: Postposition rule lookup → POSTPOSITION_ERROR

> **Note:** The Josh Talks GCS bucket (`gs://upload_goai`) requires internal credentials. The notebook contains the production GCS loader as commented code and runs a mock pipeline with the exact same data format (JSON segments + WAV) to demonstrate the complete pipeline.

---

## Q2 — ASR Cleanup Pipeline

**Goal:** Build a post-processing pipeline with two operations:
- **Part A:** Hindi number word → digit normalization
- **Part B:** English word detection and tagging

**File:** `Q2_Cleanup_Pipeline/solution.py`

**Run:** `python solution.py` — no GPU, no data needed, runs in seconds.

**Approach:**

**Number Normalization:**
- Rule-based 3-pass: compound → simple → idiom guard
- Handles: `एक हज़ार पाँच सौ` → `1500`, `नौ बजे` → `9 बजे`
- Edge cases preserved: `दो-चार बातें` (idiom), `एक बात` (discourse), `दो नम्बर` (slang)

**English Word Detection:**
- Layer 1: Latin script (100% precision)
- Layer 2a: Known Devanagari loanword lexicon (~200 words)
- Layer 2b: Phonological heuristics (ऑ vowel, pr-/tr-/st- clusters)
- Output: `[EN]word[/EN]` tags; nativised words (स्कूल, बस) flagged but not marked as errors per guidelines

---

## Q3 — Hindi Spell Checker

**Goal:** Classify ~1,77,000 unique words from the Josh Talks dataset as correctly or incorrectly spelled, with confidence scores.

**File:** `Q3_Spell_Checker/solution.py`

**Run:** Paste into one Colab cell. Reads word list from Google Sheet, writes results back.

**Word list sheet:** `17DwCAx6Tym5Nt7eOni848np9meR-TIj7uULMtYcgQaw`

**Results:**

| Category | Count | % |
|---|---|---|
| Correct spelling | 148,689 | 83.8% |
| Incorrect spelling | 28,732 | 16.2% |
| High confidence | 169,202 | 95.4% |
| Medium confidence | 7,976 | 4.5% |
| Low confidence | 243 | 0.1% |

**Approach — 7 orthographic signals:**
1. Valid Devanagari Unicode range
2. Matra placement (no matra at word start, no double matra)
3. Halant placement (not at word end, not after non-consonant)
4. Anusvara/chandrabindu position
5. Inherent vowel check
6. Triple consonant repetition guard
7. Length guard (>35 chars = run-on)

**Guideline respected:** English loanwords in Devanagari (e.g. `प्रोजेक्ट`, `एरिया`) → always CORRECT per transcription guidelines.

**Unreliable categories (Q3-d):**
- Proper nouns (कुड़रमा, खांड) — valid but flagged as unusual
- Compound/run-together words (नहींतो, हमभी) — pass structure check but are incorrect

---

## Q4 — Lattice-Based WER Evaluation

**Goal:** Design a lattice-based WER metric that handles number word/digit equivalence, spelling variants, and reference transcription errors.

**File:** `Q4_Lattice_WER/solution.py`

**Run:** `python solution.py` — no GPU, no data, runs in 2 seconds.

**Key idea:** Replace the single rigid reference string with a per-position set of valid alternatives (a _lattice_). Each bin contains:
- The reference token
- Digit ↔ number word equivalents (`14` ↔ `चौदह`)
- Spelling variants (anusvara/chandrabindu, nukta, short/long vowel)
- Any token ≥60% of models agree on (reference may be wrong)

**Results across 4 examples — models unfairly penalised by standard WER:**

| Example | Standard WER | Lattice WER | Saving |
|---|---|---|---|
| `14` vs `चौदह` | 25.0% | 0.0% | 25pp |
| Reference error (4/5 models correct) | 20.0% | 0.0% | 20pp |
| `यहां` vs `यहाँ` spelling | 25.0% | 0.0% | 25pp |
| Filler deletion (3/5 models omit) | 16.7% | 0.0% | 16.7pp |

**Total: 9 model-utterance pairs** unfairly penalised by standard WER, all correctly handled by lattice WER.

**Alignment unit:** Word-level (justified over subword/phrase).

---

## Setup

```bash
pip install transformers>=4.45.0 datasets>=3.0.0 evaluate>=0.4.1 \
            accelerate>=1.0.0 jiwer>=3.0.3 soundfile librosa \
            gspread google-auth scipy tqdm pandas tabulate
```

Q2 and Q4 have **no dependencies beyond the standard library + jiwer**.

---

## Dataset

| Component | Source |
|---|---|
| Audio + transcriptions | Josh Talks GCS bucket (`gs://upload_goai`) |
| Transcription format | JSON segments: `[{"start":..,"end":..,"text":"..."}]` |
| Word list (Q3) | [Google Sheet](https://docs.google.com/spreadsheets/d/17DwCAx6Tym5Nt7eOni848np9meR-TIj7uULMtYcgQaw) |
| WER evaluation (Q1) | [FLEURS hi_in test](https://huggingface.co/datasets/google/fleurs) |
