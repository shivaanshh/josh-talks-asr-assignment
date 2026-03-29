# Q2 — ASR Cleanup Pipeline

Post-processing pipeline for Hindi ASR output with two operations.

## How to run

```bash
python solution.py
```

No GPU, no data, no setup. Runs in under 5 seconds.

## Part A — Number Normalization

Converts Hindi number words to digits.

| Input | Output | Notes |
|---|---|---|
| `सुबह दस बज गया था` | `सुबह 10 बज गया था` | time reference |
| `एक हज़ार पाँच सौ रुपये` | `1500 रुपये` | compound number |
| `पचहत्तर परसेंट` | `75 परसेंट` | percentage |
| `नौ बजे हम पहुँचे` | `9 बजे हम पहुँचे` | clock time |
| `मैं सौ फीसद सहमत` | `मैं 100 फीसद सहमत` | percentage |
| `बस दो-चार बातें` | `बस दो-चार बातें` | **kept** — idiom |
| `एक बात बताओ` | `एक बात बताओ` | **kept** — discourse |
| `दो नम्बर का काम` | `दो नम्बर का काम` | **kept** — slang |

## Part B — English Word Detection

Tags English/loanword tokens in Hindi text.

**Three detection layers:**
1. Latin script — `interview`, `job` (100% precision)
2. Known Devanagari loanword lexicon — `प्रोजेक्ट`, `एरिया`, `मिस्टेक`
3. Phonological heuristics — ऑ vowel, pr-/tr-/st- consonant clusters

**Output format:** `[EN]word[/EN]`

**Guideline respected:** Nativised words (`स्कूल`, `बस`, `किलोमीटर`) are tagged but marked as _correct per guidelines_, not errors.
