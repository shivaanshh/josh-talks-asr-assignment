# Q3 — Hindi Spell Checker

Classify ~1,77,000 unique words as correctly or incorrectly spelled, with confidence scores.

## How to run

Paste `solution.py` into a single Google Colab cell and run it.
It will:
1. Authenticate with Google (browser popup)
2. Read the word list from the Google Sheet
3. Classify all words (~2 seconds)
4. Write results to a new `Q3_Results` tab in the same sheet
5. Download a CSV backup automatically

## Results

| Label | Count | % |
|---|---|---|
| Correct spelling | **148,689** | 83.8% |
| Incorrect spelling | 28,732 | 16.2% |
| — High confidence | 169,202 | 95.4% |
| — Medium confidence | 7,976 | 4.5% |
| — Low confidence | 243 | 0.1% |

## Approach — 7 orthographic signals

| Signal | What it catches |
|---|---|
| Valid Devanagari range | Latin/symbol garbage |
| Matra placement | Matra at word start, double matra |
| Halant placement | Halant at word end, after non-consonant |
| Anusvara position | Anusvara at word start, consecutive anusvaras |
| Inherent vowel | All-consonant strings (phonologically impossible) |
| Triple consonant | `ककक` keyboard noise |
| Length guard | >35 chars = run-on word |

**Devanagari-transcribed English (e.g. `प्रोजेक्ट`, `एरिया`) → always CORRECT** per transcription guidelines.

## Unreliable categories (Q3-d)

1. **Proper nouns** — valid names (कुड़रमा, खांड) may get low confidence
2. **Compound/run-together words** — (नहींतो, हमभी) pass structure but are wrong

## Output columns

| Column | Values |
|---|---|
| Word | original word |
| Classification | `correct spelling` / `incorrect spelling` |
| Confidence | `high` / `medium` / `low` |
| Reason | one-line explanation |
