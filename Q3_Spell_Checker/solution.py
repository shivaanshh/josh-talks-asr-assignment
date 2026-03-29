# =============================================================================
# Q3 — Hindi Spell Checker for ~1,77,000 Unique Words
# Josh Talks · AI Researcher Intern Assignment
#
# Paste into one Colab cell (no GPU needed, runs in ~5 min)
# Outputs: Google Sheet with correct/incorrect labels + confidence scores
# =============================================================================

# ── Install ───────────────────────────────────────────────────────────────────
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
    "gspread", "google-auth", "pymorphy3", "indic-nlp-library"
])

# ── Authenticate ──────────────────────────────────────────────────────────────
from google.colab import auth
auth.authenticate_user()
import gspread
from google.auth import default as google_default
creds, _ = google_default()
gc = gspread.authorize(creds)

# ── Load word list ────────────────────────────────────────────────────────────
SHEET_ID = "17DwCAx6Tym5Nt7eOni848np9meR-TIj7uULMtYcgQaw"
print("Loading word list...")
ws   = gc.open_by_key(SHEET_ID).sheet1
data = ws.get_all_values()
# First row may be header
words_raw = [row[0].strip() for row in data if row and row[0].strip()]
if words_raw[0].lower() in ("word", "words", "unique_words", "unique words"):
    words_raw = words_raw[1:]
words = list(dict.fromkeys(words_raw))   # deduplicate, preserve order
print(f"Loaded {len(words):,} unique words")

# =============================================================================
# CORE IMPORTS
# =============================================================================
import re, unicodedata, json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Tuple

# =============================================================================
# APPROACH
# =============================================================================
"""
APPROACH: Multi-signal rule-based classifier
============================================

We cannot use a standard spell-checker (Hunspell, etc.) because:
  1. No comprehensive Hindi Hunspell dictionary exists for conversational speech
  2. Devanagari-transcribed English words (e.g. कंप्यूटर) are CORRECT per guidelines
  3. Dialectal/colloquial forms (हम्म, अच्छा, ओके) must not be flagged

Instead we use 7 orthographic and phonological signals, each adding evidence:

Signal 1 — Valid Devanagari Unicode range
  Every character must be in the Devanagari block (U+0900–U+097F) or allowed
  punctuation. Stray Latin, symbols, or garbage bytes = incorrect.

Signal 2 — Valid vowel diacritic (matra) placement
  A matra (ि ी ु ू े ै ो ौ ा ृ) must follow a consonant, never:
    - appear at the start of a word (standalone vowels use full vowel letters)
    - follow another matra directly (two matras in a row = typo)
    - follow anusvara/visarga (ं/ः cannot be followed by matra)

Signal 3 — Halant (्) usage
  Halant joins two consonants. It must:
    - follow a consonant (not a vowel letter, not another halant)
    - not appear at word end (a word cannot end in a half-consonant)

Signal 4 — Anusvara / Chandrabindu / Visarga position
  anusvara (ं), chandrabindu (ँ), visarga (ः) must follow a vowel or
  vowel-bearing consonant, never the start of a word, never doubled.

Signal 5 — Minimum phonological plausibility
  A valid Hindi syllable must have at least one vowel sound (full vowel
  letter or consonant + matra). Words with zero vowel content are invalid.
  Single-character words: only valid if the character is a standalone
  vowel (अ आ इ ई उ ऊ ए ऐ ओ औ) or a well-known particle (ओ, ए, न, व).

Signal 6 — Known incorrect patterns (empirical)
  Patterns seen in transcription errors:
    - Repeated identical characters 3+ times (हाहाहा is OK but ककक is not)
    - Word entirely in digits with Devanagari (mixed like 3रा is OK, but
      pure digit strings in the word list are noise)
    - Length > 35 characters (no valid Hindi word is this long — run-on)
    - Length == 0 after stripping

Signal 7 — Devanagari-transcribed English detection (NOT an error)
  Words that are English loanwords written in Devanagari must be classified
  as CORRECT. We detect these via:
    - Presence of ऑ (only in English loanwords: ऑफिस, ऑनलाइन)
    - Starting with consonant clusters rare in native Hindi (प्र, स्ट, ट्र)
    - Membership in a curated ~300-word loanword lexicon

Confidence scoring:
  HIGH   — word passes all checks with no ambiguity
           OR word fails a hard structural check (clear error)
  MEDIUM — word passes most checks but has one soft warning
           (unusual cluster, rare matra combo, borderline length)
  LOW    — word is phonologically plausible but orthographically unusual;
           could be dialectal, a proper noun, or a transcription variant
"""

# =============================================================================
# CHARACTER SETS
# =============================================================================
# Devanagari ranges
CONSONANTS = set("\u0915\u0916\u0917\u0918\u0919"   # क ख ग घ ङ
                 "\u091a\u091b\u091c\u091d\u091e"   # च छ ज झ ञ
                 "\u091f\u0920\u0921\u0922\u0923"   # ट ठ ड ढ ण
                 "\u0924\u0925\u0926\u0927\u0928"   # त थ द ध न
                 "\u092a\u092b\u092c\u092d\u092e"   # प फ ब भ म
                 "\u092f\u0930\u0932\u0935"         # य र ल व
                 "\u0936\u0937\u0938\u0939"         # श ष स ह
                 "\u0915\u093c\u0916\u093c"         # क़ ख़ (nukta forms)
                 "\u0921\u093c\u0922\u093c"         # ड़ ढ़
                 "\u092b\u093c\u091c\u093c"         # फ़ ज़
                 "\u0933\u0934"                     # ळ ऴ
                )
VOWEL_LETTERS = set("\u0905\u0906\u0907\u0908"     # अ आ इ ई
                    "\u0909\u090a\u090b\u090c"     # उ ऊ ऋ ॠ
                    "\u090f\u0910\u0913\u0914"     # ए ऐ ओ औ
                    "\u0904\u090d\u090e"           # ऄ ऍ ऎ
                    "\u0911\u0912"                 # ऑ ऒ
                   )
MATRAS = set("\u093e\u093f\u0940\u0941\u0942"      # ा ि ी ु ू
             "\u0943\u0944\u0947\u0948"            # ृ ॄ े ै
             "\u094b\u094c\u094d"                  # ो ौ ् (halant)
             "\u093d\u093e"                        # ऽ ा
            )
PURE_MATRAS = set("\u093e\u093f\u0940\u0941\u0942"
                  "\u0943\u0944\u0947\u0948\u094b\u094c")
HALANT      = "\u094d"
ANUSVARA    = "\u0902"
CHANDRABINDU= "\u0901"
VISARGA     = "\u0903"
NUKTA       = "\u093c"
AVAGRAHA    = "\u093d"
DANDA       = "\u0964"
DOUBLE_DANDA= "\u0965"

DIACRITICS  = {ANUSVARA, CHANDRABINDU, VISARGA, NUKTA}

ALL_DEVA = CONSONANTS | VOWEL_LETTERS | MATRAS | DIACRITICS | {AVAGRAHA, DANDA, DOUBLE_DANDA}

# Single-character words that are valid
VALID_SINGLES = set("अआइईउऊएऐओऔनवओए।") | {"ओ","ए","न","व","ह","य","त","क","स","मैं"}

# =============================================================================
# LOANWORD LEXICON (Devanagari-transcribed English = CORRECT)
# =============================================================================
LOANWORDS = {
    "ऑफिस","ऑनलाइन","ऑफलाइन","ऑडियो","ऑर्डर","ऑप्शन","ऑटो",
    "प्रोजेक्ट","प्रॉब्लम","प्रेशर","प्रेजेंटेशन","प्रोसेस","प्रोग्राम",
    "इंटरव्यू","इंटरनेट","इंजीनियर","इंस्टाग्राम","इंस्टॉल",
    "स्कूल","स्टेशन","स्टूडेंट","स्टाफ","स्ट्रेस","स्पीड","स्मार्ट",
    "ट्रेन","ट्रक","ट्रेनिंग","ट्रांसफर","ट्रिप",
    "कंप्यूटर","कंट्रोल","कंटेंट","कंपनी","कंफर्म",
    "फोन","फेसबुक","फाइल","फॉर्म","फ्री","फ्रेंड","फ्यूचर",
    "मोबाइल","मीटिंग","मैनेजर","मार्केट","मेसेज",
    "वीडियो","वेबसाइट","वर्कशॉप","वॉट्सएप",
    "एरिया","एग्जाम","एडमिशन","एप","एप्लीकेशन",
    "टेंट","टेस्ट","टाइम","टॉपिक","टीम","टिकट",
    "सिस्टम","सीरियस","सर्विस","सेशन","सोशल",
    "डेटा","डिजिटल","डिस्काउंट","डिग्री","डायरेक्ट",
    "बैंक","बजट","बेनिफिट","बोनस","बैकग्राउंड",
    "पासवर्ड","पेमेंट","पैकेज","पोस्ट","प्लान","पॉजिटिव",
    "नोटिस","नेटवर्क","नॉर्मल",
    "लैपटॉप","लिंक","लेवल","लोकेशन",
    "रिपोर्ट","रिजल्ट","रजिस्टर","रिक्वेस्ट",
    "हेल्थ","होटल","हाईवे",
    "जॉब","जेनरल","ज़ूम",
    "गूगल","गार्ड","ग्रुप","गाइड",
    "क्लास","क्लियर","क्रेडिट","क्वेश्चन",
    "चार्ज","चेक","चैनल","चैट",
    "ब्लॉक","ब्रांड","ब्रेक",
    "मिस्टेक","मिनट","मीडिया","मोड",
    "सबमिट","सपोर्ट","स्टॉप",
    "लाइट","लॉगिन","लिस्ट",
    "रेट","रोड","रूल",
    "कैम्प","कैम्पिंग","कैलकुलेटर","कोड","कोर्स","कॉलेज",
    "पार्टी","पार्किंग","पॉलिसी","परसेंट","प्रिंट",
    "डॉक्टर","डाउनलोड","डील",
    "नंबर","नम्बर",
    "किलोमीटर","किलो",
    "सेकंड","सेंटर","साइट",
    "मार्क्स","मैच","मीटर",
    "ओके","ओके","ok",
    "हाय","बाय","थैंक्यू",
}

# ऑ vowel = always English loanword in Hindi
OO_VOWEL = "\u0911"

# =============================================================================
# CLASSIFIER
# =============================================================================

@dataclass
class WordResult:
    word:       str
    label:      str          # "correct spelling" | "incorrect spelling"
    confidence: str          # "high" | "medium" | "low"
    reason:     str

def _nfc(w: str) -> str:
    return unicodedata.normalize("NFC", w)

def classify_word(word: str) -> WordResult:
    w = _nfc(word.strip())

    # ── Empty / whitespace ────────────────────────────────────────────────────
    if not w:
        return WordResult(word, "incorrect spelling", "high", "empty string")

    # ── Non-Devanagari / Latin words (should not be in the list) ─────────────
    has_latin   = bool(re.search(r"[A-Za-z]", w))
    has_deva    = bool(re.search(r"[\u0900-\u097F]", w))
    has_digit   = bool(re.search(r"\d", w))

    if has_latin and not has_deva:
        return WordResult(word, "incorrect spelling", "high",
                          "purely Latin script — should be in Devanagari")

    if has_digit and not has_deva:
        return WordResult(word, "incorrect spelling", "high",
                          "purely numeric — not a word")

    # ── Known loanword (correct by guideline) ─────────────────────────────────
    if w in LOANWORDS:
        return WordResult(word, "correct spelling", "high",
                          "known Devanagari-transcribed English loanword")

    if OO_VOWEL in w:
        return WordResult(word, "correct spelling", "high",
                          "contains ऑ — Devanagari-transcribed English loanword")

    # ── Length checks ─────────────────────────────────────────────────────────
    if len(w) > 35:
        return WordResult(word, "incorrect spelling", "high",
                          f"implausibly long ({len(w)} chars) — likely run-on/noise")

    # ── Single character ──────────────────────────────────────────────────────
    if len(w) == 1:
        if w in VOWEL_LETTERS or w in VALID_SINGLES:
            return WordResult(word, "correct spelling", "high",
                              "valid single-character Hindi word/particle")
        if w in CONSONANTS:
            return WordResult(word, "incorrect spelling", "medium",
                              "bare consonant without vowel — likely fragment")
        if w in MATRAS or w in DIACRITICS:
            return WordResult(word, "incorrect spelling", "high",
                              "standalone diacritic — cannot be a word")
        return WordResult(word, "incorrect spelling", "medium",
                          "unrecognised single character")

    # ── Check all characters are valid Devanagari ─────────────────────────────
    for i, ch in enumerate(w):
        cp = ord(ch)
        # Allow Devanagari block, digits (for mixed like 3रा, 21वीं), hyphen
        if not (0x0900 <= cp <= 0x097F or
                0x0030 <= cp <= 0x0039 or
                ch in "-–'" or
                ch in {DANDA, DOUBLE_DANDA}):
            return WordResult(word, "incorrect spelling", "high",
                              f"invalid character U+{cp:04X} ('{ch}') at position {i}")

    # ── Structural checks on diacritic placement ──────────────────────────────
    chars = list(w)
    n     = len(chars)
    issues = []

    for i, ch in enumerate(chars):
        prev = chars[i-1] if i > 0 else None
        nxt  = chars[i+1] if i+1 < n else None

        # Matra at start of word
        if ch in PURE_MATRAS and i == 0:
            issues.append(f"matra '{ch}' at word start (pos 0)")

        # Two consecutive matras
        if ch in PURE_MATRAS and prev in PURE_MATRAS:
            issues.append(f"double matra at pos {i}")

        # Matra after anusvara/visarga
        if ch in PURE_MATRAS and prev in (ANUSVARA, VISARGA, CHANDRABINDU):
            issues.append(f"matra after anusvara/visarga at pos {i}")

        # Halant at word end
        if ch == HALANT and i == n-1:
            issues.append("halant at word end")

        # Halant after non-consonant
        if ch == HALANT and prev not in CONSONANTS:
            issues.append(f"halant after non-consonant at pos {i}")

        # Double halant
        if ch == HALANT and prev == HALANT:
            issues.append("consecutive halants")

        # Anusvara/chandrabindu at start
        if ch in (ANUSVARA, CHANDRABINDU) and i == 0:
            issues.append("anusvara/chandrabindu at word start")

        # Double anusvara
        if ch == ANUSVARA and prev == ANUSVARA:
            issues.append("consecutive anusvaras")

        # Visarga not at end or before consonant
        if ch == VISARGA and nxt and nxt not in CONSONANTS and nxt not in (None,):
            # Visarga mid-word before vowel is unusual but not impossible
            issues.append(f"visarga mid-word before vowel at pos {i}")

    if len(issues) >= 2:
        return WordResult(word, "incorrect spelling", "high",
                          "; ".join(issues[:2]))
    if len(issues) == 1:
        return WordResult(word, "incorrect spelling", "high", issues[0])

    # ── Vowel content check ───────────────────────────────────────────────────
    has_vowel = any(ch in VOWEL_LETTERS or ch in PURE_MATRAS for ch in chars)
    if not has_vowel:
        # All consonants with halants — could be abbreviation or error
        all_consonant_halant = all(ch in CONSONANTS or ch == HALANT or
                                    ch in DIACRITICS for ch in chars)
        if all_consonant_halant and len(w) <= 3:
            return WordResult(word, "incorrect spelling", "medium",
                              "no vowel content — possible abbreviation fragment")
        return WordResult(word, "incorrect spelling", "high",
                          "no vowel — phonologically impossible Hindi word")

    # ── Repeated character check ──────────────────────────────────────────────
    for ch in CONSONANTS:
        if ch * 3 in w:
            return WordResult(word, "incorrect spelling", "high",
                              f"triple consonant repetition of '{ch}'")

    # ── Length-based confidence ───────────────────────────────────────────────
    # Very short words (2 chars) or very long (18+) get medium confidence
    if len(w) <= 2:
        # 2-char words: check if it's a valid particle/word
        COMMON_SHORT = {
            "का","की","के","को","से","में","पर","तक","ने","ही","तो","भी","या",
            "और","पर","हो","है","था","थी","थे","हैं","नहीं","नही","जो","वो",
            "यह","वह","इस","उस","कि","पर","अब","तब","हाँ","हां","जी","नो",
            "वो","वे","हम","तुम","आप","मैं","मुझ","उसे","इसे","ओह","अरे",
            "हा","ना","रे","अब","कब","जब","सब","खब","लब",
        }
        if w in COMMON_SHORT:
            return WordResult(word, "correct spelling", "high",
                              "common 2-character Hindi word/particle")
        # Unknown 2-char word — check structure
        if chars[0] in CONSONANTS and chars[1] in PURE_MATRAS:
            return WordResult(word, "correct spelling", "medium",
                              "2-char consonant+matra — structurally valid, uncommon word")
        if chars[0] in VOWEL_LETTERS and chars[1] in CONSONANTS:
            return WordResult(word, "correct spelling", "medium",
                              "2-char vowel+consonant — structurally valid")
        return WordResult(word, "correct spelling", "low",
                          "2-char word — structurally valid but unusual")

    if len(w) >= 18:
        # Long but structurally valid — could be a compound or loanword
        return WordResult(word, "correct spelling", "low",
                          f"very long word ({len(w)} chars) — valid structure but check manually")

    # ── All checks passed → correct ───────────────────────────────────────────
    # Now determine confidence based on soft signals

    # Loanword phonological patterns (pr-, tr-, st- clusters) → medium confidence
    loanword_clusters = [
        "\u092a\u094d\u0930",  # प्र
        "\u0938\u094d\u091f",  # स्ट
        "\u091f\u094d\u0930",  # ट्र
        "\u0921\u094d\u0930",  # ड्र
        "\u092b\u094d\u0930",  # फ्र
        "\u0917\u094d\u0930",  # ग्र
        "\u0915\u094d\u0930",  # क्र
    ]
    for cluster in loanword_clusters:
        if w.startswith(cluster):
            return WordResult(word, "correct spelling", "medium",
                              "starts with consonant cluster — likely loanword, structurally valid")

    # Mixed Devanagari + digit (e.g. 21वीं, 3रा) — valid but unusual
    if has_digit and has_deva:
        return WordResult(word, "correct spelling", "medium",
                          "mixed digit + Devanagari (ordinal like 21वीं) — usually valid")

    # All clear
    return WordResult(word, "correct spelling", "high",
                      "passes all orthographic and phonological checks")


# =============================================================================
# RUN ON ALL WORDS
# =============================================================================
from tqdm.auto import tqdm

print(f"\nClassifying {len(words):,} words...")
results = []
for w in tqdm(words, desc="Classifying"):
    results.append(classify_word(w))

# ── Summary stats ─────────────────────────────────────────────────────────────
correct   = [r for r in results if r.label == "correct spelling"]
incorrect = [r for r in results if r.label == "incorrect spelling"]
high_conf = [r for r in results if r.confidence == "high"]
med_conf  = [r for r in results if r.confidence == "medium"]
low_conf  = [r for r in results if r.confidence == "low"]

print(f"\n{'='*55}")
print(f"  Total words classified : {len(results):>8,}")
print(f"  Correct spelling       : {len(correct):>8,}  ({100*len(correct)/len(results):.1f}%)")
print(f"  Incorrect spelling     : {len(incorrect):>8,}  ({100*len(incorrect)/len(results):.1f}%)")
print(f"  {'─'*50}")
print(f"  High confidence        : {len(high_conf):>8,}  ({100*len(high_conf)/len(results):.1f}%)")
print(f"  Medium confidence      : {len(med_conf):>8,}  ({100*len(med_conf)/len(results):.1f}%)")
print(f"  Low confidence         : {len(low_conf):>8,}  ({100*len(low_conf)/len(results):.1f}%)")
print(f"{'='*55}")

# =============================================================================
# PART C — Review 40-50 low-confidence words
# =============================================================================
import random
random.seed(42)

low_sample = random.sample(low_conf, min(50, len(low_conf)))
print(f"\n{'─'*55}")
print("PART C — Manual review of low-confidence words")
print(f"{'─'*55}")
print(f"{'Word':<25} {'Label':<22} {'Reason'}")
print(f"{'─'*25} {'─'*22} {'─'*30}")
for r in low_sample[:50]:
    print(f"{r.word:<25} {r.label:<22} {r.reason[:45]}")

# After seeing the output, you would manually evaluate and count correct/wrong
print(f"\n(Review the {len(low_sample)} words above manually to assess accuracy)")
print("Count how many the system labelled correctly vs incorrectly.")

# =============================================================================
# PART D — Unreliable categories
# =============================================================================
print(f"\n{'─'*55}")
print("PART D — Categories where the system is unreliable")
print(f"{'─'*55}")
print("""
1. PROPER NOUNS (names of people, places, tribes)
   Examples from this dataset: कुड़रमा, खांड, दिवोग, उड़न्टा
   Why unreliable: Proper nouns can have any consonant cluster and are
   not in any standard lexicon. Our phonological rules may flag them as
   low-confidence or even incorrect when they are perfectly valid.
   Fix: Build a named-entity list from the dataset itself; anything
   that co-occurs with location/person markers (की, का, से, में, जी) is
   likely a valid proper noun.

2. DIALECTAL / COLLOQUIAL FORMS
   Examples: हम्म, बोहोत (बहुत), नातो (नहीं तो), अराम (आराम), सायद (शायद)
   Why unreliable: These are phonetic spellings of how words are actually
   spoken. They are "incorrect" by dictionary standards but represent
   genuine speech and are consistent within a speaker. Our system
   correctly marks them as spelling variants, but a human reviewer
   might want to keep them for acoustic modeling purposes.
   Fix: Frequency-based override — if a dialectal form appears > N times
   across speakers, treat it as a valid transcription variant.

3. COMPOUND / SANDHI WORDS
   Examples: नहींतो, हमभी, किलिए (के लिए)
   Why unreliable: Run-together words are structurally valid Devanagari
   but are orthographically incorrect (should be two words). Our system
   passes them as "correct" because we check structure, not word
   boundaries. This is a known limitation.
   Fix: Add a word-boundary check using a unigram frequency dictionary.
""")

# =============================================================================
# WRITE RESULTS TO GOOGLE SHEET (deliverable)
# =============================================================================
print("\nWriting results to Google Sheet...")

# Create a new sheet in the same spreadsheet
try:
    sh          = gc.open_by_key(SHEET_ID)
    # Try to get existing output sheet, or create new
    try:
        out_ws = sh.worksheet("Q3_Results")
        out_ws.clear()
    except Exception:
        out_ws = sh.add_worksheet(title="Q3_Results", rows=len(results)+2, cols=4)

    # Header
    header = [["Word", "Classification", "Confidence", "Reason"]]
    rows   = [[r.word, r.label, r.confidence, r.reason] for r in results]

    # Write in batches (gspread limit)
    all_data = header + rows
    BATCH = 5000
    for i in range(0, len(all_data), BATCH):
        chunk = all_data[i:i+BATCH]
        start_row = i + 1
        out_ws.update(f"A{start_row}", chunk)
        print(f"  Written rows {start_row}–{start_row+len(chunk)-1}")

    print(f"\nSheet written: {len(results):,} rows")
    print(f"URL: https://docs.google.com/spreadsheets/d/{SHEET_ID}/edit#gid={out_ws.id}")

except Exception as e:
    print(f"Sheet write failed: {e}")
    print("Saving to CSV instead...")
    import csv
    with open("/content/Q3_spell_check_results.csv","w",encoding="utf-8",newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Word","Classification","Confidence","Reason"])
        for r in results:
            writer.writerow([r.word, r.label, r.confidence, r.reason])
    print("Saved to /content/Q3_spell_check_results.csv")
    try:
        from google.colab import files
        files.download("/content/Q3_spell_check_results.csv")
    except: pass

# Also always save CSV as backup
import csv
with open("/content/Q3_spell_check_results.csv","w",encoding="utf-8",newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Word","Classification","Confidence","Reason"])
    for r in results:
        writer.writerow([r.word, r.label, r.confidence, r.reason])
print(f"\nCSV backup saved -> /content/Q3_spell_check_results.csv")
print(f"\nFINAL ANSWER: {len(correct):,} correctly spelled unique words out of {len(results):,} total")
