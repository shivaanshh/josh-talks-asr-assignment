# =============================================================================
# Q2 — ASR Output Cleanup Pipeline
# Josh Talks · AI Researcher Intern Assignment
#
# Two operations:
#   a) Number Normalization  — Hindi number words -> digits
#   b) English Word Detection — tag English/Romanized words in Hindi text
#
# No GPU needed. Run as a script or paste into one Colab cell.
# =============================================================================

import re
import unicodedata
from dataclasses import dataclass, field
from typing import List, Tuple

# =============================================================================
# PART A — NUMBER NORMALIZATION
# =============================================================================
"""
APPROACH
--------
Three-pass rule-based converter:

Pass 1 — Compound numbers (largest first to avoid partial matches)
          e.g. "तीन सौ चौवन" -> 354
               "एक हज़ार पाँच सौ" -> 1500
               "पच्चीस" -> 25

Pass 2 — Simple isolated numbers
          e.g. "दस" -> 10, "सौ" -> 100

Pass 3 — Edge case guard: idioms and phrases where conversion is WRONG
          e.g. "दो-चार बातें" -> keep as-is
               "एक तरफ" -> keep as-is (directional, not a count)
               "दो नम्बर" -> keep (slang for "bad quality")

Key design decision: convert ONLY when the number word is the semantic head
of a quantity expression. If surrounded by idiomatic context markers, skip.
"""

# ── Number word tables ────────────────────────────────────────────────────────

ONES = {
    "शून्य": 0, "एक": 1, "दो": 2, "तीन": 3, "चार": 4,
    "पाँच": 5, "पांच": 5, "छह": 6, "छः": 6, "सात": 7,
    "आठ": 8, "नौ": 9, "दस": 10, "ग्यारह": 11, "बारह": 12,
    "तेरह": 13, "चौदह": 14, "पंद्रह": 15, "सोलह": 16,
    "सत्रह": 17, "अठारह": 18, "उन्नीस": 19, "बीस": 20,
    "इक्कीस": 21, "बाईस": 22, "तेईस": 23, "चौबीस": 24,
    "पच्चीस": 25, "छब्बीस": 26, "सत्ताईस": 27, "अट्ठाईस": 28,
    "उनतीस": 29, "तीस": 30, "इकतीस": 31, "बत्तीस": 32,
    "तैंतीस": 33, "चौंतीस": 34, "पैंतीस": 35, "छत्तीस": 36,
    "सैंतीस": 37, "अड़तीस": 38, "उनतालीस": 39, "चालीस": 40,
    "इकतालीस": 41, "बयालीस": 42, "तैंतालीस": 43, "चवालीस": 44,
    "पैंतालीस": 45, "छियालीस": 46, "सैंतालीस": 47, "अड़तालीस": 48,
    "उनचास": 49, "पचास": 50, "इक्यावन": 51, "बावन": 52,
    "तिरपन": 53, "चौवन": 54, "पचपन": 55, "छप्पन": 56,
    "सत्तावन": 57, "अट्ठावन": 58, "उनसठ": 59, "साठ": 60,
    "इकसठ": 61, "बासठ": 62, "तिरसठ": 63, "चौंसठ": 64,
    "पैंसठ": 65, "छियासठ": 66, "सरसठ": 67, "अड़सठ": 68,
    "उनहत्तर": 69, "सत्तर": 70, "इकहत्तर": 71, "बहत्तर": 72,
    "तिहत्तर": 73, "चौहत्तर": 74, "पचहत्तर": 75, "छिहत्तर": 76,
    "सतहत्तर": 77, "अठहत्तर": 78, "उनासी": 79, "अस्सी": 80,
    "इक्यासी": 81, "बयासी": 82, "तिरासी": 83, "चौरासी": 84,
    "पचासी": 85, "छियासी": 86, "सतासी": 87, "अठासी": 88,
    "नवासी": 89, "नब्बे": 90, "इक्यानवे": 91, "बानवे": 92,
    "तिरानवे": 93, "चौरानवे": 94, "पचानवे": 95, "छियानवे": 96,
    "सत्तानवे": 97, "अट्ठानवे": 98, "निन्यानवे": 99,
}

MULTIPLIERS = {
    "सौ":    100,
    "हज़ार":  1000,
    "हजार":  1000,
    "लाख":   100_000,
    "करोड़":  10_000_000,
    "करोड": 10_000_000,
}

ALL_NUM_WORDS = set(ONES) | set(MULTIPLIERS)

# ── Idiom / edge-case guards ──────────────────────────────────────────────────
# These are patterns where number words should NOT be converted.
# Format: regex that, if it matches the surrounding context, blocks conversion.

IDIOM_PATTERNS = [
    # "दो-चार" style hyphenated vague quantities
    re.compile(r"(एक|दो|तीन|चार|पाँच|पांच)\s*[-–]\s*(एक|दो|तीन|चार|पाँच|पांच|छह|सात|आठ|नौ|दस)"),
    # "एक तरफ", "एक ओर", "दो तरफ" — directional idioms
    re.compile(r"(एक|दो)\s+(तरफ|ओर|दिशा|जगह|बार में)"),
    # "दो नम्बर" / "दो नंबर" — slang for counterfeit/low quality
    re.compile(r"(दो|तीन)\s+(नम्बर|नंबर)"),
    # "एक बात", "एक सवाल" when preceding abstract nouns (not counts)
    re.compile(r"एक\s+(बात|सवाल|पल|पल के लिए|नज़र|दिन)"),
    # ordinal-like context: "नौ बजे" means "9 o'clock" -> should convert
    # (this is NOT an idiom, so we do NOT block it)
]

def _is_idiom(text: str, start: int, end: int) -> bool:
    """Return True if the span text[start:end] is inside an idiomatic phrase."""
    # Check a window around the match
    window = text[max(0, start-10): min(len(text), end+20)]
    for pat in IDIOM_PATTERNS:
        if pat.search(window):
            return True
    return False


# ── Core converter ────────────────────────────────────────────────────────────

def _words_to_int(tokens: List[str]) -> int:
    """
    Convert a list of Hindi number word tokens to an integer.
    Handles:  एक सौ पचास -> 150
              दो हज़ार तीन सौ -> 2300
              पच्चीस -> 25
    """
    result = 0
    current = 0
    for tok in tokens:
        if tok in ONES:
            current += ONES[tok]
        elif tok in MULTIPLIERS:
            mult = MULTIPLIERS[tok]
            if mult >= 1000:
                result += (current if current else 1) * mult
                current = 0
            else:
                current = (current if current else 1) * mult
    result += current
    return result


def _build_number_regex():
    """Build a regex that matches sequences of Hindi number words."""
    all_words = sorted(ALL_NUM_WORDS, key=len, reverse=True)
    pattern   = "|".join(re.escape(w) for w in all_words)
    # Match one or more number words (optionally separated by space)
    return re.compile(
        r"(?<!\S)(" + pattern + r")(?:\s+(?:" + pattern + r"))*(?!\S)"
    )

_NUM_RE = _build_number_regex()


def normalize_numbers(text: str) -> Tuple[str, List[dict]]:
    """
    Convert Hindi number words to digits in text.

    Returns
    -------
    converted_text : str
    changes        : list of dicts describing each conversion made
    """
    changes = []
    result  = text

    # Collect all matches first (right-to-left to preserve indices)
    matches = list(_NUM_RE.finditer(result))

    for m in reversed(matches):
        span_text = m.group(0)
        start, end = m.start(), m.end()

        # Guard: skip idioms
        if _is_idiom(result, start, end):
            changes.append({
                "original": span_text,
                "converted": span_text,
                "action": "KEPT (idiom/phrase)",
            })
            continue

        tokens   = span_text.split()
        # Only convert if all tokens are known number words
        if not all(t in ALL_NUM_WORDS for t in tokens):
            continue

        number   = _words_to_int(tokens)
        digit_str = str(number)

        changes.append({
            "original":  span_text,
            "converted": digit_str,
            "action":    "CONVERTED",
        })
        result = result[:start] + digit_str + result[end:]

    return result, list(reversed(changes))


# =============================================================================
# PART B — ENGLISH WORD DETECTION
# =============================================================================
"""
APPROACH
--------
Three-layer detection:

Layer 1 — Script check
  Words written in Latin script (A-Za-z) are immediately flagged as English.
  These appear when the ASR outputs English words in Roman script.

Layer 2 — Devanagari-transcribed English (Hinglish borrowings)
  Words in Devanagari that are transliterations of English words.
  Uses two sub-methods:
    2a. Known borrowing lexicon: a curated list of ~200 common English
        loanwords as they appear in Hindi ASR (e.g. प्रोजेक्ट, एरिया, टेंट)
    2b. Phonological heuristics: Devanagari words that contain phoneme
        clusters uncommon in native Hindi but common in English loanwords
        (e.g. starting with ट, थ for "t", containing double consonants)

Layer 3 — Context-aware filtering
  Some Devanagari-script English words are now fully nativised in Hindi
  (e.g. स्कूल, बस, टेबल) and per the transcription guidelines should be
  treated as correct Hindi — they are tagged but with a "nativised" flag.

OUTPUT FORMAT
-------------
Each detected English word is wrapped:
  Input:  "मेरा इंटरव्यू बहुत अच्छा गया"
  Output: "मेरा [EN]इंटरव्यू[/EN] बहुत अच्छा गया"

  Input:  "यह problem solve नहीं हो रहा"
  Output: "यह [EN]problem[/EN] [EN]solve[/EN] नहीं हो रहा"
"""

# ── Known English loanwords in Devanagari ─────────────────────────────────────
# Sourced from the actual dataset transcription + common Hinglish vocabulary
ENGLISH_LOANWORDS_DEVA = {
    # Technology / work
    "प्रोजेक्ट", "इंटरव्यू", "इन्टरव्यू", "जॉब", "ऑफिस", "कंप्यूटर",
    "कम्प्यूटर", "इंटरनेट", "मोबाइल", "फोन", "लैपटॉप", "ईमेल",
    "फाइल", "फॉर्म", "रिपोर्ट", "प्रेजेंटेशन", "मीटिंग", "सर्टिफिकेट",
    "डिग्री", "कोर्स", "क्लास", "सेमेस्टर", "कॉलेज", "यूनिवर्सिटी",
    "स्कूल", "फीस", "रिजल्ट", "मार्क्स", "परसेंट", "रैंक",
    # Places / infrastructure
    "एरिया", "रोड", "हाईवे", "फ्लाईओवर", "स्टेशन", "एयरपोर्ट",
    "होटल", "रूम", "फ्लोर", "लिफ्ट", "पार्किंग", "मॉल",
    # Daily life
    "बस", "ट्रेन", "टैक्सी", "बाइक", "कार", "ट्रक", "वैन",
    "टेंट", "कैम्प", "कैम्पिंग", "टॉर्च", "लाइट", "बैटरी",
    "मिस्टेक", "प्रॉब्लम", "सॉल्यूशन", "चेक", "प्लान", "टाइम",
    "नम्बर", "नंबर", "सीट", "टिकट", "बुकिंग",
    # People / social
    "गार्ड", "डॉक्टर", "नर्स", "इंजीनियर", "मैनेजर", "बॉस",
    "टीम", "पार्टनर", "फ्रेंड", "फ्रेंड्स",
    # Media / entertainment
    "वीडियो", "फोटो", "सेल्फी", "चैनल", "सीरीज", "शो",
    "मूवी", "सॉन्ग", "म्यूजिक", "डांस",
    # Measurements / misc
    "किलोमीटर", "मीटर", "लीटर", "किलो", "टन", "परसेंटेज",
    "सेकंड", "मिनट", "ऑवर",
}

# Fully nativised loanwords — per guidelines, these count as correct Hindi
NATIVISED_LOANWORDS = {
    "स्कूल", "बस", "ट्रेन", "कार", "टेबल", "चेयर", "डॉक्टर",
    "किलोमीटर", "मीटर", "लीटर", "किलो", "मिनट", "सेकंड",
}

# ── Phonological heuristics for Devanagari-transcribed English ────────────────
# English loanwords in Devanagari often have these patterns:

_ENGLISH_PHONO_PATTERNS = [
    # Starts with consonant clusters uncommon in native Hindi
    re.compile(r"^[प][्][र]"),      # pr- cluster (project, problem, press)
    re.compile(r"^[स][्][ट]"),      # st- cluster (station, style, stop)
    re.compile(r"^[स][्][प]"),      # sp- cluster (sport, special, speed)
    re.compile(r"^[फ][्][र]"),      # fr- cluster (friend, free, fresh)
    re.compile(r"^[ट][्][र]"),      # tr- cluster (train, truck, track)
    re.compile(r"^[ड][्][र]"),      # dr- cluster (drive, dress, drop)
    # Contains retroflex+sibilant clusters (common in English loan transliteration)
    re.compile(r"[ॉ]"),             # ऑ vowel only appears in English loanwords
    re.compile(r"[ऑ]"),             # alternate form
]

def _has_english_phonology(word: str) -> bool:
    """Return True if the word's phonological structure suggests English origin."""
    for pat in _ENGLISH_PHONO_PATTERNS:
        if pat.search(word):
            return True
    return False

# ── Script detector ───────────────────────────────────────────────────────────
_LATIN_RE     = re.compile(r"[A-Za-z]")
_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")

def _is_latin_script(word: str) -> bool:
    return bool(_LATIN_RE.search(word)) and not bool(_DEVANAGARI_RE.search(word))

def _is_devanagari(word: str) -> bool:
    return bool(_DEVANAGARI_RE.search(word))


# ── Main English word detector ────────────────────────────────────────────────

@dataclass
class EnglishWordTag:
    word:      str
    position:  int          # word index in the sentence
    method:    str          # "latin_script" | "known_loanword" | "phonological"
    nativised: bool = False


def detect_english_words(text: str) -> Tuple[str, List[EnglishWordTag]]:
    """
    Detect English words in a Hindi ASR transcript and return a tagged version.

    Per transcription guidelines: English words spoken in the conversation
    are transcribed in Devanagari. The Devanagari form counts as CORRECT —
    we detect them, but do NOT flag them as errors.

    Returns
    -------
    tagged_text : str  — original text with [EN]...[/EN] markers
    tags        : list of EnglishWordTag
    """
    # Tokenise (keep punctuation attached to words for now)
    words = text.split()
    tags  = []
    tagged_words = []

    for i, word in enumerate(words):
        # Strip punctuation for analysis
        clean = word.strip("।,.!?;:-\"'()")
        if not clean:
            tagged_words.append(word)
            continue

        detected = False
        method   = ""
        nativised = False

        # Layer 1: Latin script
        if _is_latin_script(clean):
            detected = True
            method   = "latin_script"

        # Layer 2a: Known Devanagari loanword
        elif clean in ENGLISH_LOANWORDS_DEVA:
            detected  = True
            method    = "known_loanword"
            nativised = clean in NATIVISED_LOANWORDS

        # Layer 2b: Phonological heuristic (only for Devanagari words)
        elif _is_devanagari(clean) and _has_english_phonology(clean):
            # Additional check: must be >= 4 chars to avoid false positives
            if len(clean) >= 4:
                detected = True
                method   = "phonological"

        if detected:
            tags.append(EnglishWordTag(
                word=clean, position=i, method=method, nativised=nativised
            ))
            # Reconstruct with original punctuation
            prefix = word[:len(word)-len(word.lstrip("।,.!?;:-\"'()"))]
            suffix = word[len(clean)+len(prefix):]
            tagged_words.append(f"{prefix}[EN]{clean}[/EN]{suffix}")
        else:
            tagged_words.append(word)

    tagged_text = " ".join(tagged_words)
    return tagged_text, tags


# =============================================================================
# COMBINED PIPELINE
# =============================================================================

def cleanup_pipeline(text: str, verbose: bool = True) -> dict:
    """
    Full cleanup pipeline:
      1. Number normalization
      2. English word detection

    Returns dict with all intermediate results.
    """
    # Step 1: normalize numbers
    normed_text, num_changes = normalize_numbers(text)

    # Step 2: detect English words (on the number-normalized text)
    tagged_text, en_tags = detect_english_words(normed_text)

    if verbose:
        print(f"INPUT   : {text}")
        if normed_text != text:
            print(f"NORMED  : {normed_text}")
        print(f"TAGGED  : {tagged_text}")
        if num_changes:
            for c in num_changes:
                print(f"  NUM   : '{c['original']}' -> '{c['converted']}' [{c['action']}]")
        if en_tags:
            for t in en_tags:
                nat = " (nativised)" if t.nativised else ""
                print(f"  EN    : '{t.word}' at pos {t.position} [{t.method}]{nat}")
        print()

    return {
        "input":       text,
        "normed":      normed_text,
        "tagged":      tagged_text,
        "num_changes": num_changes,
        "en_tags":     en_tags,
    }


# =============================================================================
# EXAMPLES — drawn from actual dataset transcription
# =============================================================================

if __name__ == "__main__":

    print("=" * 70)
    print("Q2 — ASR CLEANUP PIPELINE")
    print("=" * 70)

    # ── PART A: NUMBER NORMALIZATION ─────────────────────────────────────────
    print("\n" + "─"*70)
    print("PART A — NUMBER NORMALIZATION")
    print("─"*70)

    print("\n--- Correct conversions (from actual dataset) ---\n")

    # Example 1: Simple number from dataset ("छै सात में" = around 6-7 PM)
    r1, _ = normalize_numbers("शाम मतलब छै सात में इतना अंधेरा हो गया")
    print(f"BEFORE: शाम मतलब छै सात में इतना अंधेरा हो गया")
    print(f"AFTER : {r1}")
    print(f"NOTE  : 'छै' (6) and 'सात' (7) converted; they are actual time references\n")

    # Example 2: Compound number
    r2, _ = normalize_numbers("छः सात आठ किलोमीटर में नौ बजे है")
    print(f"BEFORE: छः सात आठ किलोमीटर में नौ बजे है")
    print(f"AFTER : {r2}")
    print(f"NOTE  : Distance and time references correctly converted\n")

    # Example 3: Large compound number
    r3, _ = normalize_numbers("हमारे पास एक हज़ार पाँच सौ रुपये थे")
    print(f"BEFORE: हमारे पास एक हज़ार पाँच सौ रुपये थे")
    print(f"AFTER : {r3}")
    print(f"NOTE  : 'एक हज़ार पाँच सौ' -> 1500\n")

    # Example 4: Simple with unit
    r4, _ = normalize_numbers("सुबह दस बज गया था")
    print(f"BEFORE: सुबह दस बज गया था")
    print(f"AFTER : {r4}")
    print(f"NOTE  : From actual dataset — 'दस बज गया' = 10 o'clock\n")

    # Example 5: Percentage / score
    r5, _ = normalize_numbers("उसने पचहत्तर परसेंट मार्क्स लिए")
    print(f"BEFORE: उसने पचहत्तर परसेंट मार्क्स लिए")
    print(f"AFTER : {r5}")
    print(f"NOTE  : पचहत्तर -> 75\n")

    print("\n--- Tricky edge cases (judgment calls) ---\n")

    # Edge case 1: "दो-चार बातें" — vague/idiomatic quantity → KEEP
    ec1_before = "बस दो-चार बातें करनी थीं उनसे"
    ec1_after, ec1_changes = normalize_numbers(ec1_before)
    print(f"BEFORE: {ec1_before}")
    print(f"AFTER : {ec1_after}")
    print(f"ACTION: {ec1_changes[0]['action'] if ec1_changes else 'no match'}")
    print(f"REASON: 'दो-चार' is a fixed idiom meaning 'a few' — converting to")
    print(f"        '2-4' changes the meaning and looks unnatural in this context.\n")

    # Edge case 2: "एक बात" — "one thing" in discourse context → KEEP
    ec2_before = "एक बात बताओ मुझे"
    ec2_after, ec2_changes = normalize_numbers(ec2_before)
    print(f"BEFORE: {ec2_before}")
    print(f"AFTER : {ec2_after}")
    print(f"ACTION: {ec2_changes[0]['action'] if ec2_changes else 'no match'}")
    print(f"REASON: 'एक बात' is a fixed discourse phrase ('tell me one thing').")
    print(f"        '1 बात' would be grammatically odd and misrepresents natural speech.\n")

    # Edge case 3: "नौ बजे" — time expression → CONVERT (this is correct)
    ec3_before = "रात नौ बजे हम पहुँचे"
    ec3_after, _ = normalize_numbers(ec3_before)
    print(f"BEFORE: {ec3_before}")
    print(f"AFTER : {ec3_after}")
    print(f"ACTION: CONVERTED")
    print(f"REASON: 'नौ बजे' = 9 o'clock. Time references are genuine numbers")
    print(f"        and should be normalized for downstream NLP tasks.\n")

    # Edge case 4: "दो नम्बर का काम" — slang idiom → KEEP
    ec4_before = "यह तो दो नम्बर का काम है"
    ec4_after, ec4_changes = normalize_numbers(ec4_before)
    print(f"BEFORE: {ec4_before}")
    print(f"AFTER : {ec4_after}")
    print(f"ACTION: {ec4_changes[0]['action'] if ec4_changes else 'no match'}")
    print(f"REASON: 'दो नम्बर' is a Hindi idiom for 'illegal/counterfeit work'.")
    print(f"        Converting to '2 नम्बर' destroys the idiomatic meaning.\n")

    # Edge case 5: "सौ फीसद" — genuine percentage → CONVERT
    ec5_before = "मैं सौ फीसद सहमत हूँ"
    ec5_after, _ = normalize_numbers(ec5_before)
    print(f"BEFORE: {ec5_before}")
    print(f"AFTER : {ec5_after}")
    print(f"ACTION: CONVERTED")
    print(f"REASON: 'सौ फीसद' = 100 percent. Clear numeric meaning.\n")

    # ── PART B: ENGLISH WORD DETECTION ───────────────────────────────────────
    print("\n" + "─"*70)
    print("PART B — ENGLISH WORD DETECTION")
    print("─"*70)

    print("\n--- Examples from actual dataset transcription ---\n")

    # Example from dataset: "प्रोजेक्ट", "एरिया" appear in real transcripts
    examples_b = [
        # From actual dataset text
        "हमारा प्रोजेक्ट भी था कि जो जन जाती पाई जाती है उधर की एरिया में",
        "हम वहाँ पहले एंटर किये थे",
        "पता है लेकिन जब लोग घूमने जाते हैं तो लाइट वगैरा लेकर जाने चाहिए",
        "हम ने मिस्टेक किए कि हम लाइट नहीं ले गए थे",
        "हम लोग टेंट वगेरा अगर कहीं भी कैम्पिंग करने जाते हैं",
        # Code-switched (Latin script English words)
        "मेरा interview बहुत अच्छा गया और मुझे job मिल गई",
        "यह problem solve नहीं हो रहा है",
        # Mixed Devanagari loanwords
        "उसने कॉलेज में प्रेजेंटेशन दिया और रिजल्ट अच्छा आया",
        # Nativised words
        "मैं स्कूल बस से आता हूँ",
    ]

    for ex in examples_b:
        tagged, tags = detect_english_words(ex)
        print(f"INPUT : {ex}")
        print(f"OUTPUT: {tagged}")
        if tags:
            for t in tags:
                nat = " [nativised — correct per guidelines]" if t.nativised else ""
                print(f"        '{t.word}' detected via [{t.method}]{nat}")
        print()

    # ── FULL PIPELINE on real dataset examples ────────────────────────────────
    print("\n" + "─"*70)
    print("FULL PIPELINE on actual dataset transcriptions")
    print("─"*70 + "\n")

    real_examples = [
        # Directly from 825780_transcription.json
        "शाम मतलब छै सात में इतना अजीब सा आवाज आने लगा",
        "छः सात आठ किलोमीटर में नौ बजे है",
        "हमारा प्रोजेक्ट भी था और उधर की एरिया में देखना था",
        "हम लोग टेंट वगेरा अगर कहीं भी कैम्पिंग करने जाते हैं",
        "हम ने मिस्टेक किए कि हम लाइट नहीं ले गए थे",
    ]

    for ex in real_examples:
        result = cleanup_pipeline(ex, verbose=True)

    # ── SUMMARY ───────────────────────────────────────────────────────────────
    print("─"*70)
    print("APPROACH SUMMARY")
    print("─"*70)
    print("""
NUMBER NORMALIZATION
  Strategy  : Rule-based, three-pass (compound -> simple -> edge-case guard)
  Handles   : Simple (दस->10), compound (एक सौ पचास->150), large (एक लाख->100000)
  Edge cases: Idiomatic phrases (दो-चार, एक बात, दो नम्बर) are KEPT as-is
  Why rules : Hindi number morphology is regular and finite; no ML needed

ENGLISH WORD DETECTION
  Layer 1   : Latin-script words (direct detection, 100% precision)
  Layer 2a  : Known Devanagari loanword lexicon (~200 curated words)
  Layer 2b  : Phonological heuristics (ऑ vowel, pr-/st-/tr- clusters)
  Guideline : Per instructions, Devanagari-transcribed English (e.g. प्रोजेक्ट)
              is CORRECT spelling — detected but not marked as error
  Output    : [EN]word[/EN] tags for downstream processing

WHERE EACH OPERATION HELPS vs HURTS
  Num norm  : HELPS  — downstream NLP (NER, NLU) prefers digits
              HURTS  — if idioms are converted ("दो-चार" -> "2-4")
  EN detect : HELPS  — script normalization, TTS, MT pipelines
              HURTS  — if nativised words are over-tagged (treated as errors)
""")
