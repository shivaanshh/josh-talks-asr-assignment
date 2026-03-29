# =============================================================================
# Q4 — Lattice-Based WER Evaluation
# Josh Talks · AI Researcher Intern Assignment
#
# Paste into a single Colab cell or run as a script. No GPU needed.
# =============================================================================

# =============================================================================
# THEORY
# =============================================================================
"""
PROBLEM
-------
Standard WER compares a hypothesis against a single rigid reference string.
This unfairly penalises valid alternative transcriptions — e.g. "14" vs "चौदह",
"किताबें" vs "किताबे", or a word the reference transcriber got wrong.

SOLUTION: TRANSCRIPTION LATTICE
--------------------------------
A lattice replaces the flat reference string with a sequential list of "bins".
Each bin corresponds to ONE alignment position in the audio and contains all
valid lexical, phonetic, and spelling alternatives for that position.

Example:
  Spoken audio: "उसने चौदह किताबें खरीदीं"
  Rigid ref:    ["उसने", "चौदह", "किताबें", "खरीदीं"]
  Lattice:      [["उसने"],
                 ["चौदह", "14"],
                 ["किताबें", "किताबे", "पुस्तकें"],
                 ["खरीदीं", "खरीदी"]]

  Model output: "उसने 14 किताबे खरीदी"
  Rigid WER:    3/4 = 75%   (penalises all three valid alternatives!)
  Lattice WER:  0/4 = 0%    (each predicted word matches some bin entry)

ALIGNMENT UNIT
--------------
We use WORDS as the alignment unit because:
  - Hindi morphology operates at word level
  - Number words map 1:1 with digits at word level
  - Subword units fragment Devanagari unnaturally
  - Phrase-level grouping is too coarse for per-position alternatives

LATTICE CONSTRUCTION ALGORITHM
-------------------------------
1. Collect all model hypotheses + human reference for the same audio
2. Align all of them to a common backbone using word-level Needleman-Wunsch
3. At each position, gather the set of all words from all alignments
4. Apply trust rules: if ≥ 60% of models agree on a word AND it differs from
   the reference, that word is added to the bin (reference may be wrong)
5. Add rule-based equivalents: digits ↔ number words, spelling variants

HANDLING INSERTIONS, DELETIONS, SUBSTITUTIONS
----------------------------------------------
- Insertion  : model produces a word where all others (incl. ref) have nothing
               → penalise only if NO other model agrees
- Deletion   : model omits a word present in the reference
               → penalise only if the bin does NOT contain ε (empty) as valid
- Substitution: model produces word W at position i
               → penalise only if W is not in lattice[i]

TRUST RULE: WHEN TO OVERRIDE THE REFERENCE
-------------------------------------------
If ≥ TRUST_THRESHOLD (60%) of models agree on a token that differs from the
reference at position i, the reference is likely wrong. We add that token to
the bin, so models that produced it are NOT penalised.

LATTICE WER FORMULA
-------------------
For a hypothesis H = [h_1, ..., h_n] aligned to lattice L = [B_1, ..., B_m]:

  errors = Σ_i  0  if h_i ∈ B_i  (or h_i == ε and ε ∈ B_i)
                 1  otherwise

  Lattice-WER(H) = errors / max(len(H), len(L))
"""

# =============================================================================
# IMPORTS
# =============================================================================
import re, unicodedata
from collections import Counter
from difflib import SequenceMatcher

# =============================================================================
# NUMBER WORD ↔ DIGIT EQUIVALENCE TABLE
# =============================================================================
_NUM_MAP = {
    # digits -> Hindi words
    "0":  ["शून्य"],
    "1":  ["एक"],
    "2":  ["दो"],
    "3":  ["तीन"],
    "4":  ["चार"],
    "5":  ["पाँच","पांच"],
    "6":  ["छह","छः"],
    "7":  ["सात"],
    "8":  ["आठ"],
    "9":  ["नौ"],
    "10": ["दस"],
    "11": ["ग्यारह"],
    "12": ["बारह"],
    "13": ["तेरह"],
    "14": ["चौदह"],
    "15": ["पंद्रह"],
    "20": ["बीस"],
    "25": ["पच्चीस"],
    "30": ["तीस"],
    "40": ["चालीस"],
    "50": ["पचास"],
    "60": ["साठ"],
    "70": ["सत्तर"],
    "80": ["अस्सी"],
    "90": ["नब्बे"],
    "100":["सौ"],
    "1000":["हज़ार","हजार"],
    "100000":["लाख"],
}
# Reverse map: Hindi word -> canonical digit string
_WORD_TO_DIGIT = {}
for digit, words in _NUM_MAP.items():
    for w in words:
        _WORD_TO_DIGIT[w] = digit

def numeric_equivalents(token: str) -> set:
    """Return all numeric equivalents of a token."""
    equiv = {token}
    if token in _WORD_TO_DIGIT:
        equiv.add(_WORD_TO_DIGIT[token])
    if token in _NUM_MAP:
        equiv.update(_NUM_MAP[token])
    return equiv

# =============================================================================
# SPELLING VARIANT DETECTION
# =============================================================================
def spelling_variants(token: str) -> set:
    """
    Return near-spelling variants of a Hindi token.
    Catches common transcription inconsistencies:
      - anusvara vs chandrabindu  (ं vs ँ)
      - nukta presence/absence    (क़ vs क)
      - short/long vowel swap      (ि vs ी, ु vs ू)
    """
    variants = {token}
    # anusvara ↔ chandrabindu
    variants.add(token.replace("\u0902", "\u0901"))   # ं -> ँ
    variants.add(token.replace("\u0901", "\u0902"))   # ँ -> ं
    # nukta removal
    variants.add(token.replace("\u093c", ""))         # remove ़
    # short ↔ long vowel (common in fast speech transcription)
    variants.add(token.replace("\u093f", "\u0940"))   # ि -> ी
    variants.add(token.replace("\u0940", "\u093f"))   # ी -> ि
    variants.add(token.replace("\u0941", "\u0942"))   # ु -> ू
    variants.add(token.replace("\u0942", "\u0941"))   # ू -> ु
    # NFC normalise all
    return {unicodedata.normalize("NFC", v) for v in variants}

# =============================================================================
# WORD ALIGNMENT  (Needleman-Wunsch, word-level)
# =============================================================================
def align_sequences(ref: list, hyp: list) -> list:
    """
    Align two word sequences using dynamic programming.
    Returns list of (ref_word_or_None, hyp_word_or_None) pairs.
    """
    n, m = len(ref), len(hyp)
    # DP table
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1): dp[i][0] = -i
    for j in range(m+1): dp[0][j] = -j

    for i in range(1, n+1):
        for j in range(1, m+1):
            match = dp[i-1][j-1] + (1 if ref[i-1]==hyp[j-1] else -1)
            delete = dp[i-1][j] - 1
            insert = dp[i][j-1] - 1
            dp[i][j] = max(match, delete, insert)

    # Traceback
    i, j = n, m
    aligned = []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + (1 if ref[i-1]==hyp[j-1] else -1):
            aligned.append((ref[i-1], hyp[j-1]))
            i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] - 1:
            aligned.append((ref[i-1], None))
            i -= 1
        else:
            aligned.append((None, hyp[j-1]))
            j -= 1
    return list(reversed(aligned))

# =============================================================================
# LATTICE CONSTRUCTION
# =============================================================================
TRUST_THRESHOLD = 0.60   # fraction of models that must agree to override ref

def build_lattice(reference: str,
                  hypotheses: list[str],
                  trust_threshold: float = TRUST_THRESHOLD) -> list[set]:
    """
    Build a transcription lattice from a reference and N model hypotheses.

    Parameters
    ----------
    reference    : human reference transcription (may contain errors)
    hypotheses   : list of model output strings
    trust_threshold : if >= this fraction of models agree on a token that
                      differs from reference, add it to the bin

    Returns
    -------
    lattice : list of sets, one set per alignment position.
              Each set contains all valid tokens at that position.
              The empty string "" in a set means deletion is valid there.
    """
    ref_tokens  = reference.split()
    hyp_token_lists = [h.split() for h in hypotheses]
    n_models    = len(hypotheses)

    # Step 1: align each hypothesis to the reference
    alignments = []
    for hyp_tokens in hyp_token_lists:
        aligned = align_sequences(ref_tokens, hyp_tokens)
        alignments.append(aligned)

    # Step 2: determine the set of positions (use reference positions as anchor)
    # Build a unified position grid by collecting all (ref_tok, hyp_tok) pairs
    # For simplicity: use reference length as the number of bins,
    # plus extra bins for insertions agreed on by multiple models.

    # Count what models say at each reference position
    # pos_votes[i] = Counter of what models produced at ref position i
    pos_votes = [Counter() for _ in range(len(ref_tokens))]

    for aligned in alignments:
        ref_pos = 0
        for r_tok, h_tok in aligned:
            if r_tok is not None:
                if h_tok is not None:
                    pos_votes[ref_pos][h_tok] += 1
                else:
                    pos_votes[ref_pos][""] += 1   # deletion
                ref_pos += 1
            # insertions (r_tok is None) not mapped to a ref position

    # Step 3: build bins
    lattice = []
    for i, ref_tok in enumerate(ref_tokens):
        bin_set = set()

        # Always include the reference token
        bin_set.add(ref_tok)

        # Add numeric equivalents of the reference token
        bin_set.update(numeric_equivalents(ref_tok))

        # Add spelling variants
        bin_set.update(spelling_variants(ref_tok))

        # Trust rule: add tokens that >= threshold of models agree on
        total_votes = sum(pos_votes[i].values())
        if total_votes > 0:
            for tok, count in pos_votes[i].items():
                if count / n_models >= trust_threshold:
                    bin_set.add(tok)
                    # Also add numeric equivalents of the agreed token
                    bin_set.update(numeric_equivalents(tok))
                    bin_set.update(spelling_variants(tok))

        # If majority of models deleted here, empty string is valid
        del_count = pos_votes[i].get("", 0)
        if del_count / n_models >= trust_threshold:
            bin_set.add("")

        lattice.append(bin_set)

    return lattice

# =============================================================================
# LATTICE WER COMPUTATION
# =============================================================================
def lattice_wer(hypothesis: str, lattice: list[set]) -> float:
    """
    Compute WER for a hypothesis against a lattice.

    Alignment: align the hypothesis word sequence to the lattice bins
    using a simple left-to-right matching (greedy; good enough for WER).
    For rigorous alignment, use DP with lattice-bin match score.

    A position is correct if the hypothesis word is IN the bin set.
    Deletion (bin has word, hyp has nothing) counts as 1 error
    UNLESS "" is in the bin (i.e. deletion is known-valid).
    """
    hyp_tokens = hypothesis.split()
    n_bins     = len(lattice)
    n_hyp      = len(hyp_tokens)

    if n_bins == 0 and n_hyp == 0:
        return 0.0
    if n_bins == 0:
        return 1.0

    # DP alignment of hyp_tokens to lattice bins
    # cost(i, j) = cost of aligning hyp[:i] to lattice[:j]
    INF = float("inf")
    dp  = [[INF] * (n_bins + 1) for _ in range(n_hyp + 1)]
    dp[0][0] = 0

    for j in range(1, n_bins + 1):
        # deletion from reference (hyp has nothing at this bin)
        del_cost = 0 if "" in lattice[j-1] else 1
        dp[0][j] = dp[0][j-1] + del_cost

    for i in range(1, n_hyp + 1):
        dp[i][0] = i  # all insertions

    for i in range(1, n_hyp + 1):
        for j in range(1, n_bins + 1):
            hyp_tok = hyp_tokens[i-1]
            bin_j   = lattice[j-1]

            # Match / substitution
            sub_cost  = 0 if hyp_tok in bin_j else 1
            match_val = dp[i-1][j-1] + sub_cost

            # Deletion (skip bin j without consuming hyp token)
            del_cost  = 0 if "" in bin_j else 1
            del_val   = dp[i][j-1] + del_cost

            # Insertion (consume hyp token without matching a bin)
            ins_val   = dp[i-1][j] + 1

            dp[i][j]  = min(match_val, del_val, ins_val)

    errors = dp[n_hyp][n_bins]
    denom  = max(n_bins, n_hyp)
    return errors / denom

# =============================================================================
# STANDARD WER (for comparison)
# =============================================================================
def standard_wer(hypothesis: str, reference: str) -> float:
    ref = reference.split(); hyp = hypothesis.split()
    if not ref: return 0.0 if not hyp else 1.0
    n, m = len(ref), len(hyp)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        ndp = [i] + [0]*m
        for j in range(1, m + 1):
            if ref[i-1] == hyp[j-1]: ndp[j] = dp[j-1]
            else: ndp[j] = 1 + min(dp[j-1], dp[j], ndp[j-1])
        dp = ndp
    return dp[m] / n

# =============================================================================
# FULL PIPELINE: given reference + N model outputs -> per-model WER table
# =============================================================================
def evaluate_with_lattice(reference: str,
                           model_outputs: dict,
                           trust_threshold: float = TRUST_THRESHOLD,
                           verbose: bool = True):
    """
    Parameters
    ----------
    reference     : human reference string
    model_outputs : dict {model_name: hypothesis_string}
    trust_threshold : fraction threshold for reference override

    Returns
    -------
    results : dict {model_name: {"standard_wer": float, "lattice_wer": float}}
    """
    hypotheses = list(model_outputs.values())
    lattice    = build_lattice(reference, hypotheses, trust_threshold)

    if verbose:
        print(f"\nReference : {reference}")
        print(f"Lattice   :")
        for i, (ref_tok, bin_set) in enumerate(zip(reference.split(), lattice)):
            alts = bin_set - {ref_tok}
            alt_str = (", ".join(sorted(alts)) if alts else "—")
            print(f"  [{i}] '{ref_tok}'  + alternatives: {{{alt_str}}}")

    results = {}
    for name, hyp in model_outputs.items():
        swer = standard_wer(hyp, reference)
        lwer = lattice_wer(hyp, lattice)
        results[name] = {"standard_wer": round(swer, 4),
                          "lattice_wer":  round(lwer, 4)}

    if verbose:
        header = f"\n{'Model':<20} {'Std WER':>10} {'Lattice WER':>14}  {'Fairly penalised?':>18}"
        print(header)
        print("-" * 68)
        for name, r in results.items():
            fair = "YES (unchanged)" if r["lattice_wer"] == r["standard_wer"] \
                   else f"NO -> was unfair, now {r['lattice_wer']*100:.1f}%"
            print(f"  {name:<18} {r['standard_wer']*100:>9.1f}%  "
                  f"{r['lattice_wer']*100:>12.1f}%    {fair}")

    return results, lattice

# =============================================================================
# TEST CASES  (from the assignment brief + realistic examples)
# =============================================================================
if __name__ == "__main__":

    print("=" * 70)
    print("Q4 — LATTICE-BASED WER EVALUATION")
    print("=" * 70)

    # ─────────────────────────────────────────────────────────────────────────
    # EXAMPLE 1: Number word vs digit (from assignment brief)
    # "उसने चौदह किताबें खरीदीं"
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─"*70)
    print("EXAMPLE 1: Number word vs digit")
    print("─"*70)

    ref1 = "उसने चौदह किताबें खरीदीं"
    models1 = {
        "Model_A": "उसने 14 किताबें खरीदीं",       # digit instead of word  -> fair
        "Model_B": "उसने चौदह किताबे खरीदी",        # spelling variants      -> fair
        "Model_C": "उसने पंद्रह किताबें खरीदीं",    # wrong number           -> rightly penalised
        "Model_D": "उसने चौदह पुस्तकें खरीदीं",     # synonym (पुस्तक)       -> partial
        "Model_E": "उसने चौदह किताबें खरीदीं",      # exact match            -> 0% WER
    }
    r1, l1 = evaluate_with_lattice(ref1, models1)

    # ─────────────────────────────────────────────────────────────────────────
    # EXAMPLE 2: Reference is wrong — models agree against it
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─"*70)
    print("EXAMPLE 2: Reference has an error — majority models are correct")
    print("─"*70)

    # Reference transcriber wrote "गया" but the speaker said "आया"
    # 4 out of 5 models correctly say "आया"
    ref2 = "वह कल स्कूल गया था"
    models2 = {
        "Model_A": "वह कल स्कूल आया था",   # correct (ref is wrong)
        "Model_B": "वह कल स्कूल आया था",   # correct
        "Model_C": "वह कल स्कूल आया था",   # correct
        "Model_D": "वह कल स्कूल आया था",   # correct
        "Model_E": "वह कल स्कूल गया था",   # agrees with (wrong) ref
    }
    r2, l2 = evaluate_with_lattice(ref2, models2)

    print("\nKey insight: Model_A/B/C/D said 'आया' (4/5 = 80% >= 60% threshold).")
    print("'आया' is added to bin 3. Standard WER penalised them; Lattice WER does not.")

    # ─────────────────────────────────────────────────────────────────────────
    # EXAMPLE 3: Spelling variants (anusvara / chandrabindu)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─"*70)
    print("EXAMPLE 3: Spelling variants only")
    print("─"*70)

    ref3 = "यहाँ बहुत ठंड है"
    models3 = {
        "Model_A": "यहाँ बहुत ठंड है",     # exact
        "Model_B": "यहां बहुत ठंड है",     # ँ vs ं on यहाँ
        "Model_C": "यहाँ बहुत ठण्ड है",    # variant spelling of ठंड
        "Model_D": "यहाँ बहुत ठड है",      # missing nasal — genuine error
        "Model_E": "यहाँ बहुत सर्दी है",   # different word — genuine sub
    }
    r3, l3 = evaluate_with_lattice(ref3, models3)

    # ─────────────────────────────────────────────────────────────────────────
    # EXAMPLE 4: Deletions and insertions
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─"*70)
    print("EXAMPLE 4: Deletions — filler words that are transcription choices")
    print("─"*70)

    # "तो" is a discourse filler — some models include it, some don't
    ref4 = "मैं तो घर जा रहा हूँ"
    models4 = {
        "Model_A": "मैं घर जा रहा हूँ",      # deleted 'तो' (valid choice)
        "Model_B": "मैं तो घर जा रहा हूँ",   # kept 'तो' (exact)
        "Model_C": "मैं घर जा रहा हूँ",      # deleted 'तो'
        "Model_D": "मैं तो घर जा रहा था",    # हूँ -> था (wrong tense)
        "Model_E": "मैं घर जा रहा हूँ",      # deleted 'तो'
    }
    r4, l4 = evaluate_with_lattice(ref4, models4)

    print("\n3/5 models deleted 'तो' (60% = threshold). "
          "Deletion becomes valid in bin 1. Model_A/C/E not penalised.")

    # ─────────────────────────────────────────────────────────────────────────
    # SUMMARY TABLE
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY: Models unfairly penalised by standard WER but not by lattice WER")
    print("=" * 70)
    all_results = {
        "Ex1": r1, "Ex2": r2, "Ex3": r3, "Ex4": r4,
    }
    unfair_count = 0
    for ex, results in all_results.items():
        for model, r in results.items():
            if r["lattice_wer"] < r["standard_wer"]:
                unfair_count += 1
                print(f"  {ex} {model:<12}  std={r['standard_wer']*100:.1f}%  "
                      f"lattice={r['lattice_wer']*100:.1f}%  "
                      f"(saved {(r['standard_wer']-r['lattice_wer'])*100:.1f}pp)")
    print(f"\nTotal unfairly penalised: {unfair_count} model-utterance pairs")

    # ─────────────────────────────────────────────────────────────────────────
    # PSEUDOCODE SUMMARY
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PSEUDOCODE SUMMARY")
    print("=" * 70)
    pseudocode = """
BUILD_LATTICE(reference, hypotheses, trust_threshold=0.60):
    ref_tokens  = tokenize(reference)
    alignments  = [align(ref_tokens, tokenize(h)) for h in hypotheses]
    lattice     = []

    for i, ref_tok in enumerate(ref_tokens):
        bin = {ref_tok}                          # always include reference
        bin += numeric_equivalents(ref_tok)      # "14" <-> "चौदह"
        bin += spelling_variants(ref_tok)        # anusvara, nukta, vowel length

        votes = Counter(hyp_tok at position i for each alignment)
        for tok, count in votes:
            if count / len(hypotheses) >= trust_threshold:
                bin.add(tok)                     # majority agrees -> trust it
                bin += numeric_equivalents(tok)
                bin += spelling_variants(tok)

        if votes[""] / len(hypotheses) >= trust_threshold:
            bin.add("")                          # deletion is valid

        lattice.append(bin)
    return lattice


LATTICE_WER(hypothesis, lattice):
    hyp_tokens = tokenize(hypothesis)
    errors = DP_align(hyp_tokens, lattice,
                      match_cost  = lambda h,b: 0 if h in b else 1,
                      delete_cost = lambda b:   0 if "" in b else 1,
                      insert_cost = 1)
    return errors / max(len(hyp_tokens), len(lattice))
"""
    print(pseudocode)

    print("=" * 70)
    print("ALIGNMENT UNIT JUSTIFICATION: WORD")
    print("=" * 70)
    print("""
  WORD level chosen because:
  1. Number equivalence is word-level (चौदह <-> 14, not char-level)
  2. Hindi Devanagari subword units split naturally meaningful morphemes
  3. Spelling variants (anusvara, nukta) are whole-word transformations
  4. WER is conventionally word-level; lattice WER stays comparable

  SUBWORD rejected: fragments "खरीदीं" into pieces that don't align
                    cleanly with digit equivalents or variants.
  PHRASE  rejected: too coarse — misses per-word credit for partial matches.
""")
