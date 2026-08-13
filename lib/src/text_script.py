"""
Script-awareness helpers for text assembly.

The text pipeline assumes space-separated words, which makes an inserted space a
visible artifact in CJK. Decided by codepoint range, never by language detection.

Thai is excluded on purpose: no spaces between words, but spaces between phrases.
"""

import re

# Scripts written without inter-word spaces. Inclusive ranges.
_NO_SPACE_RANGES = (
    (0x3000, 0x303F),   # CJK symbols and punctuation (。、「」)
    (0x3040, 0x30FF),   # Hiragana, Katakana
    (0x3100, 0x318F),   # Bopomofo, Hangul compatibility Jamo
    (0x31F0, 0x31FF),   # Katakana phonetic extensions
    (0x3400, 0x4DBF),   # CJK Unified Ideographs Extension A
    (0x4E00, 0x9FFF),   # CJK Unified Ideographs
    (0xF900, 0xFAFF),   # CJK Compatibility Ideographs
    (0xAC00, 0xD7AF),   # Hangul syllables
    (0x1100, 0x11FF),   # Hangul Jamo
    (0xFF00, 0xFFDC),   # Full-width forms (，！？), halfwidth Katakana and Hangul
    (0x20000, 0x2A6DF), # CJK Unified Ideographs Extension B (rare given names)
)


def is_no_space_char(char: str) -> bool:
    """True if char belongs to a script written without inter-word spaces."""
    if not char:
        return False
    code = ord(char[0])
    return any(start <= code <= end for start, end in _NO_SPACE_RANGES)


def ends_with_no_space_script(text: str) -> bool:
    """True if the last non-whitespace character is CJK."""
    stripped = (text or "").rstrip()
    return is_no_space_char(stripped[-1:]) if stripped else False


def needs_word_boundaries(term: str) -> tuple:
    """Whether (start, end) of term should be guarded by a word boundary.

    A boundary only means something in a space-delimited script: wrapped in
    \\b...\\b a CJK term never fires mid-sentence, ß never matches in Straße,
    and c++ never matches at all.
    """
    if not term:
        return (False, False)
    # A mixed term (AI助手) is anchored by its CJK half, so guarding the Latin
    # edge is enough to stop it matching: CJK characters are \w, so the guard
    # can never be satisfied inside CJK text.
    if any(is_no_space_char(char) for char in term):
        return (False, False)
    if len(term) == 1:
        return (False, False)

    def guarded(char: str) -> bool:
        return char.isalnum() or char == '_'

    return (guarded(term[0]), guarded(term[-1]))


def boundaried_pattern(term: str) -> str:
    """Escaped regex for term, guarded only on edges where a boundary applies."""
    lead_guard, tail_guard = needs_word_boundaries(term)
    return (
        (r'(?<!\w)' if lead_guard else '')
        + re.escape(term)
        + (r'(?!\w)' if tail_guard else '')
    )


def join_segments(parts) -> str:
    """Join transcript segments, omitting the space between CJK neighbours.

    Parts are stripped first: Whisper segment text carries a leading space, which
    would otherwise survive the join and reinstate the space we just skipped.
    """
    cleaned = [part.strip() for part in (parts or []) if part and part.strip()]
    if not cleaned:
        return ""
    joined = cleaned[0]
    for part in cleaned[1:]:
        # Parts are stripped, so the tail is never whitespace: check it directly
        # rather than rstrip()-ing the whole accumulated transcript each pass.
        if not is_no_space_char(joined[-1]):
            joined += ' '
        joined += part
    return joined
