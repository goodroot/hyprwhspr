"""
Filler word removal for hyprwhspr.

Whisper-style backends emit bare fillers ("well um okay"), but backends that
punctuate their own transcripts attach marks to the filler itself ("Um." /
"Uh,"). Deleting only the letters strands those marks, so removal has to decide
which of the marks around a filler the sentence still needs:

1. A wrapper pair around the filler ("(um)", '"Um,"') belongs to the filler and
   leaves with it. An unpaired wrapper is not the filler's, and stays.
2. A separator (`,;:` and full-width kin) attached to the filler leaves with it:
   it marked the pause the filler itself created.
3. A sentence-terminal mark (`.!?…` and full-width kin) attached to the filler is
   structural and survives -- unless nothing precedes it, or a terminal already
   stands to its left, in which case that one wins, kept as spelled.
4. The mark that introduced the filler survives, except where rule 3 supersedes
   it: "Well, um. Okay." is one sentence break, not a comma plus one.
5. A capitalized filler that opened a sentence hands the capital to the next
   surviving word.

Rules 2 and 3 are deliberately asymmetric. A comma the recognizer attached to a
filler is transcribing the hesitation, so it has nothing left to mark once the
filler is gone; a sentence break is structure the speaker meant regardless.
"""

import re

try:
    from .text_script import boundaried_pattern
except ImportError:
    from text_script import boundaried_pattern


# Marks a filler can carry. Full-width forms are listed so CJK transcripts take
# the same path as Latin ones.
FILLER_PUNCTUATION = '.,!?;:…。，、！？：；'
FILLER_SENTENCE_END = '.!?…。！？'
# A wrapper only counts as the filler's when the closer is the opener's own
# partner: "(Um] Next" is two unrelated delimiters, not a pair around a filler.
WRAPPER_PAIRS = {
    '"': '"', "'": "'", '\u201c': '\u201d', '\u2018': '\u2019',
    '(': ')', '[': ']', '{': '}',
    '\u300c': '\u300d', '\u300e': '\u300f', '\uff08': '\uff09',
}
FILLER_OPENERS = ''.join(WRAPPER_PAIRS)
FILLER_CLOSERS = ''.join(WRAPPER_PAIRS.values())
# Marks the spot a sentence-initial filler vacated, so the word that inherits
# the sentence can be re-capitalized once every filler has been removed.
CAPITALIZE_MARKER = '\ue002'


def _trailing_character(chunk, previous):
    """Last character of `chunk` that survives into the output, else `previous`.

    A line break ends the search: nothing on an earlier line is in front of the
    filler in any sense that punctuation cares about.
    """
    for char in reversed(chunk):
        if char in '\r\n':
            return ''
        if not char.isspace() and char != CAPITALIZE_MARKER:
            return char
    return previous


def _filler_pattern(words):
    """One alternation over every filler, longest first so a short filler cannot
    win over a longer one that shares its opening (CJK terms carry no boundary).

    `close` is matched only when `open` did, so an unpaired bracket is never
    mistaken for the filler's own wrapper, and whitespace before the closer is
    part of that wrapper: "(um )" is the pair "(um)". The opener has to sit
    directly against the filler -- a quote closing an earlier word would
    otherwise be read as opening this one.
    """
    marks = '[' + re.escape(FILLER_PUNCTUATION) + ']*'
    alternatives = '|'.join(
        boundaried_pattern(word) for word in sorted(words, key=len, reverse=True)
    )
    return re.compile(
        '(?P<pre>' + marks + r')(?P<lead>[ \t]*)'
        '(?P<open>[' + re.escape(FILLER_OPENERS) + '])?'
        '(?P<word>(?:' + alternatives + '))'
        '(?P<punct>' + marks + ')'
        '(?(open)' + r'(?:(?P<gap>[ \t]*)(?P<close>['
        + re.escape(FILLER_CLOSERS) + r']))?)'
        r'(?P<trail>[ \t]*)',
        re.IGNORECASE,
    )


def _filler_replacement(match, preceding):
    """Replacement for one filler match, given the last character kept so far.

    `preceding` is empty when nothing survives in front of the filler, which an
    earlier removal can arrange.
    """
    pre = match.group('pre')
    lead = match.group('lead')
    punct = match.group('punct')
    trail = match.group('trail')
    # A wrapper is the filler's own only when both halves are present and the
    # closer is the opener's partner. Anything else is unrelated text that has
    # to survive where it stood.
    # An opener is only the filler's when it sits directly against it: a quote
    # is both an opener and a closer, so "Say \"yes\" um." must not read the
    # quote that closed "yes" as the one that opened the filler.
    opener = match.group('open') or ''
    closer = match.group('close') or ''
    wrapper = bool(opener) and closer == WRAPPER_PAIRS.get(opener)
    kept_open = '' if wrapper else opener
    # An unpaired closer keeps the spacing it was written with.
    kept_close = '' if wrapper else (match.group('gap') or '') + closer
    # The mark that closed whatever precedes the filler, whether it sits in
    # `pre` or was left standing by an earlier removal.
    preceding_mark = pre[-1:] or preceding
    opens_sentence = not preceding or preceding_mark in FILLER_SENTENCE_END
    # Nothing survives in front of the filler on this line, so there is nothing
    # for a mark to close and nothing for a space to separate it from.
    nothing_before = not preceding and not pre
    # A capitalized filler opening a sentence hands that sentence to the next
    # word. Gating on the filler's own case keeps this off abbreviations.
    marker = (
        CAPITALIZE_MARKER
        if match.group('word')[:1].isupper() and opens_sentence
        else ''
    )

    # A mark on a filler that opens a bracket is inside that bracket with the
    # filler, so it cannot be closing the sentence in front of it and leaves
    # with the filler -- the bracket itself stays where it was written.
    if punct and punct[0] in FILLER_SENTENCE_END and not opener:
        # The sentence break is real, so it outlives the filler -- unless there
        # is no sentence left in front of it to close.
        if nothing_before:
            return marker
        # A break already standing in front of the filler is that same break,
        # kept as the recognizer spelled it ("Sure... Um." keeps the ellipsis).
        if pre and pre[-1] in FILLER_SENTENCE_END:
            return pre + marker + trail
        # Otherwise the filler's own mark supersedes the comma that introduced
        # it: "Well, um. Okay." is one sentence break, not a comma plus one.
        return punct + marker + trail
    # Nothing structural to preserve: a separator on the filler only marked the
    # pause the filler created, so it leaves with the filler. The space in front
    # of the filler only separated two words, so it goes too when the filler was
    # the last thing inside a bracket: "(so um) fine" must not become "(so ) fine".
    following = match.string[match.end():match.end() + 1]
    hugs_closer = bool(following) and following in FILLER_CLOSERS
    # A delimiter left standing is content in its own right, so it still needs
    # separating from what follows even when the filler opened the line.
    vacant = nothing_before and not (kept_open or kept_close)
    separator = '' if hugs_closer or vacant else lead
    # An opener left standing clings to whatever follows it, so the space the
    # filler occupied goes with the filler: "He said (um. next" -> "He said (next".
    hugs_following = bool(kept_open) and not kept_close
    closing = '' if hugs_following or vacant else trail
    return pre + separator + kept_open + kept_close + marker + closing


def filter_filler_words(text, words):
    """Remove `words` from `text` along with the punctuation they own."""
    words = [word for word in (words or ()) if word]
    if not text or not words:
        return text

    # One left-to-right pass over every filler at once. Substituting per word
    # instead would judge each match against the text as it stood before that
    # word's pass, so a second "um." would keep a period whose sentence the
    # first "um." had already taken away.
    kept = []
    cursor = 0
    preceding = ''
    for match in _filler_pattern(words).finditer(text):
        kept.append(text[cursor:match.start()])
        preceding = _trailing_character(kept[-1], preceding)
        replacement = _filler_replacement(match, preceding)
        kept.append(replacement)
        preceding = _trailing_character(replacement, preceding)
        cursor = match.end()
    kept.append(text[cursor:])

    filtered = re.sub(
        re.escape(CAPITALIZE_MARKER) + r'([ \t]*)(\w)',
        lambda match: match.group(1) + match.group(2).upper(),
        ''.join(kept),
    )
    return re.sub(r' +', ' ', filtered.replace(CAPITALIZE_MARKER, '')).strip()
