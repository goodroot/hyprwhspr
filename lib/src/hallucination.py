"""
Whisper hallucination markers for silence/noise segments.

Whisper emits stock subtitle-corpus phrases when handed audio with no speech in
it. The list is user-configurable because entries like "you" are both the most
common phantom and a legitimate one-word dictation.
"""

# Ordered, not a set: the schema-sync test compares this against the JSON default.
DEFAULT_HALLUCINATION_MARKERS = [
    'blank audio', 'blank', 'silence', 'no speech',
    'you', 'thank you', 'thanks for watching', 'thank you for watching',
    'video playback', 'music', 'music playing', 'keyboard clicking',
]


def _normalize(text: str) -> str:
    return text.lower().replace('_', ' ').strip('[]().!?, ')


def is_hallucination(text: str, markers=None) -> bool:
    """True if text is a stock Whisper phantom rather than real dictation."""
    if not text:
        return False
    if text.startswith('♪'):
        return True
    if markers is None:
        markers = DEFAULT_HALLUCINATION_MARKERS
    # Markers are normalized too: the docs promise case/bracket-insensitive
    # matching, so a hand-written "[Background noise]" has to match as well.
    return _normalize(text) in {_normalize(marker) for marker in markers if marker}
