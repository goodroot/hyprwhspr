"""Pure, language-aware dictation processing shared by batch and live delivery."""

import re
from dataclasses import dataclass
from typing import Optional

try:
    from .backend_utils import normalize_backend
    from .text_script import boundaried_pattern
    from .filler_filter import filter_filler_words
except ImportError:
    from backend_utils import normalize_backend
    from text_script import boundaried_pattern
    from filler_filter import filter_filler_words


_SPOKEN_REPLACEMENTS = (
    ('period', '.'), ('comma', ','), ('question mark', '?'),
    ('exclamation mark', '!'), ('colon', ':'), ('semicolon', ';'),
    ('tab', '\t'), ('dash', '-'), ('underscore', '_'),
    ('open paren', '('), ('close paren', ')'), ('open bracket', '['),
    ('close bracket', ']'), ('open brace', '{'), ('close brace', '}'),
    ('at symbol', '@'), ('hash', '#'), ('dollar sign', '$'),
    ('percent', '%'), ('caret', '^'), ('ampersand', '&'),
    ('asterisk', '*'), ('plus', '+'), ('equals', '='),
    ('less than', '<'), ('greater than', '>'), ('slash', '/'),
    ('backslash', '\\'), ('pipe', '|'), ('tilde', '~'), ('grave', '`'),
    ('quote', '"'), ('apostrophe', "'"),
)


def _config_setting(config_manager, name, default=None):
    if config_manager is None:
        return default
    return config_manager.get_setting(name, default)


def _apply_configured_word_overrides(text, config_manager):
    if config_manager is None:
        return text
    for original, replacement in (config_manager.get_word_overrides() or {}).items():
        if original:
            # A single character is a spelling fix ("ß" -> "ss"), so it applies inside words.
            pattern = re.escape(original) if len(original) == 1 else boundaried_pattern(original)
            # Overrides are literal text, including backslashes and regex-like snippets.
            text = re.sub(
                pattern, lambda _match: replacement, text, flags=re.IGNORECASE
            )
    return re.sub(r' +', ' ', text).strip()


def _filter_configured_filler_words(text, config_manager):
    if config_manager is None or not config_manager.get_filter_filler_words():
        return text
    return filter_filler_words(text, config_manager.get_filler_words())


@dataclass(frozen=True)
class CommandProfile:
    language: str
    replacements: tuple
    line_breaks: tuple
    space_before: str = ''

    @property
    def phrases(self):
        return self.line_breaks + tuple(command for command, _ in self.replacements)


_FRENCH_REPLACEMENTS = (
    ("point d'interrogation", '?'), ("point d’interrogation", '?'),
    ("point d'exclamation", '!'), ("point d’exclamation", '!'),
    ('point-virgule', ';'), ('point virgule', ';'),
    ('point de suspension', '…'), ('points de suspension', '…'),
    ('insérer deux points', ':'), ('insérer 2 points', ':'),
    ('insérer un point', '.'), ('insérer une virgule', ','),
    ('crochet ouvrant', '['), ('ouvrir le crochet', '['),
    ('crochet fermant', ']'), ('fermer le crochet', ']'),
    ('parenthèse ouvrante', '('), ('ouvrir la parenthèse', '('),
    ('parenthèse fermante', ')'), ('fermer la parenthèse', ')'),
    ('insérer une arobase', '@'), ('insérer une arrobase', '@'),
    ('insérer une barre verticale', '|'),
) + tuple(
    (f'{word}{number} espace', '- ')
    for word in ('tiret', 'tirer', 'tiré')
    for number in ('', ' du 6', ' du six')
)

_PROFILES = {
    'en': CommandProfile('en', _SPOKEN_REPLACEMENTS, ('new line',)),
    'fr': CommandProfile('fr', _FRENCH_REPLACEMENTS,
                         ('retour à la ligne', 'retour a la ligne'), '?!:;'),
}


@dataclass(frozen=True)
class ProcessingContext:
    """Output language for one request; None retains the English fallback."""
    output_language: Optional[str] = None

    @property
    def profile(self):
        # Commands follow the base language: "fr-FR" and "fr_CA" speak French.
        language = re.split(r'[-_]', self.output_language or 'en', maxsplit=1)[0]
        return _PROFILES.get(language)


def resolve_processing_context(config=None, language_override=None):
    language = (language_override if language_override is not None
                else _config_setting(config, 'language'))
    language = (language.strip().lower() or None) if isinstance(language, str) else None
    backend = normalize_backend(_config_setting(config, 'transcription_backend', 'pywhispercpp'))
    if (backend in ('pywhispercpp', 'cpu', 'nvidia', 'vulkan', 'faster-whisper')
            and _config_setting(config, 'task', 'transcribe') == 'translate'):
        language = 'en'
    return ProcessingContext(language)


def command_pattern(command):
    return r'[ \t]+'.join(re.escape(word) for word in command.split())


def phrase_word(word):
    """Compare command tokens independently of recognizer punctuation."""
    return re.sub(r'[^\w]', '', word.lower())


def line_break_length(words, profile):
    if profile is not None:
        for phrase in profile.line_breaks:
            tokens = tuple(phrase_word(w) for w in phrase.split())
            if tuple(phrase_word(w) for w in words[:len(tokens)]) == tokens:
                return len(tokens)
    return 0


def preprocess_text(text: str, config_manager=None, *, context=None) -> str:
    """Apply dictation corrections without injection or external hooks."""
    processed = text.replace('\r\n', ' ').replace('\r', ' ').replace('\n', ' ')
    processed = _apply_configured_word_overrides(processed, config_manager)
    processed = _filter_configured_filler_words(processed, config_manager)

    if not _config_setting(config_manager, 'symbol_replacements', True):
        return re.sub(r'[ \t]+', ' ', processed).strip()

    context = context or resolve_processing_context(config_manager)
    profile = context.profile
    if profile is None:
        return re.sub(r'[ \t]+', ' ', processed).strip()
    line_break = '|'.join(command_pattern(command) for command in profile.line_breaks)
    replacements = sorted(profile.replacements, key=lambda item: len(item[0]), reverse=True)

    # Recognizers commonly punctuate the spoken command itself.
    processed = re.sub(
        rf'\b(?:{line_break})\b(?:[ \t]*[.!?]+)?', '\n', processed,
        flags=re.IGNORECASE,
    )

    # Mark commands before spacing them. Identical literal punctuation touching
    # a command is another rendering of the same requested symbol.
    markers = {}
    for index, (command, symbol) in enumerate(replacements):
        marker = f'\ue000{index}\ue001'
        markers[marker] = symbol
        # Recognizers render a spoken ellipsis as periods too: "Attends... points
        # de suspension." is one ellipsis.
        escaped_symbol = r'(?:…|\.+)' if symbol == '…' else re.escape(symbol)
        pattern_command = rf'(?:{command_pattern(command)})'
        single_symbol = (r'(?:…|\.+)(?![.…])' if symbol == '…'
                         else rf'{escaped_symbol}(?!{escaped_symbol})')
        # Sentence punctuation the recognizer put before the command is the
        # command's own, even with nothing after it: "Prêt. Insérer un point".
        echo_after = '?' if symbol in '.,?!:;…' else ''
        pattern = (
            rf'(?:{escaped_symbol}+[ \t]+\b{pattern_command}\b(?:{single_symbol}){echo_after}'
            rf'|{escaped_symbol}+\b{pattern_command}\b'
            rf'|\b{pattern_command}\b{single_symbol}'
            rf'|\b{pattern_command}\b)'
        )
        processed = re.sub(pattern, marker, processed, flags=re.IGNORECASE)

    # Only command markers receive punctuation spacing; unrelated raw
    # punctuation, including ellipses and repeated emphasis, stays untouched.
    for marker, symbol in markers.items():
        if symbol == '- ':
            # Keep deliberate spacing on a Markdown list marker through strip().
            processed = re.sub(rf'{re.escape(marker)}(?:[ \t]*[.,:;?!…]+)?[ \t]*', marker, processed)
        elif symbol in '.,?!:;…':
            if symbol in profile.space_before:
                symbol = ' ' + symbol
            processed = re.sub(rf'[ \t]*{re.escape(marker)}[ \t]*', marker, processed)
            # Add separation only before following prose. A different raw
            # punctuation mark is preserved directly beside the spoken one.
            processed = re.sub(
                rf'{re.escape(marker)}(?=[^\s\ue000)\]}}>,.?!:;。，、！？：；])',
                symbol + ' ', processed,
            )
            processed = processed.replace(marker, symbol)
        elif symbol in '([{':
            processed = re.sub(rf'{re.escape(marker)}[ \t]*', marker, processed)
            processed = processed.replace(marker, symbol)
        elif symbol in ')]}':
            processed = re.sub(rf'[ \t]*{re.escape(marker)}', marker, processed)
            processed = processed.replace(marker, symbol)
        else:
            processed = processed.replace(marker, symbol)

    processed = re.sub(r'[ \t]+', ' ', processed)
    processed = re.sub(r' *\n *', '\n', processed)
    processed = processed.strip()
    for marker, symbol in markers.items():
        if symbol == '- ':
            processed = processed.replace(marker, symbol)
    return processed
