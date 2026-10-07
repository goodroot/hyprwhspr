# Text processing

What happens to words between the model and the paste.

## Word overrides

Customize transcriptions:

```jsonc
{
    "word_overrides": {
        "hyper whisper": "hyprwhspr",
        "um": ""
    }
}
```

Use empty string `""` to delete words entirely.

`{"hyper whisper": "hyprwhspr"}` ships as the default (the product name is spoken "hyper whisper").

Single-character overrides match anywhere in a word (not just at word boundaries):

```jsonc
{
    "word_overrides": {
        "ß": "ss"
    }
}
```

- `"Straße"` → `"Strasse"`, `"Fuß"` → `"Fuss"`, etc.
- Multi-character overrides use whole-word matching, applied per edge
- Terms containing a CJK character match anywhere — `{"你好": "HI"}` applies inside `"我说你好世界"`
- An edge that isn't a letter or digit is unanchored, so `{"c++": "C++"}` matches

## Filler word filtering

Remove common filler words automatically:

```jsonc
{
    "filter_filler_words": true,  // Enable automatic filler word removal (default: false)
    "filler_words": ["uh", "um", "er", "ah", "eh", "hmm", "hm", "mm", "mhm"]  // Customize list
}
```

- Backends that punctuate their own transcripts attach the mark to the filler (`"Um."`, `"Uh,"`); the mark is removed along with the filler, not left stranded
- Punctuation belonging to the surrounding sentence survives -- `"I said, um, no."` keeps its first comma, and a sentence break the filler carried is kept: `"Well, um. Okay."` -> `"Well. Okay."`
- A word left starting a sentence is re-capitalized: `"Fair enough. Um. Uh, what about it?"` -> `"Fair enough. What about it?"`
- A bracket or quote pair wrapping a filler goes with it (`"Um," he said.` -> `He said.`); an unpaired one stays (`(so um) fine` -> `(so) fine`)
- Dictated punctuation is preserved: filtering runs before speech-to-symbol replacement, so a spoken `comma` beside a filler survives
- An utterance that was nothing but fillers pastes nothing at all

## Hallucination markers

Whisper invents stock subtitle phrases when handed audio with no speech in it. Transcriptions matching this list are discarded rather than pasted:

```jsonc
{
    "hallucination_markers": ["blank audio", "silence", "no speech", "thanks for watching"]
}
```

- Setting the key replaces the built-in list; `hyprwhspr config show --all` prints the default
- Matching ignores case, underscores, brackets and trailing punctuation, so `[Silence]` and `blank_audio` are caught
- Text starting with `♪` is always discarded
- `"you"` ships in the list — Whisper's most common phantom, and a real one-word dictation. Drop it if you'd rather keep the phantoms than lose a dictated "you"

## Symbol replacements

Automatically converts spoken words to symbols and punctuation:

```jsonc
{
    "symbol_replacements": true  // default: true (set false to disable speech-to-symbol replacements)
}
```

**Punctuation:**

- "period" → "."
- "comma" → ","
- "question mark" → "?"
- "exclamation mark" → "!"
- "colon" → ":"
- "semicolon" → ";"

**Symbols:**

- "at symbol" → "@"
- "hash" → "#"
- "plus" → "+"
- "equals" → "="
- "dash" → "-"
- "underscore" → "_"

**Brackets:**

- "open paren" → "("
- "close paren" → ")"
- "open bracket" → "["
- "close bracket" → "]"
- "open brace" → "{"
- "close brace" → "}"

**Special commands:**

- "new line" → new line
- "tab" → tab character

The table is English-only. See [Non-Latin scripts](#non-latin-scripts).

## Trailing space

Each transcription is followed by a space so the next word you type doesn't collide with it:

```jsonc
{
    "append_trailing_space": "auto"  // "auto" (default), true, or false
}
```

- `"auto"` checks the last character pasted: Han, Kana, Hangul and full-width punctuation (`。`, `，`) get no space
- Everything else gets one, Thai included — it has no spaces between words, but does use them between phrases
- Set `true` or `false` to fix the answer regardless of content
- The check runs after [`post_transcription_hook`](#post-transcription-hook), on the final text

## Non-Latin scripts

Chinese, Japanese and Korean write without spaces between words:

- Trailing spaces are handled by [`append_trailing_space`](#trailing-space); the default needs no configuration
- Realtime and long-form segments are joined without a space after a CJK character, so sentences aren't broken up
- The symbol replacements and [hallucination markers](#hallucination-markers) ship English entries only — add your own
- Spoken punctuation goes in `word_overrides`, which match anywhere in CJK text:

```jsonc
{ "word_overrides": { "句号": "。", "逗号": "，", "问号": "？" } }
```

## Post-transcription hook

Pipe each transcription through a shell command before it's pasted. Stdin receives the (preprocessed) transcription; non-empty stdout replaces it. Empty stdout leaves the text unchanged, so the same mechanism works for both transforms and fire-and-forget observers. A hook that exits with status `77` consumes the transcription successfully and prevents it from being pasted.

```jsonc
{
    "post_transcription_hook": "sed 's|.*|<dictation>&</dictation>|'"
}
```

The example above wraps every injected transcription in `<dictation>...</dictation>` — a useful signal to downstream LLMs that the text came from ASR and may contain transcription artifacts (homophones, proper-noun misspellings).

The hook runs before the [trailing space](#trailing-space) is applied, so it can't strip it — use `append_trailing_space`.

Other patterns:

Archive transcriptions to a log, leave text unchanged (observer-only):

```jsonc
{ "post_transcription_hook": "tee -a ~/.local/share/hyprwhspr/log.txt >/dev/null" }
```

User-provided transform script on `$PATH`:

```jsonc
{ "post_transcription_hook": "~/.local/bin/filler-word-coach" }
```

Two environment variables are exported to the hook:

- `HYPRWHSPR_MODEL` — the active whisper model
- `HYPRWHSPR_BACKEND` — the active transcription backend

The hook runs under a 5-second timeout. Exit status `77` is reserved for an intentional consume result; stdout is ignored and the transcription is not pasted. On timeout, any other non-zero exit, or any subprocess error, the original text is preserved — a broken hook will never silently eat a dictation. Errors are logged to the service journal.

Note: the command runs under `shell=True`, so pipes, redirects, and command chaining work as expected. Treat `post_transcription_hook` as trusted config (same threat model as the rest of `config.json`).
