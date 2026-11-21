"""Language selection and shared batch/stream spoken-command regressions."""
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'lib' / 'src'))

from text_processing import ProcessingContext, preprocess_text, resolve_processing_context
from processing_trace import build_processing_trace
from text_injector import InjectionOutcome
from tests.text_injector_helpers import ConfigStub, make_injector


class ProcessingLanguageTests(unittest.TestCase):
    def test_language_and_translation_matrix(self):
        for backend in ('pywhispercpp', 'local', 'cpu', 'nvidia', 'amd', 'vulkan',
                        'faster-whisper', 'rest-api', 'remote', 'realtime-ws',
                        'onnx-asr', 'parakeet-cpp', 'cohere-transcribe', 'qwen3-asr'):
            for task in ('transcribe', 'translate'):
                with self.subTest(backend=backend, task=task):
                    config = ConfigStub({'language': ' FR ', 'task': task,
                                         'transcription_backend': backend})
                    translated = task == 'translate' and backend in (
                        'pywhispercpp', 'local', 'cpu', 'nvidia', 'amd', 'vulkan', 'faster-whisper')
                    context = resolve_processing_context(config)
                    self.assertEqual(context.output_language, 'en' if translated else 'fr')
                    self.assertEqual(preprocess_text('one plus two', config),
                                     'one + two' if translated else 'one plus two')

    def test_override_and_snapshot(self):
        for configured, override, raw, expected in (
            ('en', 'fr', 'prêt point d’interrogation', 'prêt ?'),
            ('fr', 'en', 'ready question mark', 'ready?'),
            ('en', 'de', 'one plus two', 'one plus two'),
        ):
            with self.subTest(configured=configured, override=override):
                config = ConfigStub({'language': configured})
                context = resolve_processing_context(config, override)
                config.settings['language'] = 'ja'
                self.assertEqual(preprocess_text(raw, config, context=context), expected)
                trace = build_processing_trace(raw, config, context=context)
                self.assertEqual(trace['preprocessed'], expected)
                self.assertEqual(trace['output_language'], override)
                self.assertEqual(trace['spoken_command_language'], override if override in ('en', 'fr') else None)

    def test_unknown_language_fallback_and_disabled_profiles(self):
        for language in (None, '', 'en', 'fr', 'de', 'ja'):
            for enabled in (False, True):
                with self.subTest(language=language, enabled=enabled):
                    config = ConfigStub({'language': language, 'symbol_replacements': enabled,
                                         'word_overrides': {'nom': 'NAME'}})
                    trace = build_processing_trace('nom plus nom', config)
                    english = enabled and language in (None, '', 'en')
                    self.assertEqual(trace['preprocessed'], 'NAME + NAME' if english else 'NAME plus NAME')
                    self.assertEqual(trace['spoken_command_language'],
                                     ('en' if english else 'fr') if enabled and language in (None, '', 'en', 'fr') else None)

    def test_region_codes_use_base_language_commands(self):
        for language, raw, expected in (
            ('fr-FR', 'prêt point d’interrogation', 'prêt ?'),
            ('fr_CA', 'prêt point d’interrogation', 'prêt ?'),
            ('en-US', 'ready question mark', 'ready?'),
            ('de-DE', 'one plus two', 'one plus two'),
        ):
            with self.subTest(language=language):
                config = ConfigStub({'language': language})
                self.assertEqual(preprocess_text(raw, config), expected)

    def test_english_command_absorbs_recognizer_punctuation_before_it(self):
        for raw, expected in (
            ('Really? question mark', 'Really?'),
            ("That's it. period", "That's it."),
            ('a - dash b', 'a - - b'),
        ):
            with self.subTest(raw=raw):
                self.assertEqual(preprocess_text(raw), expected)

    def test_french_prose_and_command_word_boundaries(self):
        config = ConfigStub({'language': 'fr'})
        for raw in ('Il marque deux points.', 'Je suis à la ligne de départ.',
                    'un point de départ', 'plus de temps', 'une virgule',
                    'une arobase', 'la barre verticale', 'retour à la lignée',
                    'insérer un pointage', 'les deux points sont importants'):
            with self.subTest(raw=raw):
                self.assertEqual(preprocess_text(raw, config), raw)
        self.assertEqual(preprocess_text('a new liner'), 'a new liner')

    def test_french_recognizer_punctuation_and_aliases(self):
        config = ConfigStub({'language': 'fr'})
        for raw, expected in (
            ('Prêt. Insérer un point.', 'Prêt.'),
            ('Oui, insérer une virgule, ensuite', 'Oui, ensuite'),
            ('Titre: insérer deux points: suite', 'Titre : suite'),
            ('Attends… points de suspension… ensuite', 'Attends… ensuite'),
            ('Titre retour a la ligne. Tiret espace. lire', 'Titre\n- lire'),
            ('Un point d’interrogation', 'Un ?'),
            ('Un point d\'interrogation', 'Un ?'),
            ('TIRET DU SIX ESPACE.', '- '),
            ('attends... vraiment!!!', 'attends... vraiment!!!'),
            ('ATTENDS POINTS DE SUSPENSION.', 'ATTENDS…'),
            ('Attends... points de suspension ensuite', 'Attends… ensuite'),
            ('Prêt. insérer un point', 'Prêt.'),
            ('Ok ? point d’interrogation', 'Ok ?'),
        ):
            with self.subTest(raw=raw):
                self.assertEqual(preprocess_text(raw, config), expected)

    def test_deliberate_space_survives_hooks_without_duplication(self):
        for setting in ('auto', False, True):
            for stdout in ('', '- ', 'changed'):
                with self.subTest(setting=setting, stdout=stdout):
                    injector = make_injector()
                    injector.config_manager = ConfigStub({
                        'language': 'fr', 'append_trailing_space': setting,
                        'post_transcription_hook': 'test-hook',
                    })
                    with mock.patch('text_injector.subprocess.run', return_value=mock.Mock(
                            returncode=0, stdout=stdout, stderr='')), mock.patch.object(
                            injector, '_inject_via_clipboard_and_hotkey', return_value=True) as paste:
                        self.assertEqual(injector.inject_text('tiret espace'), InjectionOutcome.INJECTED)
                    expected = 'changed' + (' ' if setting is not False else '') if stdout == 'changed' else '- '
                    paste.assert_called_once_with(expected, retain=True)


class FrenchStreamTests(unittest.TestCase):
    def _stream(self, chunks, settings=None, context=None, cancel=False):
        injector = make_injector()
        injector.config_manager = ConfigStub({'language': 'fr', 'append_trailing_space': False, **(settings or {})})
        handle = injector.begin_stream(context=context)
        pasted = []
        with mock.patch.object(injector, '_inject_via_clipboard_and_hotkey',
                               side_effect=lambda text, **kwargs: pasted.append(text) or True), \
                mock.patch.object(injector, '_send_enter_if_auto_submit'):
            for chunk in chunks:
                injector.inject_stream_chunk(handle, chunk)
            injector.end_stream(handle, submit=not cancel)
        return ''.join(pasted), injector._last_text

    def test_every_french_alias_at_every_word_boundary(self):
        profile = ProcessingContext('fr').profile
        for command in profile.phrases:
            words = command.split()
            raw = f'avant {command} après'
            expected = preprocess_text(raw, context=ProcessingContext('fr'))
            # Include boundaries immediately before and after the command.
            for split in range(len(words) + 1):
                for variant in (str.lower, str.upper):
                    with self.subTest(command=command, split=split, case=variant.__name__):
                        chunks = ['avant ' + variant(' '.join(words[:split])),
                                  variant(' '.join(words[split:])), 'après']
                        pasted, retained = self._stream(chunks)
                        self.assertEqual(pasted, expected)
                        self.assertEqual(retained, expected)

    def test_list_spacing_and_line_breaks(self):
        for setting in ('auto', False, True):
            for chunks, expected in (
                (['tiret', 'espace'], '- '),
                (['titre retour à', 'la ligne tiret', 'espace lire'], 'titre\n- lire'),
                (['titre retour à la ligne'], 'titre'),
            ):
                with self.subTest(setting=setting, chunks=chunks):
                    actual, retained = self._stream(chunks, {'append_trailing_space': setting})
                    if setting is not False and not expected.endswith(' '):
                        expected += ' '
                    self.assertEqual(actual, expected)
                    self.assertEqual(retained, expected)

    def test_stream_uses_snapshot_for_delivery_and_recovery(self):
        actual, retained = self._stream(['ready question', 'mark'], context=ProcessingContext('en'))
        self.assertEqual(actual, 'ready?')
        self.assertEqual(retained, 'ready?')
        actual, retained = self._stream(['titre', 'insérer deux'], cancel=True)
        self.assertEqual(actual, 'titre')
        self.assertEqual(retained, 'titre insérer deux')

    def test_overrides_keep_precedence_when_overlapping_a_command(self):
        for settings, expected in (
            ({'word_overrides': {'point suivant': 'étape'}}, 'insérer un étape'),
            ({'filter_filler_words': True, 'filler_words': ['point suivant']}, 'insérer un'),
        ):
            actual, retained = self._stream(['insérer un point', 'suivant'], settings)
            self.assertEqual(actual, expected)
            self.assertEqual(retained, expected)

    def test_disabled_or_unsupported_profiles_do_not_buffer_commands(self):
        for settings in ({'language': 'de'}, {'symbol_replacements': False}):
            actual, retained = self._stream(['retour à', 'la ligne'], settings)
            self.assertEqual(actual, 'retour à la ligne')
            self.assertEqual(retained, actual)


if __name__ == '__main__':
    unittest.main()
