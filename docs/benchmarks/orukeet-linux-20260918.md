# Paired Orukeet / Parakeet check on Linux ARM

This is a check of the existing `OnnxAsrBackend` at `72207dbaac0008bb5d1537e4a0c24c0454e6ff0e`, requested during the optional Orukeet review. It changes no runtime code or default model.

## Environment and method

- Linux 6.8 aarch64 in an existing Colima VM on Apple M5 Max, 4 vCPUs and 8 GiB. CPU execution, onnx-asr 0.12.0 and ONNX Runtime 1.30.0, default ORT session/thread settings. This is not an x86 or bare-metal measurement.
- Both models initialized through the unchanged production backend: `orukeet` and `nemo-parakeet-tdt-0.6b-v3`, both INT8. The official required Orukeet manifest verifies its cached files. Stock uses pinned snapshot `8f23f0c03c8761650bdb5b40aaf3e40d2c15f1ce`.
- 640 previously selected English clips, 64 from each of 10 locally cached public corpora, 65.8 minutes total. Selection predates this run: SHA-256 order of `orukeet-app-eval-v1:dataset:row`, with nonempty references and duration below 120 seconds. Every selected clip and every empty output is retained. LibriSpeech test-other was used to train and select the released checkpoint; overlap for the remaining clips has not been resolved (see the review qualification below).
- Common preprocessing before timing: stereo channel mean, then soxr conversion to 16 kHz float32 mono. Both models receive the identical array. The backend enables Silero VAD at its unchanged 30-second threshold; the selected clips are all shorter than 25 seconds.
- One persistent instance of each backend in the same process, one unmeasured warm-up per model, then sequential paired calls with alternating order. Timings include the production `transcribe` call and result handling, excluding input preparation, model load, and warm-up.
- All measured inference runs inside a Linux network namespace with no network route. A separate probe returned `ENETUNREACH`; no inference upload or accounting-only request occurs.
- Whisper English normalization 0.1.15 plus Levenshtein word errors, identically for both models. Sanitized input IDs/hashes, counts, timings, package versions and model hashes are in the [receipt](orukeet-linux-20260918.json).

## Results

| Metric | Stock Parakeet v3 INT8 | Orukeet INT8 |
| --- | ---: | ---: |
| Word errors / 9,783 reference words | 845 | 800 |
| Pooled WER | 8.637% | 8.177% |
| Warm weighted real-time factor | 0.05744 | 0.04456 |
| Warm median call | 307.35 ms | 240.43 ms |
| One measured model initialization | 1.509 s | 2.195 s |
| Backend exceptions | 0 | 0 |
| Empty outputs (scored as returned) | 8 | 5 |

Orukeet has 45 fewer word errors on this sample, a 0.460 percentage-point difference. A 10,000-resample paired bootstrap stratified by corpus gives a 95% interval of [-0.863, -0.070] percentage points for Orukeet minus stock. This describes this sample; it does not establish a general accuracy advantage.

| Corpus (64 clips each) | Stock WER | Orukeet WER |
| --- | ---: | ---: |
| ami | 19.149% | 15.957% |
| common_voice | 11.628% | 11.628% |
| earnings22 | 15.030% | 14.162% |
| gigaspeech | 9.813% | 9.468% |
| l2_arctic | 6.879% | 6.040% |
| librispeech_test.clean | 1.829% | 2.265% |
| librispeech_test.other | 4.863% | 4.522% |
| speechocean_test | 26.392% | 21.065% |
| spgispeech | 4.871% | 4.668% |
| voxpopuli | 7.095% | 7.631% |

Orukeet improves 7 corpora, regresses on LibriSpeech test-clean and VoxPopuli, and ties on Common Voice. Its initialization was slower. VM scheduling and this one paired run limit the timing claim; these numbers do not establish performance on low-end or x86 CPUs.

Both backends also passed identical repeated speech, empty 0.1/1/5-second silence, a cold offline reload with identical text, and a 31-second padded input that exercises the actual Silero VAD route. The latter is a routing check, not long-recording accuracy. No microphone, compositor, hotkey, or paste behavior was tested.

These absolute WER/latency values should not be compared directly with the earlier OpenWhispr/sherpa results. That experiment used split ONNX graphs, sherpa-onnx, production FFmpeg normalization, 15-second segmentation, IPC, and normalizer 0.1.12. This run uses combined graphs and the direct hyprwhspr/onnx-asr path with normalizer 0.1.15. Both experiments have the same 9,783-word denominator, but we have not isolated how much each pipeline difference contributes.

## Review qualification (September 26, 2026)

The [pinned conversion record](https://huggingface.co/oruk/orukeet/blob/eac739d754bb171287930e6e63386f5b88f8179e/onnx/combined-v0.1.0-int8/VOXTYPE-CONVERSION.json)
identifies this export as r3. Its [training provenance](https://github.com/Oruk-AI/orukeet/blob/main/docs/data-and-licenses.md#final-r3-continuation)
states that all 2,939 LibriSpeech test-other recordings were used for final
adaptation and checkpoint selection. This corpus is not independent test evidence.
SpeechOcean762 also appears in earlier training; membership of this sample's
`speechocean_test` clips in that training set remains unverified. Corpus names
alone establish neither overlap nor independence for the other rows.

Reaggregating the existing receipt gives the following sensitivity checks. These
are arithmetic on the submitted counts, not new inference runs or certified
held-out results. The original receipt and runner are retained unchanged.

| Included sample | Clips | Words | Stock errors / WER | Orukeet errors / WER |
| --- | ---: | ---: | ---: | ---: |
| Original sample | 640 | 9,783 | 845 / 8.637% | 800 / 8.177% |
| Excluding LibriSpeech test-other | 576 | 8,611 | 788 / 9.151% | 747 / 8.675% |
| Also excluding SpeechOcean as a sensitivity check | 512 | 8,198 | 679 / 8.283% | 660 / 8.051% |

The original bootstrap interval applies only to the original sample; it does not
account for training contamination or speaker dependence. No independent-sample
confidence claim follows from these exclusions.

The [pinned notice](https://huggingface.co/oruk/orukeet/blob/eac739d754bb171287930e6e63386f5b88f8179e/onnx/combined-v0.1.0-int8/NOTICE.md#onnx-depthwise-execution)
also records a depthwise-convolution graph optimization. The timing comparison
measures the shipped exports, not an isolated effect of fine-tuning. It remains
useful deployment evidence for this ARM VM, with slower measured initialization.

### Outstanding evidence for optional inclusion

To reconstruct the exact sample, the contributor needs to supply each corpus's
repository or source URL, immutable revision, configuration, split, stable clip
identifiers, and the mapping from receipt row indices to those identifiers.
Include the selection and audio-preparation script (including filtering order,
hash ordering and tie handling), a reproducible reference-normalization recipe,
and clip-level training/selection-overlap accounting. References and audio can
remain at their original providers; these metadata and instructions must allow
an authorized reviewer to rebuild the manifest and match its audio hashes.
The receipt currently lacks these source identities, so exact replay remains
blocked on contributor input. Do not infer revisions from today's dataset contents.

### Separate gate for CPU-default promotion

Parakeet remains the default. Optional support does not approve a default switch.
Before proposing promotion, collect the following evidence through the production
backend with pinned model/runtime versions:

- Repeated paired Linux x86 runs on a low-power Intel CPU and an AMD laptop CPU,
  recording CPU identity, actual session providers/thread settings, p50/p95 call
  latency, weighted RTF, peak memory, cached initialization and first-call latency.
- Independent held-out accuracy across supported languages and realistic dictation:
  short utterances, names/numbers, casing/punctuation, noise, and genuine speech
  beyond the VAD threshold. Report corpus/language regressions and overlap checks
  alongside pooled metrics; a padded short clip is only a VAD routing check.
- A separate setup change resolving model selection before acquisition, downloading
  and verifying the selected model and required VAD assets, reporting failures
  accurately, and proving a first transcription after setup with networking disabled.
  Cover interactive and automatic setup, preserved explicit choices, interrupted
  acquisition, partial/corrupt caches, and a documented Parakeet fallback. The current
  setup implementation prepares the selection and verifies offline initialization;
  a 2026-09-27 local smoke test also initialized both INT8 models and transcribed
  the bundled 24-second `share/assets/test.wav` through the production backend
  with networking disabled (`unshare -Urn`). Direct and forced Silero VAD paths
  produced nonempty transcripts using onnx-asr 0.12.0 and ONNX Runtime 1.30.0
  on x86_64 CPU. This is an offline functionality check, not accuracy validation
  or representative long-form/hardware coverage.

Review these results before deciding on promotion; this document does not claim
that those hardware, accuracy or setup checks have passed.

## Runner and input contract

The [recorded runner](orukeet-linux-20260918.py) is an experiment recipe, not an installed command. In an isolated directory place it beside `manifest.json`, `audio/`, `hub/`, and `hyprwhspr/lib/src/` from the source commit above. Each manifest clip needs `id`, `dataset`, `index`, `path` (relative to that directory), `sha256`, `reference`, and `audio_seconds`. Do not run it in a directory containing results you want to keep: it writes `receipt*.json` and `paired-results.jsonl` there.

Install the recorded versions in a temporary Python 3.12 environment:

```sh
python -m pip install "onnx-asr[cpu,hub]==0.12.0" "onnxruntime==1.30.0" \
  "numpy==2.5.3" "soundfile==0.14.0" "soxr==1.1.0" \
  "whisper-normalizer==0.1.15" "rapidfuzz==3.14.6"
```

Populate the real caches once before entering the offline namespace. Orukeet uses its checked-in verifier; the stock snapshot needs its `config.json` as well as the encoder, decoder/joint, and vocabulary. Silero needs its own genuine model cache. An initial fixture preflight exposed a missing stock config and stereo source files; both were corrected before the 640-clip run. No model weights were changed.

On Linux, `sudo unshare -n /absolute/path/to/venv/bin/python orukeet-linux-20260918.py` runs in a new network namespace. It does not change the host network. The recipe records the historical host description above; update that description when using a different machine.

This repository does not redistribute corpus audio, references, or returned transcripts. The receipt retains selected row IDs and audio hashes for audit. Replaying the exact sample requires access to the corresponding corpus records; it is not a bundled benchmark dataset.
