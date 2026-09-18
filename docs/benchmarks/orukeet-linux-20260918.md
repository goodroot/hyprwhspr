# Paired Orukeet / Parakeet check on Linux ARM

This is a check of the existing `OnnxAsrBackend` at `72207dbaac0008bb5d1537e4a0c24c0454e6ff0e`, requested during the optional Orukeet review. It changes no runtime code or default model.

## Environment and method

- Linux 6.8 aarch64 in an existing Colima VM on Apple M5 Max, 4 vCPUs and 8 GiB. CPU execution, onnx-asr 0.12.0 and ONNX Runtime 1.30.0, default ORT session/thread settings. This is not an x86 or bare-metal measurement.
- Both models initialized through the unchanged production backend: `orukeet` and `nemo-parakeet-tdt-0.6b-v3`, both INT8. The official required Orukeet manifest verifies its cached files. Stock uses pinned snapshot `8f23f0c03c8761650bdb5b40aaf3e40d2c15f1ce`.
- 640 previously selected English clips, 64 from each of 10 locally cached public corpora, 65.8 minutes total. Selection predates this run: SHA-256 order of `orukeet-app-eval-v1:dataset:row`, with nonempty references and duration below 120 seconds. Every selected clip and every empty output is retained. Training overlap is unknown.
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
