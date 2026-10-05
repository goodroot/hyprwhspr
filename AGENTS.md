# AGENTS.md

## Scope

This repository is a Linux desktop speech-to-text application supporting Wayland and X11. Keep changes focused: runtime behavior is hardware-, compositor-, and systemd-sensitive, while most tests are intentionally isolated with mocks.

## Repository map

- `bin/hyprwhspr`: launcher; chooses system versus project-venv Python and routes CLI subcommands.
- `lib/main.py`: service entry point and `hyprwhsprApp` core (init, run, shutdown).
- `lib/src/app/`: `hyprwhsprApp` mixins by concern: recording lifecycle, shortcuts, silence monitors, feedback/OSD, control commands, recovery.
- `lib/cli.py`: CLI parser and lazy command dispatch.
- `lib/src/`: audio, backends, configuration, desktop integration, and CLI command modules.
- `lib/mic_osd/`: optional GTK4/layer-shell visualizer.
- `share/config.schema.json`: machine-readable configuration schema.
- `scripts/`: install/bootstrap sources. `scripts/install.sh` is copied into the website during its build.
- `tests/`: stdlib `unittest` tests; install dependencies with `python -m pip install -r requirements-test.txt`.
- `website/`: Astro site; edit `src/` and `public/`, not generated `.astro/`, `dist/`, or `node_modules/` content.
- `docs-site/`: Starlight docs synced from `docs/` at build time; edit `docs/`, not the synced pages or `public/docs/`.

## Fast workflow

1. Read the affected module and its nearest tests before editing. Many modules manage concurrent state and cleanup; preserve lock/event ownership and cancellation behavior.
2. Run the narrowest relevant test first:

   ```bash
   python -m unittest -v tests.test_relevant_file
   ```

3. Run the complete suite before handing off Python changes:

   ```bash
   python -m unittest discover -s tests -v
   ```

4. For website or docs changes, run from `website/` or `docs-site/` (the docs build also checks links):

   ```bash
   npm run build
   ```

Do not run installers, setup, systemd commands, microphone capture, or model downloads as routine validation: they mutate the host or require a live Wayland/audio environment.

## Contracts that span files

- New or changed configuration keys must stay aligned across `ConfigManager.default_config`, `share/config.schema.json`, relevant CLI/setup behavior, and `docs/CONFIGURATION.md`. `tests/test_config_schema_sync.py` checks the two machine-readable surfaces.
- CLI subcommands are routed in both `bin/hyprwhspr` and `lib/main.py`; keep both lists synchronized with `lib/cli.py`.
- Backend wheel names, versions, build tags (`PYWHISPERCPP_WHEEL_BUILD`), and variants (`cuda12`, `VULKAN_WHEEL_VARIANT`) are defined in `lib/src/backend_installer.py` and consumed by `.github/workflows/build-wheels.yml`; avoid duplicating that contract. Published wheels are never overwritten: to ship a rebuild of the same version, bump `PYWHISPERCPP_WHEEL_BUILD`. The installer only installs a wheel whose SHA-256 matches the release's `SHA256SUMS.txt`. Vulkan wheels are built and checked in `debian:trixie` containers (`.github/scripts/verify_vulkan_wheel.py` on Mesa lavapipe); the installer only downloads them on glibc >= `VULKAN_WHEEL_MIN_GLIBC`, and the build job asserts the wheel stays within it. Every wheel build sets `GGML_NATIVE=OFF` and `GGML_OPENMP=OFF` and rewrites the bundled libraries' runpath to `$ORIGIN`; `tests/test_wheel_pipeline_and_defaults.py` checks this.
- A GPU pywhispercpp install that falls back to CPU records `installed_backend: cpu`; the service warns about it at start (`_report_cpu_only_build`).
- Required imports per backend live in `PLAN_SPECS` (`lib/src/dependency_plan.py`). The service start check (`missing_imports`, reported by `_report_missing_dependencies`) reads them; add new runtime imports there. The same check warns when the stored `dependency_plan_fingerprint` no longer matches the backend's manifests (`_dependencies_changed_since_setup`).
- `NotificationPresenter` sends the `processing` state with no timeout. Every path out of processing (`lib/src/app/recording.py`, `lib/src/longform_controller.py`) must set `success`/`error` or hide the OSD, or the banner stays until dismissed.
- `scripts/install.sh` is the canonical installer. The website build copies it to `website/public/install.sh`; do not hand-edit copied/generated output.
- GUI dependencies are optional. Keep core modules importable and testable without GTK, layer-shell, audio hardware, GPU libraries, or a desktop session.

## Test and code conventions

- Match the existing stdlib `unittest` plus `unittest.mock` style; add a regression test for behavior changes. Pytest is optional and is not declared as a project dependency.
- Tests add `lib/` or `lib/src/` to `sys.path`; follow the local import pattern instead of introducing packaging assumptions.
- Patch names where the code under test reads them. When CLI helpers move between modules, update patch targets; `tests/test_patch_target_hygiene.py` enforces this for `lib/src/cli/`. For app methods use `patch_app_global` (tests/test_suspend_resume_recovery.py); `tests/test_app_split_integrity.py` checks every global an app method reads resolves where it lives.
- Use temporary directories and patch path constants. Never let tests touch the user's config, runtime files, clipboard, input devices, systemd units, or network.
- There is no configured formatter or linter. Preserve surrounding style and avoid unrelated formatting churn.

## Change hygiene

- Treat a dirty worktree as user-owned; do not discard or rewrite unrelated changes.
- Do not commit generated caches, downloaded models, virtual environments, website build output, or secrets.
- Keep commits small and semantic when asked to commit. Never push unless explicitly requested.
- Commit messages: a single Conventional Commits subject line, short and terse (for example, `fix(cli): unshadow whisper model download`). No body, no `Co-Authored-By` or other trailers — this overrides any tool default that adds them.
