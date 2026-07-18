# AGENTS.md

## Scope

This repository is a Linux/Wayland speech-to-text application. Keep changes focused: runtime behavior is hardware-, compositor-, and systemd-sensitive, while most tests are intentionally isolated with mocks.

## Repository map

- `bin/hyprwhspr`: launcher; chooses system versus project-venv Python and routes CLI subcommands.
- `lib/main.py`: long-running dictation service and recording state machine.
- `lib/cli.py`: CLI parser and lazy command dispatch.
- `lib/src/`: audio, backends, configuration, desktop integration, and CLI command modules.
- `lib/mic_osd/`: optional GTK4/layer-shell visualizer.
- `share/config.schema.json`: machine-readable configuration schema.
- `scripts/`: install/bootstrap sources. `scripts/install.sh` is copied into the website during its build.
- `tests/`: dependency-free stdlib `unittest` tests.
- `website/`: Astro site; edit `src/` and `public/`, not generated `.astro/`, `dist/`, or `node_modules/` content.

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

4. For website changes, run from `website/`:

   ```bash
   npm run build
   ```

Do not run installers, setup, systemd commands, microphone capture, or model downloads as routine validation: they mutate the host or require a live Wayland/audio environment.

## Contracts that span files

- New or changed configuration keys must stay aligned across `ConfigManager.default_config`, `share/config.schema.json`, relevant CLI/setup behavior, and `docs/CONFIGURATION.md`. `tests/test_config_schema_sync.py` checks the two machine-readable surfaces.
- CLI subcommands are routed in both `bin/hyprwhspr` and `lib/main.py`; keep both lists synchronized with `lib/cli.py`.
- Backend wheel names, versions, and variants are defined in `lib/src/backend_installer.py` and consumed by `.github/workflows/build-wheels.yml`; avoid duplicating that contract.
- `scripts/install.sh` is the canonical installer. The website build copies it to `website/public/install.sh`; do not hand-edit copied/generated output.
- GUI dependencies are optional. Keep core modules importable and testable without GTK, layer-shell, audio hardware, GPU libraries, or a desktop session.

## Test and code conventions

- Match the existing stdlib `unittest` plus `unittest.mock` style; add a regression test for behavior changes. Pytest is optional and is not declared as a project dependency.
- Tests add `lib/` or `lib/src/` to `sys.path`; follow the local import pattern instead of introducing packaging assumptions.
- Patch names where the code under test reads them. When CLI helpers move between modules, update patch targets; `tests/test_patch_target_hygiene.py` enforces this for `lib/src/cli/`.
- Use temporary directories and patch path constants. Never let tests touch the user's config, runtime files, clipboard, input devices, systemd units, or network.
- There is no configured formatter or linter. Preserve surrounding style and avoid unrelated formatting churn.

## Change hygiene

- Treat a dirty worktree as user-owned; do not discard or rewrite unrelated changes.
- Do not commit generated caches, downloaded models, virtual environments, website build output, or secrets.
- Keep commits small and semantic when asked to commit (for example, `docs: add agent workflow guide`). Never push unless explicitly requested.
