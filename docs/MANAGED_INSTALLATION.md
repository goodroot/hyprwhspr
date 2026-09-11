# Managed installations

Verified releases for Debian/Ubuntu, Fedora and openSUSE. AUR updates stay with
your package manager; development checkouts keep their existing workflow.

## Commands

```sh
hyprwhspr update
hyprwhspr update --version v1.2.3
hyprwhspr install status
hyprwhspr install repair
hyprwhspr install repair --python /absolute/path/to/python
hyprwhspr uninstall
hyprwhspr uninstall --purge
hyprwhspr uninstall --purge --keep-models
```

Updates reuse models and skip onboarding and host dependency installation.
`update --version` can select an older supported release; settings must validate
before activation. There is no retained instant rollback command.
`install auto` remains an alias for automated setup.

## Python and storage

Python 3.11–3.14 is required. Updates retain the recorded interpreter, falling back
to a compatible system Python if it disappears. `update --python PATH` selects a
replacement explicitly. Paths from uv, mise and pyenv work without activation;
hyprwhspr never installs Python or changes manager settings.

Application releases and private CLI/backend environments live under
`$XDG_DATA_HOME/hyprwhspr`. Receipts and recovery state live under
`$XDG_STATE_HOME/hyprwhspr`. Both default to the usual `~/.local` locations.
Settings, credentials, models and IPC keep their existing locations.

The configuration directory is pinned at installation. To change it, run
`install repair` from a session with the desired `XDG_CONFIG_HOME`.

Updates temporarily need space for both environments, downloads and build files.
The 256 MiB staging minimum is not a build-size estimate; GPU builds need more.
Unreferenced generations are removed after success. Files still in use are queued
for later cleanup. Shared pip, uv and model caches are preserved.

Distro packages supply native libraries and desktop bindings. Host package changes
are separate from application rollback.

## Recovery

Stop a manually launched daemon before updating. Managed services stop gracefully;
only a previously running service restarts. Activation checks 15 seconds of stable
startup within a 60-second window, not transcription accuracy.

Failed activation restores the previous generation. Interrupted operations recover
on the next lifecycle command. `install status` reports pending cleanup, restoration
jobs and retained recovery files without changing them.

- Broken environment or payload: run `install repair`.
- Backend rebuild: run `backend repair`; it uses the local payload, but Python
  dependencies may need downloading.
- No usable Python or launcher: rerun bootstrap with `--repair --python PATH`.
- Cleanup failure: correct the reported filesystem permissions, then retry repair.
  A new cleanup failure returns nonzero even when activation succeeded; subsequent
  commands warn and retry previously recorded failures.

Unreadable state is reported and preserved where possible. Unknown ownership or
recovery references preserve files rather than authorizing deletion. Inspect paths
listed by `install status` before deleting recovery evidence; it is not expired
automatically. Failure to save recovery state retains the transaction journal.

## Migration and removal

Migration recognizes the old `~/.local/share/hyprwhspr/src` bootstrap checkout and
builds replacement environments. It removes the checkout only after activation,
with the expected upstream, no local changes or local-only commits, and no remaining
desktop references. Python bytecode is disposable; uncertain content is preserved.

Recognized services, bindings and bar integrations move to stable launcher paths.
Customized files are preserved with guidance. Restart bars still using an old
configuration, reconcile reported references, then retry cleanup.

**Uninstall preserves settings, credentials and models.** `--purge` removes only
recorded personal files; `--keep-models` overrides model removal. Shared model
directories and unrecorded or external models remain.

Modified integrations and customized services are preserved. Remove reported bar
references before retrying integration removal. Customized services retain their
runtime and launcher until reconciled. Receipts remain for cleanup retries.

`--remove-permissions` removes only recorded group memberships or rules that
installation added. `--skip-permissions` preserves them. Recognized pre-existing
udev rules are not treated as installation-owned.

## Release rollout

Fresh website bootstraps still use the legacy installer. Explicit `--version`,
`--python`, `--repair`, `HYPRWHSPR_RELEASE_BOOTSTRAP=1`, or an existing managed
installation select the release path.

Before changing that default, publish the first compatible application release
and test install, CPU/GPU selection, update, interrupted recovery, interpreter
replacement, migration and uninstall in disposable distro desktop VMs. Container
smoke tests and mocked tests do not validate live Wayland/X11 service behavior.

Release archives and recovery helpers use HTTPS and SHA-256 checksums. Extraction
rejects unsafe archives. Checksums verify publisher-asset consistency, not an
independent signature. Backend-wheel and GUI-runtime releases are excluded.
