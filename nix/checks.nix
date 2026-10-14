{ pkgs, package, nixpkgs, nixosModule, system }:
let
  evaluate = settings: nixpkgs.lib.nixosSystem {
    inherit system;
    modules = [ nixosModule ({ ... }: {
      system.stateVersion = "26.05";
      users.users.alice.isNormalUser = true;
      users.users.bob.isNormalUser = true;
    }) settings ];
  };
  enabled = (evaluate { services.hyprwhspr = { enable = true; users = [ "alice" ]; }; }).config;
  disabled = (evaluate {}).config;
  empty = (evaluate { services.hyprwhspr.enable = true; }).config;
  unknown = (evaluate { services.hyprwhspr = { enable = true; users = [ "missing" ]; }; }).config;
  ownFailures = config: builtins.filter (a: !a.assertion && pkgs.lib.hasPrefix "services.hyprwhspr.users" a.message) config.assertions;
  service = enabled.systemd.user.services.hyprwhspr;
  probe = pkgs.writeText "hyprwhspr-runtime-probe.py" ''
    import os
    import shutil
    from pathlib import Path
    import _pywhispercpp
    import sounddevice, soundfile, soxr, evdev, pyudev, pulsectl, dbus, cairo
    import gi
    for command in ('pactl', 'paplay', 'arecord', 'awk', 'grep', 'sed'):
        assert shutil.which(command), command
    if host_bin := os.environ.get('TEST_HOST_BIN'):
        for command in ('pactl', 'systemctl'):
            assert shutil.which(command) == str(Path(host_bin, command)), command
    gi.require_version('Gtk', '4.0')
    gi.require_version('Gtk4LayerShell', '1.0')
    from gi.repository import Gtk, Gtk4LayerShell
    assert Gtk.get_major_version() == 4
    Gtk4LayerShell.get_major_version()
    from mic_osd.runner import MicOSDRunner
    assert MicOSDRunner._layer_shell_ld_preload().startswith('/nix/store/')
    Path(os.environ['XDG_RUNTIME_DIR'], 'probe-ok').touch()
  '';
  probeEditor = pkgs.writeShellScript "hyprwhspr-runtime-probe" ''
    exec ${package.python}/bin/python ${probe}
  '';
in {
  package = package;
  smoke = pkgs.runCommand "hyprwhspr-smoke" {
    nativeBuildInputs = [ package ];
  } ''
    export XDG_CONFIG_HOME="$TMPDIR/config"
    export XDG_DATA_HOME="$TMPDIR/data"
    export XDG_CACHE_HOME="$TMPDIR/cache"
    export XDG_RUNTIME_DIR="$TMPDIR/runtime"
    mkdir -p "$XDG_RUNTIME_DIR"
    hyprwhspr --help
    hyprwhspr --version | grep -F '${package.version}'
    hyprwhspr install status | grep Nix-managed
    command -v hyprwhspr-tray
    hyprwhspr config init
    hyprwhspr config show
    test -f "$XDG_CONFIG_HOME/hyprwhspr/config.json"
    test ! -e "$XDG_DATA_HOME/hyprwhspr/venv"
    if hyprwhspr setup > "$TMPDIR/setup-output" 2>&1; then
      echo 'Nix setup unexpectedly succeeded' >&2
      exit 1
    fi
    grep 'Nix manages installation' "$TMPDIR/setup-output"
    setBackend() {
      ${package.python}/bin/python -c 'import json, sys
    path, backend = sys.argv[1:]
    with open(path) as f: config = json.load(f)
    config["transcription_backend"] = backend
    with open(path, "w") as f: json.dump(config, f)' "$XDG_CONFIG_HOME/hyprwhspr/config.json" "$1"
    }
    # Unsupported backends stay inspectable but cannot fetch models.
    setBackend nvidia
    hyprwhspr model list
    if hyprwhspr model download base > "$TMPDIR/download-output" 2>&1; then
      echo 'Nix model download accepted an unsupported backend' >&2
      exit 1
    fi
    grep 'CPU transcription' "$TMPDIR/download-output"
    setBackend pywhispercpp

    # Use an editor subprocess as a harmless way to inspect the actual wrapper
    # environment inherited by children, including the OSD. No display needed.
    export EDITOR=${probeEditor}
    PATH= ${package}/bin/hyprwhspr config edit
    test -f "$XDG_RUNTIME_DIR/probe-ok"
    # Prefer host clients when available, but never invoke the audio server or
    # service manager. The probe only checks executable discovery.
    export TEST_HOST_BIN="$TMPDIR/host-bin"
    mkdir -p "$TEST_HOST_BIN"
    for command in pactl systemctl; do
      printf '#!${pkgs.bash}/bin/bash\nexit 99\n' > "$TEST_HOST_BIN/$command"
      chmod +x "$TEST_HOST_BIN/$command"
    done
    PATH="$TEST_HOST_BIN" ${package}/bin/hyprwhspr config edit
    touch "$out"
  '';
  module = assert service.unitConfig.ConditionUser == [ "|alice" ];
    assert service.unitConfig.ConditionPathExists == "%E/hyprwhspr/config.json";
    assert service.serviceConfig.ExecStart == "${package}/bin/hyprwhspr";
    assert pkgs.lib.hasInfix "/run/current-system/sw/bin" service.environment.PATH;
    assert pkgs.lib.hasInfix "%h/.nix-profile/bin" service.environment.PATH;
    assert pkgs.lib.hasPrefix "${pkgs.bash}/bin/bash -c " service.serviceConfig.ExecStartPre;
    assert pkgs.lib.hasPrefix "${pkgs.bash}/bin/bash -c " service.serviceConfig.ExecStopPost;
    assert builtins.elem "input" enabled.users.users.alice.extraGroups;
    assert !(builtins.elem "input" enabled.users.users.bob.extraGroups);
    assert builtins.elem "uinput" enabled.boot.kernelModules;
    assert pkgs.lib.hasInfix ''KERNEL=="uinput", GROUP="input", MODE="0660"'' enabled.services.udev.extraRules;
    assert !(disabled.systemd.user.services ? hyprwhspr);
    assert !(builtins.elem package disabled.environment.systemPackages);
    assert ownFailures enabled == [];
    assert ownFailures empty != [];
    assert ownFailures unknown != [];
    pkgs.runCommand "hyprwhspr-module-check" {} ''
      # Materialize the rendered unit too, so systemd option serialization is checked.
      cp ${enabled.systemd.user.units."hyprwhspr.service".unit}/hyprwhspr.service "$out"
    '';
}
