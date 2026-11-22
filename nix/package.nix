{ lib, stdenvNoCC, pkgs, callPackage, python313, makeWrapper, wrapGAppsHook4,
  gobject-introspection, gtk4, gtk4-layer-shell, glib, wl-clipboard, wtype,
  ydotool, xclip, xdotool, xprop, libnotify, systemd, coreutils, procps,
  bash, gawk, gnugrep, gnused, pulseaudio, alsa-utils,
  version ? "0-unstable-local", extraBackends ? [ ],
  llama-cpp, llama-cpp-vulkan }:
let
  registry = import ./backends.nix { inherit pkgs llama-cpp llama-cpp-vulkan; };
  unknown = builtins.filter (name: !builtins.hasAttr name registry) extraBackends;
  selected =
    if unknown != [ ] then
      throw "Unknown hyprwhspr extraBackends: ${toString unknown}. Known: ${toString (builtins.attrNames registry)}."
    else map (name: registry.${name}) extraBackends;
  backendNames = [ "cpu" "pywhispercpp" ] ++ lib.concatMap (spec: spec.names) selected;
  extraPython = lib.concatMap (spec: spec.pythonPackages) selected;
  wrapperBackendArgs =
    [ "--set" "HYPRWHSPR_NIX_BACKENDS" (lib.concatStringsSep "," backendNames) ]
    ++ lib.concatLists (map (spec: spec.wrapperArgs spec.runtime) selected);
  manifests = map (name: ../. + "/${name}") (builtins.filter
    (name: lib.hasPrefix "requirements" name && lib.hasSuffix ".txt" name)
    (builtins.attrNames (builtins.readDir ../.)));
  pywhispercpp = callPackage ./pywhispercpp.nix { python3Packages = python313.pkgs; };
  python = python313.withPackages (ps: (with ps; [
    sounddevice numpy soxr soundfile evdev pyperclip pyudev pulsectl rich
    jsonschema dbus-python pygobject3 pycairo pywhispercpp
  ]) ++ extraPython);
  # Fallbacks only: they must match the host's running systemd and audio server.
  hostTools = lib.makeBinPath [ systemd pulseaudio ];
in stdenvNoCC.mkDerivation {
  pname = "hyprwhspr";
  inherit version;
  src = lib.fileset.toSource {
    root = ../.;
    fileset = lib.fileset.difference
      (lib.fileset.unions ([ ../bin ../lib ../config ../share ../LICENSE ] ++ manifests))
      (lib.fileset.fileFilter (file: file.hasExt "pyc" || file.hasExt "pyo") ../lib);
  };
  nativeBuildInputs = [ makeWrapper wrapGAppsHook4 gobject-introspection ];
  buildInputs = [ gtk4 gtk4-layer-shell glib ];
  dontBuild = true;
  dontWrapGApps = true;
  installPhase = ''
    runHook preInstall
    mkdir -p "$out/lib/hyprwhspr" "$out/bin"
    cp -r bin lib config share LICENSE requirements*.txt "$out/lib/hyprwhspr/"
    echo '${builtins.toJSON { inherit version; }}' > "$out/lib/hyprwhspr/release.json"
    runHook postInstall
  '';
  preFixup = ''
    makeWrapper "$out/lib/hyprwhspr/bin/hyprwhspr" "$out/bin/hyprwhspr" \
      "''${gappsWrapperArgs[@]}" \
      --set HYPRWHSPR_INSTALL_METHOD nix \
      --set HYPRWHSPR_NIX_ROOT "$out/lib/hyprwhspr" \
      --set HYPRWHSPR_BACKEND_ENV ${python} \
      --unset HYPRWHSPR_GENERATION \
      --unset PYTHONPATH \
      --prefix PATH : ${lib.makeBinPath [
        python bash coreutils procps wl-clipboard wtype ydotool xclip xdotool
        xprop libnotify glib alsa-utils gawk gnugrep gnused
      ]} \
      --suffix PATH : ${hostTools} \
      ${lib.escapeShellArgs wrapperBackendArgs}
    # Integrations call this by name; store paths vanish after garbage collection.
    makeWrapper "$out/lib/hyprwhspr/config/hyprland/hyprwhspr-tray.sh" "$out/bin/hyprwhspr-tray" \
      --set HYPRWHSPR_ROOT "$out/lib/hyprwhspr" \
      --set HYPRWHSPR_INSTALL_METHOD nix \
      --set HYPRWHSPR_BACKEND_ENV ${python} \
      --prefix PATH : "$out/bin:${lib.makeBinPath [
        python bash coreutils procps libnotify glib alsa-utils gawk gnugrep gnused
      ]}" \
      --suffix PATH : ${hostTools} \
      ${lib.escapeShellArgs wrapperBackendArgs}
  '';
  passthru = {
    inherit python pywhispercpp;
    nixBackends = backendNames;
  } // lib.optionalAttrs (lib.elem "qwen3-asr" extraBackends) {
    qwen3AsrRuntime = registry.qwen3-asr.runtime;
  };
  meta = {
    description = if lib.elem "qwen3-asr" extraBackends then
      "Linux desktop speech-to-text with Whisper.cpp CPU transcription and a Qwen3-ASR sidecar (Nix beta)"
    else
      "Linux desktop speech-to-text with Whisper.cpp CPU transcription (Nix beta)";
    homepage = "https://github.com/goodroot/hyprwhspr";
    license = lib.licenses.mit;
    mainProgram = "hyprwhspr";
    platforms = [ "x86_64-linux" ];
  };
}
