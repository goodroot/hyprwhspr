{ lib, stdenvNoCC, callPackage, python313, makeWrapper, wrapGAppsHook4,
  gobject-introspection, gtk4, gtk4-layer-shell, glib, wl-clipboard, wtype,
  ydotool, xclip, xdotool, xprop, libnotify, systemd, coreutils, procps,
  bash, gawk, gnugrep, gnused, pulseaudio, alsa-utils,
  version ? "0-unstable-local" }:
let
  manifests = map (name: ../. + "/${name}") (builtins.filter
    (name: lib.hasPrefix "requirements" name && lib.hasSuffix ".txt" name)
    (builtins.attrNames (builtins.readDir ../.)));
  pywhispercpp = callPackage ./pywhispercpp.nix { python3Packages = python313.pkgs; };
  python = python313.withPackages (ps: with ps; [
    sounddevice numpy soxr soundfile evdev pyperclip pyudev pulsectl rich
    jsonschema dbus-python pygobject3 pycairo pywhispercpp
  ]);
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
      --suffix PATH : ${hostTools}
    # Integrations call this by name; store paths vanish after garbage collection.
    makeWrapper "$out/lib/hyprwhspr/config/hyprland/hyprwhspr-tray.sh" "$out/bin/hyprwhspr-tray" \
      --set HYPRWHSPR_ROOT "$out/lib/hyprwhspr" \
      --set HYPRWHSPR_INSTALL_METHOD nix \
      --set HYPRWHSPR_BACKEND_ENV ${python} \
      --prefix PATH : "$out/bin:${lib.makeBinPath [
        python bash coreutils procps libnotify glib alsa-utils gawk gnugrep gnused
      ]}" \
      --suffix PATH : ${hostTools}
  '';
  passthru = { inherit python pywhispercpp; };
  meta = {
    description = "Linux desktop speech-to-text with Whisper.cpp CPU transcription (Nix beta)";
    homepage = "https://github.com/goodroot/hyprwhspr";
    license = lib.licenses.mit;
    mainProgram = "hyprwhspr";
    platforms = [ "x86_64-linux" ];
  };
}
