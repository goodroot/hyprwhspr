# Nix and NixOS (beta)

Nix support is **beta**. We'd appreciate [issue reports](https://github.com/goodroot/hyprwhspr/issues)
with your NixOS/distro version, compositor, and steps to reproduce.

The flake provides an `x86_64-linux` package with Whisper.cpp **CPU transcription**,
desktop integration tools, and the GTK4 mic OSD. GPU backends, other transcription
backends, ARM packages, and a Home Manager module are not included in this version.
The OSD retains its usual compositor requirements; see [desktop integration](DESKTOPS.md).

## Build or install the package

Enable Nix's `nix-command` and `flakes` experimental features. From a checkout:

```bash
nix build
nix run . -- --help
```

Or install into a Nix profile:

```bash
nix profile install github:goodroot/hyprwhspr
```

The package supplies Python and its dependencies. It does not create a virtual
environment, install models at build time, or configure the host's audio server.
On other Linux distributions, the host provides the desktop session, audio
server, and input-device permissions; `hyprwhspr systemd install` writes a user
service that runs the profile's `hyprwhspr`. On NixOS, the module below manages
both.

## NixOS module

Add the flake input and import its module in your existing NixOS configuration:

```nix
{
  inputs.hyprwhspr.url = "github:goodroot/hyprwhspr";

  outputs = { nixpkgs, hyprwhspr, ... }: {
    nixosConfigurations.desktop = nixpkgs.lib.nixosSystem {
      system = "x86_64-linux";
      modules = [
        ./configuration.nix
        hyprwhspr.nixosModules.default
        {
          services.hyprwhspr = {
            enable = true;
            users = [ "alice" ];
          };
        }
      ];
    };
  };
}
```

Replace `desktop` and `alice` with your host and an existing user declared in
`users.users`. The module adds those users to the `input` group, loads `uinput`,
and declares its permission rule. Input-group membership allows reading input
devices and injecting keystrokes; list only users who should have that access.
Rebuild NixOS and log out and back in for the group change to apply.

`services.hyprwhspr.package` can override the default flake package. The module
does not enable PipeWire, replace your compositor, or start a separate ydotool
service: hyprwhspr manages its own private ydotoold process.

## First run

Create and edit your user configuration, then download a CPU-sized model:

```bash
hyprwhspr config init
hyprwhspr config edit
hyprwhspr model download base
```

Set `EDITOR` if your editor is not `nano`. New configurations already default to
the compatible `pywhispercpp` backend and `base` model. For an existing configuration,
set these fields while preserving your other settings:

```json
{
  "transcription_backend": "cpu",
  "model": "base"
}
```

Start the NixOS user service after configuration and model download:

```bash
systemctl --user start hyprwhspr
hyprwhspr systemd status
```

The module starts the service with `graphical-session.target` on later logins,
only for listed users and only when their configuration exists. Your compositor
must activate that target and import its display variables into the user service
manager. If your session does not do this, configure its startup hook to import
the variables it sets (for example `WAYLAND_DISPLAY`, `DISPLAY`,
`HYPRLAND_INSTANCE_SIGNATURE`, or `SWAYSOCK`) using
`systemctl --user import-environment`, then start the graphical session target.
See [desktop integration](DESKTOPS.md) for compositor-specific setup.

For a foreground run without the module, invoke `hyprwhspr` or `nix run .` in
your graphical session after completing the configuration and permissions setup.
The default recording shortcut is **Super+Alt+D**.

`hyprwhspr waybar install` and `hyprwhspr noctalia install` point at your profile's
`hyprwhspr-tray`, so they survive updates and garbage collection. For a Hyprland
keybinding, add `exec, hyprwhspr-tray record` to your configuration.

Configuration stays in `$XDG_CONFIG_HOME/hyprwhspr` (normally
`~/.config/hyprwhspr`). Whisper models stay in
`$XDG_DATA_HOME/pywhispercpp/models` (normally `~/.local/share/pywhispercpp/models`).
Models, credentials, and user settings are never placed in the Nix store.

## Updates, removal, and migration

Nix owns the application, Python dependencies, and declarative service.
`hyprwhspr setup`, `update`, `uninstall`, backend repair/reset, and state reset
explain the Nix workflow instead of changing the installation; on NixOS, so do
service install/enable/disable. Configuration, model commands, recording,
transcription, service status/restart, and `hyprwhspr install status` remain
available. With a backend other than CPU, model downloads are refused and the
service reports the setting at start.

For a NixOS installation, update the `hyprwhspr` input in your system flake and
rebuild. Disable `services.hyprwhspr.enable` and rebuild to remove the module's
service and permission declarations. For a profile installation, use
`nix profile list` to find its name, then `nix profile upgrade NAME` or
`nix profile remove NAME`. User configuration and models remain intact.

When migrating, stop the previous hyprwhspr service first. Inspect
`systemctl --user cat hyprwhspr` and move aside any old user-owned
`~/.config/systemd/user/hyprwhspr.service` and related overrides before enabling
the NixOS module: user units take precedence over the declarative unit. Reload
the user service manager after moving them. Check `command -v hyprwhspr` to
ensure an old launcher is not shadowing the Nix package. Migration does not
automatically remove your previous installation or rewrite backend settings.

## Development checks

```bash
nix flake check
nix build --no-link
```

These checks build the native CPU extension, verify packaged dependencies and OSD
library discovery, exercise configuration commands without a user venv, and
validate the NixOS service and permissions. They do not download models, capture
audio, or start a desktop service. Live transcription, paste, and OSD behavior
still require testing in a NixOS desktop session.
