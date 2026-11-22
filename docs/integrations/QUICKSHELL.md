# Quickshell

Add a microphone button to a custom [Quickshell](https://quickshell.org/)
bar with the [QML components](https://github.com/goodroot/hyprwhspr/tree/main/config/quickshell).
The button shows service status, a microphone level meter during regular
dictation, and a spinner during transcription. It works with the floating
OSD disabled, so recording feedback can stay in your bar.

Left-click starts or stops recording. Right-click restarts the service.
Hover over the button for status details.

## Add the button

Use an existing Quickshell installation and a configured hyprwhspr service.
This is a manual integration; hyprwhspr setup does not add it to your shell.

Copy `HyprwhsprButton.qml`, `HyprwhsprStatus.qml`, and `microphone.svg` from
`config/quickshell/` into a `hyprwhspr/` directory next to your Quickshell
`shell.qml`. Import that directory and create one status monitor at the shell
level. Pass the
same monitor to each button if you have multiple bars or outputs.

```qml
import QtQuick
import Quickshell
import "hyprwhspr" as Dictation

ShellRoot {
    Dictation.HyprwhsprStatus { id: dictation }

    PanelWindow {
        anchors.top: true
        implicitWidth: 40
        implicitHeight: 32

        Dictation.HyprwhsprButton {
            anchors.centerIn: parent
            status: dictation
        }
    }
}
```

### Locate the tray helper

The monitor uses the same tray helper as [Waybar](WAYBAR.md) and
[Noctalia](NOCTALIA.md). It looks under `HYPRWHSPR_ROOT` or
`/usr/lib/hyprwhspr` by default. If your installation lives elsewhere,
set `installRoot` to the directory containing `config/hyprland/hyprwhspr-tray.sh`.

```qml
Dictation.HyprwhsprStatus {
    id: dictation
    installRoot: "/absolute/path/to/hyprwhspr"
}
```

Replace the path with your installation or checkout directory. The default
`/usr/lib/hyprwhspr` is not the location used by managed or Nix installations.

## Appearance

Set `mic_osd_enabled: false` in [your hyprwhspr config](../CONFIGURATION.md)
to use the bar without the overlay or recording-status notification.
See [Feedback](../usage/FEEDBACK.md#bar-integrations) for the shared status signals.

Set `readyColor`, `inactiveColor`, `recordingColor`, `processingColor`, and `errorColor` on the
button to match your bar.

If you prefer a standard tray icon, Quickshell's system tray can show the
[built-in tray icon](../usage/FEEDBACK.md#tray-icon).
