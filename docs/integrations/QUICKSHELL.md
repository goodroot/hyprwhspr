# Quickshell

Add a microphone button to a custom [Quickshell](https://quickshell.org/)
bar with the [QML components](https://github.com/goodroot/hyprwhspr/tree/main/config/quickshell).
The button shows whether hyprwhspr is ready, recording, stopped, unloaded, or
reporting a problem. If hyprwhspr's recording indicator provides a processing
signal, the button also shows when transcription is in progress.

## Add the button

Copy the three files from `config/quickshell/` in the hyprwhspr checkout into
a `hyprwhspr/` directory next to your Quickshell `shell.qml`. Import that
directory and instantiate one status monitor at the shell level. Pass the
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

The monitor uses the same tray helper as [Waybar](WAYBAR.md) and
[Noctalia](NOCTALIA.md). It looks under `HYPRWHSPR_ROOT` or
`/usr/lib/hyprwhspr` by default. If your installation lives elsewhere,
set `installRoot` to the directory containing `config/hyprland/hyprwhspr-tray.sh`.

Left-click starts or stops recording. Right-click restarts the service.
Hover over the button to see details. You can set `readyColor`,
`inactiveColor`, `recordingColor`, `processingColor`, and `errorColor` on the
button to match your bar.

If you already use Quickshell's system tray, you can also use the
[built-in tray icon](../usage/FEEDBACK.md#tray-icon).
