import QtQuick
import Quickshell
import Quickshell.Io

Scope {
    id: root

    property string installRoot: Quickshell.env("HYPRWHSPR_ROOT") || "/usr/lib/hyprwhspr"
    property int refreshInterval: 2000
    readonly property bool busy: pendingAction !== "" || (runner.running && runner.action !== "status")
    readonly property string state: ["recording", "processing"].includes(visualizerState)
            && ["ready", "recording"].includes(result.state)
        ? visualizerState : result.state
    readonly property string tooltip: state === "processing"
        ? "hyprwhspr: Transcribing…" : result.tooltip

    property var result: ({ state: "loading", tooltip: "hyprwhspr: Waiting for status…" })
    property string pendingAction: ""
    readonly property string runtimeDirectory: Quickshell.env("XDG_RUNTIME_DIR")
    readonly property string visualizerState: visualizer.loaded ? visualizer.text().trim() : ""
    readonly property real audioLevel: state === "recording" && meter.loaded
        ? Math.min(1, Math.max(0, Number(meter.text().trim()) || 0)) : 0

    function toggleRecording() {
        request("record")
    }

    function restartService() {
        request("restart")
    }

    function request(action: string) {
        if (busy || state === "loading")
            return;
        // Let a pending status read finish before sending one click.
        if (runner.running)
            pendingAction = action;
        else
            run(action);
    }

    function run(action: string) {
        if (runner.running)
            return;
        runner.action = action;
        runner.running = true;
    }

    function runPendingAction() {
        // Keep polling and further clicks blocked until the queued click runs.
        const action = pendingAction;
        pendingAction = "";
        run(action);
    }

    function readStatus(output: string): var {
        try {
            const status = JSON.parse(output);
            if (["ready", "recording", "unloaded", "stopped", "error"].includes(status?.class)
                    && typeof status.tooltip === "string")
                return {
                    state: status.class,
                    tooltip: status.tooltip.replace(/\n_ts:\d+$/, "")
                };
        } catch (error) {
            // The helper can fail before it emits JSON.
        }
        return { state: "error", tooltip: "hyprwhspr: Invalid status from the tray helper" };
    }

    Process {
        id: runner

        property string action: "status"
        command: [
            "timeout", "--kill-after=1s", "8s",
            root.installRoot + "/config/hyprland/hyprwhspr-tray.sh", action
        ]
        stdout: StdioCollector { id: output }

        onExited: function(exitCode, exitStatus) {
            if (exitCode !== 0 || exitStatus !== 0) {
                const message = exitCode === 124 ? "Tray helper timed out"
                    : exitCode === 126 || exitCode === 127 ? "Cannot run tray helper. Check installRoot"
                    : "Tray helper failed (" + action + ")";
                root.result = { state: "error", tooltip: "hyprwhspr: " + message };
                root.pendingAction = "";
            } else if (action !== "status") {
                // Read daemon state after sending a command, including service start.
                Qt.callLater(root.run, "status");
            } else {
                root.result = root.readStatus(output.text);
                if (root.pendingAction !== "")
                    Qt.callLater(root.runPendingAction);
            }
        }
    }

    FileView {
        id: visualizer
        path: root.runtimeDirectory ? root.runtimeDirectory + "/hyprwhspr/visualizer_state" : ""
        watchChanges: path !== ""
        printErrors: false
        onFileChanged: reload()
    }

    FileView {
        id: meter
        path: root.state === "recording" && root.runtimeDirectory
            ? root.runtimeDirectory + "/hyprwhspr/audio_level" : ""
        printErrors: false
        blockLoading: false
    }

    Timer {
        interval: 100
        running: meter.path !== ""
        repeat: true
        onTriggered: meter.reload()
    }

    Timer {
        interval: root.refreshInterval
        running: true
        repeat: true
        onTriggered: {
            if (visualizer.path !== "")
                visualizer.reload();
            if (!root.busy)
                root.run("status");
        }
    }

    Component.onCompleted: run("status")
}
