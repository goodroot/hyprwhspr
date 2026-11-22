import QtQuick
import QtQuick.Controls

Button {
    id: button

    required property HyprwhsprStatus status
    property color readyColor: palette.buttonText
    property color inactiveColor: palette.mid
    property color recordingColor: "#e45649"
    property color processingColor: "#d18c19"
    property color errorColor: recordingColor

    display: AbstractButton.IconOnly
    text: qsTr("Toggle hyprwhspr recording")
    icon.source: Qt.resolvedUrl("microphone.svg")
    icon.color: {
        switch (status.state) {
        case "recording": return recordingColor;
        case "processing": return processingColor;
        case "ready": return readyColor;
        case "error": return errorColor;
        default: return inactiveColor;
        }
    }

    hoverEnabled: true
    Accessible.description: status.tooltip
    ToolTip.visible: hovered
    ToolTip.text: status.tooltip

    onClicked: status.toggleRecording()

    TapHandler {
        acceptedButtons: Qt.RightButton
        onTapped: button.status.restartService()
    }
}
