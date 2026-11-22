pragma ComponentBehavior: Bound
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
    readonly property bool recording: status.state === "recording"
    readonly property bool processing: status.state === "processing"

    display: AbstractButton.IconOnly
    text: qsTr("Toggle hyprwhspr recording")
    icon.source: recording || processing ? "" : Qt.resolvedUrl("microphone.svg")
    implicitWidth: Math.max(36, implicitBackgroundWidth + leftInset + rightInset,
                            implicitContentWidth + leftPadding + rightPadding)
    implicitHeight: Math.max(28, implicitBackgroundHeight + topInset + bottomInset,
                             implicitContentHeight + topPadding + bottomPadding)
    icon.color: {
        switch (status.state) {
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

    Row {
        anchors.centerIn: parent
        height: 16
        spacing: 2
        visible: button.recording
        Repeater {
            model: [0.55, 0.78, 1, 0.85, 0.58]
            Rectangle {
                required property real modelData
                anchors.verticalCenter: parent.verticalCenter
                width: 2
                height: 2 + 14 * Math.sqrt(button.status.audioLevel) * modelData
                radius: 1
                color: button.recordingColor
                Behavior on height { NumberAnimation { duration: 90 } }
            }
        }
    }

    Item {
        anchors.centerIn: parent
        width: 14
        height: width
        visible: button.processing
        Rectangle {
            anchors.fill: parent
            radius: width / 2
            color: "transparent"
            border.color: button.processingColor
            opacity: 0.4
        }
        Rectangle {
            anchors.horizontalCenter: parent.horizontalCenter
            width: 4
            height: width
            radius: width / 2
            color: button.processingColor
        }
        RotationAnimation on rotation {
            from: 0
            to: 360
            duration: 850
            loops: Animation.Infinite
            running: button.processing
        }
    }

    TapHandler {
        acceptedButtons: Qt.RightButton
        onTapped: button.status.restartService()
    }
}
