#!/bin/bash

# hyprwhspr System Tray Status Script
# Shows hyprwhspr status in the Hyprland system tray with JSON output

PACKAGE_ROOT="/usr/lib/hyprwhspr"

ICON_PATH="$PACKAGE_ROOT/share/assets/hyprwhspr.png"

# Performance optimization: command caching
_now=$(date +%s%3N 2>/dev/null || date +%s)  # ms if available
declare -A _cache

# Cached command execution with timeout
cmd_cached() {
    local key="$1" ttl_ms="${2:-500}" cmd="${3}"; shift 3 || true
    local now=$(_date_ms)
    if [[ -n "${_cache[$key.time]:-}" && $((now - _cache[$key.time])) -lt $ttl_ms ]]; then
        printf '%s' "${_cache[$key.val]}"; return 0
    fi
    local out
    out=$(timeout 0.25s bash -c "$cmd" 2>/dev/null) || out=""
    _cache[$key.val]="$out"; _cache[$key.time]=$now
    printf '%s' "$out"
}

_date_ms(){ date +%s%3N 2>/dev/null || date +%s; }

# Tiny helper for fast, safe command execution
try() { timeout 0.2s bash -lc "$*" 2>/dev/null; }

# Function to check if hyprwhspr is running
is_hyprwhspr_running() {
    systemctl --user is-active --quiet hyprwhspr.service
}

# Function to check if ydotoold is running and working
is_ydotoold_running() {
    # Check if service is active
    if systemctl --user is-active --quiet ydotool.service; then
        # Test if ydotool actually works by using a simple command
        timeout 1s ydotool help > /dev/null 2>&1
        return $?
    fi
    return 1
}

# Function to check PipeWire health comprehensively
is_pipewire_ok() {
    timeout 0.2s pactl info >/dev/null 2>&1 || return 1
    pactl list short sources 2>/dev/null | grep -qE 'RUNNING|input' || return 1
    return 0
}

# Function to check if model file exists
model_exists() {
    local cfg="$HOME/.config/hyprwhspr/config.json"
    local model_path
    model_path=$(grep -oE '"model"\s*:\s*"[^"]+"' "$cfg" 2>/dev/null | cut -d\" -f4)
    [[ -n "$model_path" ]] || return 0  # use defaults; skip
    
    # If it's a short name like "base.en", resolve to pywhispercpp full path
    if [[ "$model_path" != /* ]]; then
        model_path="${XDG_DATA_HOME:-$HOME/.local/share}/pywhispercpp/models/ggml-${model_path}.bin"
    fi
    
    [[ -f "$model_path" ]] || return 1
}

# Microphone detection functions (clean, fast, reliable)
mic_present() {
    # prefer Pulse/PipeWire view; fall back to ALSA card list
    [[ -n "$(try 'pactl list short sources | grep -v monitor')" ]] && return 0
    [[ -n "$(try 'arecord -l | grep -E ^card')" ]] && return 0
    return 1
}

mic_accessible() {
    # if we can ask for a default source, the session can likely capture
    try 'pactl get-default-source' >/dev/null || return 1
    # /dev/snd should exist; don't over-enforce groups (PipeWire brokers access)
    [[ -d /dev/snd ]] || return 1
    return 0
}

mic_recording_now() {
    # Only consider it recording if hyprwhspr service is active AND actually recording
    if ! is_hyprwhspr_running; then
        return 1
    fi
    
    # Check if hyprwhspr process is actually running
    if ! pgrep -f "hyprwhspr" > /dev/null 2>&1; then
        return 1
    fi
    
    # Check recording status file written by hyprwhspr
    local status_file="$HOME/.config/hyprwhspr/recording_status"
    if [[ -f "$status_file" ]]; then
        local status
        status=$(cat "$status_file" 2>/dev/null)
        if [[ "$status" == "true" ]]; then
            return 0
        else
            return 1
        fi
    else
        # Fallback: check if mic is accessible and in use
        if ! mic_accessible; then
            return 1
        fi
        
        # Check PipeWire state as fallback
        local def state
        def="$(try 'pactl get-default-source')"
        [[ -n "$def" ]] || def='@DEFAULT_SOURCE@'
        state="$(try "pactl list sources | grep -B 5 -A 5 \"Name: $def\" | grep 'State:' | awk '{print \$2}'")"
        
        # Only consider RUNNING as recording (not SUSPENDED) to avoid false positives
        [[ "$state" == "RUNNING" ]]
    fi
}

mic_fidelity_label() {
    local def spec rate ch fmt
    def="$(try 'pactl get-default-source')"
    [[ -n "$def" ]] || def='@DEFAULT_SOURCE@'
    spec="$(try "pactl list sources | awk -v D=\"$def\" '
        /^[[:space:]]*Name:/{name=\$2}
        /^[[:space:]]*Sample Specification:/{spec=\$3\" \"\$4\" \"\$5}
        name==D && spec{print spec; exit}'")"
    # spec looks like: s16le 2ch 48000Hz
    fmt=$(awk '{print $1}' <<<"$spec")
    ch=$(awk '{print $2}' <<<"$spec" | tr -dc '0-9')
    rate=$(awk '{print $3}' <<<"$spec" | tr -dc '0-9')

    # super simple heuristic:
    # ≥48k and (24/32-bit OR plain 16-bit) → "hi-fi"; else "standard"
    if [[ -n "$rate" && $rate -ge 48000 ]]; then
        echo "hi-fi ($spec)"
    else
        [[ -n "$spec" ]] && echo "standard ($spec)" || echo ""
    fi
}

mic_tooltip_line() {
    local bits=()
    mic_present     && bits+=("present") || bits+=("not present")
    mic_accessible  && bits+=("access:ok") || bits+=("access:denied")
    mic_recording_now && bits+=("recording") || bits+=("idle")
    local fid; fid="$(mic_fidelity_label)"
    [[ -n "$fid" ]] && bits+=("$fid")
    echo "Mic: ${bits[*]}"
}

# Function to check if we can actually start recording
can_start_recording() {
    mic_present && mic_accessible
}

# Function to check if hyprwhspr is currently recording
is_hyprwhspr_recording() {
    # Check if hyprwhspr is running
    if ! is_hyprwhspr_running; then
        return 1
    fi
    
    # Use clean mic detection instead of heavy process scanning
    mic_recording_now
}



# Function to show notification
show_notification() {
    local title="$1"
    local message="$2"
    local urgency="${3:-normal}"
    
    if command -v notify-send &> /dev/null; then
        notify-send -i "$ICON_PATH" "$title" "$message" -u "$urgency"
    fi
}

# Function to toggle hyprwhspr
toggle_hyprwhspr() {
    if is_hyprwhspr_running; then
        echo "Stopping hyprwhspr..."
        systemctl --user stop hyprwhspr.service
        show_notification "hyprwhspr" "Stopped" "low"
    else
        if can_start_recording; then
            echo "Starting hyprwhspr..."
            systemctl --user start hyprwhspr.service
            show_notification "hyprwhspr" "Started" "normal"
        else
            echo "Cannot start hyprwhspr - no microphone available"
            show_notification "hyprwhspr" "No microphone available" "critical"
            return 1
        fi
    fi
}

# Function to start ydotoold if needed
start_ydotoold() {
    if ! is_ydotoold_running; then
        echo "Starting ydotoold..."
        systemctl --user start ydotool.service  # Using system service
        sleep 1
        if is_ydotoold_running; then
            show_notification "hyprwhspr" "ydotoold started" "low"
        else
            show_notification "hyprwhspr" "Failed to start ydotoold" "critical"
        fi
    fi
}

# Function to check service health and recover from stuck states
check_service_health() {
    if is_hyprwhspr_running; then
        # Check if service has been in "activating" state too long
        local service_status=$(systemctl --user show hyprwhspr.service --property=ActiveState --value)
        
        if [ "$service_status" = "activating" ]; then
            # Service is stuck starting, restart it
            echo "Service stuck in activating state, restarting..."
            systemctl --user restart hyprwhspr.service
            return 1
        fi
        
        # Check if recording state is stuck (running but no actual audio)
        if is_hyprwhspr_running && ! is_hyprwhspr_recording; then
            # Service is running but not recording - this is normal
            return 0
        fi
    fi
    return 0
}

# Function to emit JSON output for waybar with granular error classes
emit_json() {
    local state="$1" reason="${2:-}" custom_tooltip="${3:-}"
    local icon text tooltip class="$state"
    
    case "$state" in
        "recording")
            icon="󰍬"
            text="$icon REC"
            tooltip="hyprwhspr: Currently recording\n\nLeft-click: Stop recording\nRight-click: Restart\nMiddle-click: Restart"
            ;;
        "error")
            icon="󰆉"
            text="$icon ERR"
            tooltip="hyprwhspr: Issue detected${reason:+ ($reason)}\n\nLeft-click: Toggle service\nRight-click: Start service\nMiddle-click: Restart service"
            class="error"
            ;;
        "ready")
            icon="󰍬"
            text="$icon RDY"
            tooltip="hyprwhspr: Ready to record\n\nLeft-click: Start recording\nRight-click: Start service\nMiddle-click: Restart service"
            ;;
        *)
            icon="󰆉"
            text="$icon"
            tooltip="hyprwhspr: Unknown state\n\nLeft-click: Toggle service\nRight-click: Start service\nMiddle-click: Restart service"
            class="error"
            state="error"
            ;;
    esac
    
    # Add mic status to tooltip if provided
    if [[ -n "$custom_tooltip" ]]; then
        tooltip="$tooltip\n$custom_tooltip"
    fi
    
    # Output JSON for waybar
    printf '{"text":"%s","class":"%s","tooltip":"%s"}\n' "$text" "$class" "$tooltip"
}

# Function to get current state with detailed error reasons
get_current_state() {
    local reason=""
    
    # Check service health first
    check_service_health
    
    # Check if service is running
    if ! systemctl --user is-active --quiet hyprwhspr.service; then
        # Distinguish failed from inactive
        if systemctl --user is-failed --quiet hyprwhspr.service; then
            local result exec_code
            result=$(systemctl --user show hyprwhspr.service -p Result --value 2>/dev/null)
            exec_code=$(systemctl --user show hyprwhspr.service -p ExecMainStatus --value 2>/dev/null)
            reason="service_failed:${result:-unknown}:${exec_code:-}"
        else
            reason="service_inactive"
        fi
        echo "error:$reason"; return
    fi
    
    # Service is running - check if recording
    if is_hyprwhspr_recording; then
        echo "recording"; return
    fi
    
    # Service running but not recording - check dependencies
    if ! is_ydotoold_running; then
        echo "error:ydotoold"; return
    fi
    
    # Check PipeWire health
    if ! is_pipewire_ok; then
        echo "error:pipewire_down"; return
    fi
    
    # Check model existence
    if ! model_exists; then
        echo "error:model_missing"; return
    fi
    
    echo "ready"
}

# Main menu
case "${1:-status}" in
    "status")
        IFS=: read -r s r <<<"$(get_current_state)"
        emit_json "$s" "$r" "$(mic_tooltip_line)"
        ;;
    "toggle")
        toggle_hyprwhspr
        IFS=: read -r s r <<<"$(get_current_state)"
        emit_json "$s" "$r" "$(mic_tooltip_line)"
        ;;
    "start")
        if ! is_hyprwhspr_running; then
            if can_start_recording; then
                systemctl --user start hyprwhspr.service
                show_notification "hyprwhspr" "Started" "normal"
            else
                show_notification "hyprwhspr" "No microphone available" "critical"
            fi
        fi
        IFS=: read -r s r <<<"$(get_current_state)"
        emit_json "$s" "$r" "$(mic_tooltip_line)"
        ;;
    "stop")
        if is_hyprwhspr_running; then
            systemctl --user stop hyprwhspr.service
            show_notification "hyprwhspr" "Stopped" "low"
        fi
        IFS=: read -r s r <<<"$(get_current_state)"
        emit_json "$s" "$r" "$(mic_tooltip_line)"
        ;;
    "ydotoold")
        start_ydotoold
        IFS=: read -r s r <<<"$(get_current_state)"
        emit_json "$s" "$r" "$(mic_tooltip_line)"
        ;;
    "restart")
        systemctl --user restart hyprwhspr.service
        show_notification "hyprwhspr" "Restarted" "normal"
        IFS=: read -r s r <<<"$(get_current_state)"
        emit_json "$s" "$r" "$(mic_tooltip_line)"
        ;;
    "health")
        check_service_health
        if [ $? -eq 0 ]; then
            echo "Service health check passed"
        else
            echo "Service health check failed, attempting recovery"
        fi
        IFS=: read -r s r <<<"$(get_current_state)"
        emit_json "$s" "$r" "$(mic_tooltip_line)"
        ;;
    *)
        echo "Usage: $0 [status|toggle|start|stop|ydotoold|restart|health]"
        echo ""
        echo "Commands:"
        echo "  status    - Show current status (JSON output)"
        echo "  toggle    - Toggle hyprwhspr on/off"
        echo "  start     - Start hyprwhspr"
        echo "  stop      - Stop hyprwhspr"
        echo "  ydotoold  - Start ydotoold daemon"
        echo "  restart   - Restart hyprwhspr"
        echo "  health    - Check service health and recover if needed"
        ;;
esac
