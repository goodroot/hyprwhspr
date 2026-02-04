#!/bin/bash

# hyprwhspr waybar tray

# Find system Python, avoiding mise/pyenv/asdf managed versions
# This ensures we use /usr/bin/python3 even if mise has hijacked PATH
get_system_python() {
    local system_path="/usr/bin:/bin:/usr/local/bin:/usr/local/sbin:/usr/sbin:/sbin"
    local py
    py="$(PATH="$system_path" command -v python3 2>/dev/null)" && [ -x "$py" ] && { echo "$py"; return 0; }
    py="$(PATH="$system_path" command -v python 2>/dev/null)" && [ -x "$py" ] && { echo "$py"; return 0; }
    return 1
}
SYSTEM_PYTHON="$(get_system_python)" || SYSTEM_PYTHON="python3"

# Detect PACKAGE_ROOT dynamically
if [ -n "${HYPRWHSPR_ROOT:-}" ]; then
    PACKAGE_ROOT="$HYPRWHSPR_ROOT"
elif [ -f "${BASH_SOURCE[0]}" ]; then
    # Try to detect from script location
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    if [ -f "$SCRIPT_DIR/../../bin/hyprwhspr" ]; then
        PACKAGE_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
    else
        PACKAGE_ROOT="/usr/lib/hyprwhspr"
    fi
else
    PACKAGE_ROOT="/usr/lib/hyprwhspr"
fi

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
# Uses retry logic to handle startup timing issues (PipeWire may take a moment to initialize)
is_pipewire_ok() {
    local retries=3
    local delay=0.1  # 100ms between retries
    
    # Retry loop to handle startup timing
    for i in $(seq 1 $retries); do
        # Check if pactl is accessible
        if timeout 0.2s pactl info >/dev/null 2>&1; then
            # Check if we have any input sources (not monitors)
            # Note: pactl list short sources shows both inputs and output monitors
            # We need actual input sources, which don't have ".monitor" in the name
            local sources
            sources=$(pactl list short sources 2>/dev/null | grep -v "\.monitor" | grep -v "^$")
            if [[ -n "$sources" ]]; then
                return 0  # Success
            fi
        fi
        
        # If not last retry, wait before trying again
        if [[ $i -lt $retries ]]; then
            sleep "$delay"
        fi
    done
    
    # All retries failed
    return 1
}

# Function to check if model file exists
model_exists() {
    local cfg="$HOME/.config/hyprwhspr/config.json"
    [[ -f "$cfg" ]] || return 0

    # Check backend first - remote backends don't require local model validation
    local backend
    backend=$("$SYSTEM_PYTHON" - <<'PY' "$cfg" 2>/dev/null
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
try:
    data = json.loads(path.read_text())
    backend = data.get("transcription_backend", "pywhispercpp")
    # Backward compatibility
    if backend == "local":
        backend = "pywhispercpp"
    elif backend == "remote":
        backend = "rest-api"
    print(backend)
except Exception:
    print("pywhispercpp")
PY
    )

    # Backward compatibility: map old values
    if [[ "$backend" == "local" ]]; then
        backend="pywhispercpp"
    elif [[ "$backend" == "remote" ]]; then
        backend="rest-api"
    fi
    backend="${backend:-pywhispercpp}"

    # Remote backends don't require a local model file
    if [[ "$backend" == "rest-api" ]] || [[ "$backend" == "remote" ]] || [[ "$backend" == "realtime-ws" ]]; then
        return 0
    fi

    # Check onnx-asr availability (lightweight, non-blocking)
    if [[ "$backend" == "onnx-asr" ]]; then
        local venv_python="${XDG_DATA_HOME:-$HOME/.local/share}/hyprwhspr/venv/bin/python"
        # Fast timeout check - verify onnx_asr is importable
        # Uses absolute path to venv Python (resilient to MISE/PATH issues)
        if [[ -f "$venv_python" ]]; then
            # Only return success if import actually succeeds
            if timeout 0.5s "$venv_python" -c 'import onnx_asr' >/dev/null 2>&1; then
                return 0
            fi
        fi
        # If venv doesn't exist or import fails, return failure
        return 1
    fi

    # Only read model setting for pywhispercpp backends
    local model_path
    model_path=$("$SYSTEM_PYTHON" - <<'PY' "$cfg" 2>/dev/null
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
try:
    data = json.loads(path.read_text())
    print(data.get("model", ""))
except Exception:
    print("")
PY
    )

    [[ -n "$model_path" ]] || return 0  # use defaults; skip
    
    # If it's a short name like "base.en", resolve to pywhispercpp full path
    if [[ "$model_path" != /* ]]; then
        # Check for both multilingual and English-only versions (like pywhispercpp does)
        local models_dir="${XDG_DATA_HOME:-$HOME/.local/share}/pywhispercpp/models"
        local multilingual="${models_dir}/ggml-${model_path}.bin"
        local english_only="${models_dir}/ggml-${model_path}.en.bin"
        
        # Return success if either version exists
        [[ -f "$multilingual" ]] && return 0
        [[ -f "$english_only" ]] && return 0
        return 1
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
    local default_source
    default_source="$(try 'pactl get-default-source')"
    [[ -n "$default_source" ]] || return 1

    [[ -d /dev/snd ]] || return 1

    # Default source must exist (stale default is a real failure)
    if ! try 'pactl list short sources' | awk '{print $2}' | grep -qxF "$default_source"; then
        return 1
    fi

    # Don't treat monitor sources as a microphone
    [[ "$default_source" == *.monitor ]] && return 1

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
    if [[ ! -f "$status_file" ]]; then
        # No recording status file means hyprwhspr is not recording
        return 1
    fi
    
    local status
    status=$(cat "$status_file" 2>/dev/null)
    if [[ "$status" != "true" ]]; then
        return 1
    fi
    
    # Verify recording is actually active by checking audio_level file staleness
    # If recording is active, audio_level should be updated regularly (every ~100ms)
    # If the file is stale (>2 seconds old), recording likely stopped/crashed
    local level_file="$HOME/.config/hyprwhspr/audio_level"
    if [[ -f "$level_file" ]]; then
        # Check file modification time (seconds since epoch)
        local file_age
        file_age=$(($(date +%s) - $(stat -c %Y "$level_file" 2>/dev/null || echo 0)))
        
        # If audio_level file is stale (>2 seconds), recording is not actually happening
        if [[ $file_age -gt 2 ]]; then
            # Stale file - recording status file is likely left over from a crash
            return 1
        fi
    else
        # No audio_level file - if recording was active, this file should exist
        # But give it a grace period (maybe recording just started)
        # Check if recording_status file is very recent (<1 second)
        local status_age
        status_age=$(($(date +%s) - $(stat -c %Y "$status_file" 2>/dev/null || echo 0)))
        if [[ $status_age -gt 1 ]]; then
            # Status file exists but no audio_level file and status is >1s old
            # This suggests recording never actually started or crashed immediately
            return 1
        fi
    fi
    
    # All checks passed - recording is active
    return 0
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

# Function to check and show recovery result notification
check_recovery_result() {
    local result_file="$HOME/.config/hyprwhspr/recovery_result"
    local notification_lock_dir="$HOME/.config/hyprwhspr/.recovery_notification_lock"
    local notification_lock_file="${notification_lock_dir}/lock"

    if [[ ! -f "$result_file" ]]; then
        return 0
    fi

    local result
    result=$(cat "$result_file" 2>/dev/null)

    if [[ -z "$result" ]]; then
        return 0
    fi

    # Parse result format: status:reason:timestamp
    local status="${result%%:*}"
    local rest="${result#*:}"
    local reason="${rest%%:*}"
    local timestamp="${rest#*:}"

    # Check if result is fresh (within last 10 seconds)
    local current_time=$(date +%s)
    local result_age=$((current_time - timestamp))

    # Only process recent results
    if [[ $result_age -gt 10 ]]; then
        # Stale result, remove it
        rm -f "$result_file" 2>/dev/null
        # Clean up stale lock directory
        [[ -d "$notification_lock_dir" ]] && rmdir "$notification_lock_dir" 2>/dev/null
        return 0
    fi

    # Atomic check-and-set using directory creation
    # Directory creation is atomic - only one process can succeed
    local result_key="${status}:${reason}"
    local should_show=false
    
    # Try to create lock directory atomically
    if mkdir "$notification_lock_dir" 2>/dev/null; then
        # We successfully created the lock - we're the first to process this
        should_show=true
        # Write metadata to lock file inside the directory
        echo "${current_time}:${status}:${reason}" > "$notification_lock_file" 2>/dev/null
    else
        # Lock directory already exists - check if it's for the same result
        if [[ -f "$notification_lock_file" ]]; then
            local lock_content
            lock_content=$(cat "$notification_lock_file" 2>/dev/null)
            if [[ -n "$lock_content" ]]; then
                # Parse lock file format: timestamp:status:reason
                local lock_timestamp="${lock_content%%:*}"
                local lock_rest="${lock_content#*:}"
                local lock_status="${lock_rest%%:*}"
                local lock_reason="${lock_rest#*:}"
                local lock_key="${lock_status}:${lock_reason}"
                
                if [[ -n "$lock_timestamp" ]]; then
                    local lock_age=$((current_time - lock_timestamp))
                    # If same result and within 60 seconds, suppress notification
                    if [[ "$lock_key" == "$result_key" && $lock_age -lt 60 ]]; then
                        should_show=false
                    else
                        # Different result or stale lock (>60s) - clean up and allow notification
                        rm -rf "$notification_lock_dir" 2>/dev/null
                        # Retry creating lock (but only once to avoid infinite loop)
                        if mkdir "$notification_lock_dir" 2>/dev/null; then
                            should_show=true
                            echo "${current_time}:${status}:${reason}" > "$notification_lock_file" 2>/dev/null
                        fi
                    fi
                else
                    # Invalid lock timestamp - clean up stale lock
                    rm -rf "$notification_lock_dir" 2>/dev/null
                    if mkdir "$notification_lock_dir" 2>/dev/null; then
                        should_show=true
                        echo "${current_time}:${status}:${reason}" > "$notification_lock_file" 2>/dev/null
                    fi
                fi
            else
                # Lock file missing but directory exists - clean up stale lock
                rmdir "$notification_lock_dir" 2>/dev/null
                # Retry once
                if mkdir "$notification_lock_dir" 2>/dev/null; then
                    should_show=true
                    echo "${current_time}:${status}:${reason}" > "$notification_lock_file" 2>/dev/null
                fi
            fi
        else
            # Directory exists but no lock file - clean up and retry
            rmdir "$notification_lock_dir" 2>/dev/null
            if mkdir "$notification_lock_dir" 2>/dev/null; then
                should_show=true
                echo "${current_time}:${status}:${reason}" > "$notification_lock_file" 2>/dev/null
            fi
        fi
    fi

    if [[ "$should_show" == "true" ]]; then
        # Show notification based on status
        if [[ "$status" == "success" ]]; then
            case "$reason" in
                hotplug)
                    # Auto-restart service after hotplug to ensure clean state
                    # Only restart if not currently recording (avoid interrupting user)
                    sleep 1.5  # Let device enumeration fully settle
                    if ! is_hyprwhspr_recording; then
                        echo "Auto-restarting service after mic reconnection..." >&2
                        show_notification "hyprwhspr" "Microphone reconnected - restarting service..." "normal"
                        systemctl --user restart hyprwhspr.service
                        # Give service a moment to restart before showing ready notification
                        sleep 0.5
                        show_notification "hyprwhspr" "Ready to record" "low"
                    else
                        echo "Skipping auto-restart (recording in progress)" >&2
                        show_notification "hyprwhspr" "Microphone reconnected successfully" "normal"
                    fi
                    ;;
                mic_unavailable|mic_no_audio)
                    show_notification "hyprwhspr" "Microphone recovered successfully" "normal"
                    ;;
                *)
                    show_notification "hyprwhspr" "Recovery successful" "normal"
                    ;;
            esac

            # Clear any cached error states after successful recovery
            # Give device enumeration a moment to settle before next status check
            sleep 0.5
        elif [[ "$status" == "failed" ]]; then
            case "$reason" in
                hotplug)
                    show_notification "hyprwhspr" "Microphone detected but recovery failed - please wait or restart service" "critical"
                    ;;
                mic_unavailable|mic_no_audio)
                    show_notification "hyprwhspr" "Recovery failed - please replug microphone" "critical"
                    ;;
                *)
                    show_notification "hyprwhspr" "Recovery failed - please check microphone connection" "critical"
                    ;;
            esac
        fi
    fi

    # Keep result file for short grace period; stale results are cleared above
    # Lock directory will be cleaned up when stale (age > 60s) or on next check
}

# Function to toggle hyprwhspr
toggle_hyprwhspr() {
    if is_hyprwhspr_running; then
        echo "Stopping hyprwhspr..." >&2
        systemctl --user stop hyprwhspr.service
        show_notification "hyprwhspr" "Stopped" "low"
    else
        if can_start_recording; then
            echo "Starting hyprwhspr..." >&2
            systemctl --user start hyprwhspr.service
            show_notification "hyprwhspr" "Started" "normal"
        else
            echo "Cannot start hyprwhspr - no microphone available" >&2
            show_notification "hyprwhspr" "No microphone available" "critical"
            return 1
        fi
    fi
}

# Function to control recording (start/stop)
control_recording() {
    local control_file="$HOME/.config/hyprwhspr/recording_control"

    # Check if currently recording
    if is_hyprwhspr_recording; then
        # Stop recording
        echo "stop" > "$control_file"
        # No notification - Waybar icon change provides visual feedback
    else
        # Start recording - ensure service is running first
        if ! is_hyprwhspr_running; then
            if can_start_recording; then
                echo "Starting hyprwhspr service..." >&2
                systemctl --user start hyprwhspr.service
                # Wait a moment for service to initialize
                sleep 0.5
            else
                show_notification "hyprwhspr" "No microphone available" "critical"
                return 1
            fi
        fi

        # Write start command to control file
        echo "start" > "$control_file"
        # No notification - Waybar icon change provides visual feedback
    fi
}

# Function to start ydotoold if needed
start_ydotoold() {
    if ! is_ydotoold_running; then
        echo "Starting ydotoold..." >&2
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
            echo "Service stuck in activating state, restarting..." >&2
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
            icon=""
            text="$icon"
            tooltip="hyprwhspr: Currently recording\n\nLeft-click: Stop recording\nRight-click: Restart service"
            ;;
        "error")
            icon="󰆉"
            text="$icon ERR"
            case "$reason" in
                mic_unavailable)
                    tooltip="hyprwhspr: Microphone not available\n\nMicrophone hardware is present but cannot capture audio.\nThis often happens after suspend/resume or boot.\n\nPlease unplug and replug your USB microphone.\n\nLeft-click: Start recording\nRight-click: Restart service"
                    ;;
                mic_no_audio)
                    tooltip="hyprwhspr: Recording but no audio input\n\nRecording is active but microphone is not providing audio.\nThis indicates the mic needs to be reconnected.\n\nPlease unplug and replug your USB microphone.\n\nLeft-click: Start recording\nRight-click: Restart service"
                    ;;
                *)
            tooltip="hyprwhspr: Issue detected${reason:+ ($reason)}\n\nLeft-click: Start recording\nRight-click: Restart service"
;;
            esac
            class="error"
            ;;
        "ready")
            icon=""
            text="$icon"
            tooltip="hyprwhspr: Ready to record\n\nLeft-click: Start recording\nRight-click: Restart service"
            ;;
        "stopped")
            icon=""
            text="$icon"
            tooltip="hyprwhspr: Stopped\n\nLeft-click: Start recording\nRight-click: Restart service"
            ;;
        *)
            icon="󰆉"
            text="$icon"
            tooltip="hyprwhspr: Unknown state\n\nLeft-click: Start recording\nRight-click: Restart service"
            class="error"
            state="error"
            ;;
    esac
    
    # Add mic status to tooltip if provided
    if [[ -n "$custom_tooltip" ]]; then
        tooltip="$tooltip\n$custom_tooltip"
    fi
    
    # Add cache-busting timestamp to tooltip (invisible but forces waybar refresh)
    # This ensures waybar sees each output as new, preventing stale state display
    local ts
    ts=$(_date_ms)
    tooltip="${tooltip}\n_ts:${ts}"
    
    # Escape newlines for JSON (replace \n with \\n)
    tooltip="${tooltip//$'\n'/\\n}"
    
    # Force waybar refresh by making text unique each time with zero-width space
    # Waybar may cache based on text field, so we add an invisible character
    # Using zero-width space (U+200B) - completely invisible but makes text unique
    # We cycle through a few zero-width spaces based on timestamp to ensure uniqueness
    local zws_count=$((ts % 10))
    local zws=""
    # Add 0-9 zero-width spaces based on timestamp (invisible but unique)
    for ((i=0; i<zws_count; i++)); do
        zws="${zws}$(printf '\u200B')"
    done
    text="${text}${zws}"
    
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
            echo "error:$reason"; return
        else
            echo "stopped"; return
        fi
    fi
    
    # Service is running - check if recording
    if is_hyprwhspr_recording; then
        # Recording is active - don't check audio levels (low levels are normal during speech pauses)
        # Only check mic availability when service is running but NOT recording
        echo "recording"; return
    fi
    
    # Service running but not recording - check dependencies
    if ! is_ydotoold_running; then
        echo "error:ydotoold"; return
    fi

    # Check if mic is present and accessible
    # BUT: if recovery just succeeded (within last 5 seconds), give it grace period
    local recovery_file="$HOME/.config/hyprwhspr/recovery_result"
    local in_recovery_grace=false
    if [[ -f "$recovery_file" ]]; then
        local result=$(cat "$recovery_file" 2>/dev/null)
        local status="${result%%:*}"
        local rest="${result#*:}"
        local reason="${rest%%:*}"
        local timestamp="${rest#*:}"
        local current_time=$(date +%s)
        local result_age=$((current_time - timestamp))

        # If recovery succeeded within last 5 seconds, we're in grace period
        if [[ "$status" == "success" && $result_age -lt 5 ]]; then
            in_recovery_grace=true
        fi
    fi

    # Only check mic if NOT in recovery grace period
    if [[ "$in_recovery_grace" == "false" ]]; then
        if ! mic_present || ! mic_accessible; then
            echo "error:mic_unavailable"; return
        fi
    fi
    
    # Check for zero-volume signal from main app (mic present but not working)
    local zero_volume_file="$HOME/.config/hyprwhspr/.mic_zero_volume"
    if [[ -f "$zero_volume_file" ]]; then
        # Check file age - if recent (<60s), show error
        local file_age
        file_age=$(($(date +%s) - $(stat -c %Y "$zero_volume_file" 2>/dev/null || echo 0)))
        if [[ $file_age -lt 60 ]]; then
            echo "error:mic_no_audio"; return
        else
            # File is stale - remove it
            rm -f "$zero_volume_file" 2>/dev/null || true
        fi
    fi
    
    # Check PipeWire health (after mic check - less specific error)
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
        # Check for recovery results and show notifications
        check_recovery_result
        IFS=: read -r s r <<<"$(get_current_state)"
        emit_json "$s" "$r" "$(mic_tooltip_line)"
        ;;
    "toggle")
        toggle_hyprwhspr
        IFS=: read -r s r <<<"$(get_current_state)"
        # Only output JSON if stdout is not a TTY (i.e., being called by Waybar)
        if [ ! -t 1 ]; then
            emit_json "$s" "$r" "$(mic_tooltip_line)"
        fi
        ;;
    "record")
        control_recording
        IFS=: read -r s r <<<"$(get_current_state)"
        # Only output JSON if stdout is not a TTY (i.e., being called by Waybar)
        if [ ! -t 1 ]; then
            emit_json "$s" "$r" "$(mic_tooltip_line)"
        fi
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
        # Only output JSON if stdout is not a TTY (i.e., being called by Waybar)
        if [ ! -t 1 ]; then
            emit_json "$s" "$r" "$(mic_tooltip_line)"
        fi
        ;;
    "stop")
        if is_hyprwhspr_running; then
            systemctl --user stop hyprwhspr.service
            show_notification "hyprwhspr" "Stopped" "low"
        fi
        IFS=: read -r s r <<<"$(get_current_state)"
        # Only output JSON if stdout is not a TTY (i.e., being called by Waybar)
        if [ ! -t 1 ]; then
            emit_json "$s" "$r" "$(mic_tooltip_line)"
        fi
        ;;
    "ydotoold")
        start_ydotoold
        IFS=: read -r s r <<<"$(get_current_state)"
        # Only output JSON if stdout is not a TTY (i.e., being called by Waybar)
        if [ ! -t 1 ]; then
            emit_json "$s" "$r" "$(mic_tooltip_line)"
        fi
        ;;
    "restart")
        systemctl --user restart hyprwhspr.service
        show_notification "hyprwhspr" "Restarted" "normal"
        IFS=: read -r s r <<<"$(get_current_state)"
        # Only output JSON if stdout is not a TTY (i.e., being called by Waybar)
        if [ ! -t 1 ]; then
            emit_json "$s" "$r" "$(mic_tooltip_line)"
        fi
        ;;
    "health")
        check_service_health
        if [ $? -eq 0 ]; then
            echo "Service health check passed" >&2
        else
            echo "Service health check failed, attempting recovery" >&2
        fi
        IFS=: read -r s r <<<"$(get_current_state)"
        # Only output JSON if stdout is not a TTY (i.e., being called by Waybar)
        if [ ! -t 1 ]; then
            emit_json "$s" "$r" "$(mic_tooltip_line)"
        fi
        ;;
    *)
        echo "Usage: $0 [status|toggle|record|start|stop|ydotoold|restart|health]"
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
