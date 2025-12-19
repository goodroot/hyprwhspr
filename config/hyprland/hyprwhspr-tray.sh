#!/bin/bash

# hyprwhspr System Tray Status Script
# Shows hyprwhspr status in the Hyprland system tray with JSON output

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

# Persistent cache file for mic_actually_works() to survive across script invocations
# This prevents expensive Python test from running on every waybar poll (every 1 second)
MIC_CACHE_FILE="$HOME/.config/hyprwhspr/.mic_works_cache"

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
    backend=$(python - <<'PY' "$cfg" 2>/dev/null
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

    # REST API backends don't require a local model file
    if [[ "$backend" == "rest-api" ]] || [[ "$backend" == "remote" ]]; then
        return 0
    fi

    # Only read model setting for pywhispercpp backends
    local model_path
    model_path=$(python - <<'PY' "$cfg" 2>/dev/null
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
    # if we can ask for a default source, the session can likely capture
    local default_source
    default_source="$(try 'pactl get-default-source')"
    [[ -n "$default_source" ]] || return 1
    
    # /dev/snd should exist; don't over-enforce groups (PipeWire brokers access)
    [[ -d /dev/snd ]] || return 1
    
    # Check if default source is SUSPENDED (critical check)
    # Use 'pactl list sources' (not 'short') to get reliable state information
    local source_state
    source_state="$(try "pactl list sources | awk -v D=\"$default_source\" '
        /^[[:space:]]*Name: /{current_name=\$2}
        /^[[:space:]]*State: /{if(current_name==D){print \$2; exit}}'")"
    
    # Source is not accessible if SUSPENDED
    [[ "$source_state" == "SUSPENDED" ]] && return 1
    
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

# Function to verify mic can actually capture audio (not just that stream opens)
# Uses Python sd.rec() for quick test, cached persistently to avoid heavy calls
mic_actually_works() {
    local now cache_time cache_result test_script result max_level
    
    # Fast-path: If mic isn't present or accessible, skip expensive test and invalidate cache
    # This catches suspend/resume issues immediately
    if ! mic_present || ! mic_accessible; then
        # Invalidate cache immediately (mic state changed)
        [[ -f "$MIC_CACHE_FILE" ]] && rm -f "$MIC_CACHE_FILE" 2>/dev/null || true
        _cache[mic_works.time]=0
        _cache[mic_works.result]=""
        return 1
    fi
    
    # If recovery file exists and mic is present/accessible, mic might have been reseated
    # Invalidate cache to force fresh check (detect reseat immediately)
    local recovery_file="$HOME/.config/hyprwhspr/recovery_requested"
    if [[ -f "$recovery_file" ]]; then
        # Recovery was requested - mic might have been reseated, force fresh check
        [[ -f "$MIC_CACHE_FILE" ]] && rm -f "$MIC_CACHE_FILE" 2>/dev/null || true
        _cache[mic_works.time]=0
        _cache[mic_works.result]=""
    fi
    
    # Check persistent cache file (survives across script invocations)
    # Use shorter TTL (10s) to catch suspend/resume issues faster
    # This prevents expensive Python test from running on every waybar poll
    now=$(_date_ms)
    if [[ -f "$MIC_CACHE_FILE" ]]; then
        local cache_data
        cache_data=$(cat "$MIC_CACHE_FILE" 2>/dev/null)
        if [[ -n "$cache_data" ]]; then
            IFS=: read -r cache_time cache_result <<<"$cache_data"
            # If cache valid and recent (< 10s), return cached result
            # Shorter TTL ensures we catch suspend/resume issues within 10s
            if [[ -n "$cache_time" && -n "$cache_result" && $((now - cache_time)) -lt 10000 ]]; then
                [[ "$cache_result" == "1" ]] && return 0 || return 1
            fi
        fi
    fi
    
    # Also check in-memory cache (for same script invocation)
    cache_time="${_cache[mic_works.time]:-0}"
    cache_result="${_cache[mic_works.result]:-}"
    
    # If cache valid and recent (< 10s), return cached result
    if [[ -n "$cache_result" && $((now - cache_time)) -lt 10000 ]]; then
        [[ "$cache_result" == "1" ]] && return 0 || return 1
    fi
    
    # Cache expired or missing - run Python test
    test_script=$(cat <<'PYTHON_EOF'
import sys
import json
import sounddevice as sd
import numpy as np
import threading
import time

result = {"can_capture": False, "max_level": 0.0, "error": None}
rec_done = threading.Event()
audio_data = [None]
rec_error = [None]

def rec_thread():
    try:
        # Use sd.rec() - simpler blocking call, 0.05s test (faster)
        data = sd.rec(
            int(16000 * 0.05),  # 0.05 seconds of samples (faster)
            samplerate=16000,
            channels=1,
            dtype='float32'
        )
        sd.wait()  # Wait for recording to finish
        audio_data[0] = data
    except Exception as e:
        rec_error[0] = str(e)
    finally:
        rec_done.set()

thread = threading.Thread(target=rec_thread, daemon=True)
thread.start()

# Wait with strict timeout (0.2s max - faster)
if rec_done.wait(timeout=0.2):
    if rec_error[0]:
        result["error"] = rec_error[0]
    elif audio_data[0] is not None:
        data = audio_data[0]
        result["max_level"] = float(np.abs(data).max())
        # If max level is essentially zero, mic isn't working
        result["can_capture"] = result["max_level"] > 1e-5
    else:
        result["error"] = "No audio data returned"
else:
    result["error"] = "Recording timed out"

print(json.dumps(result))
PYTHON_EOF
)
    
    # Run the test script with strict timeout (0.5s total max)
    # This prevents hanging if sounddevice blocks indefinitely
    result=$(timeout 0.5s python3 -c "$test_script" 2>/dev/null)
    
    if [[ -n "$result" ]]; then
        # Parse JSON result using Python (more reliable than grep)
        can_capture=$(echo "$result" | python3 -c "import sys, json; d=json.load(sys.stdin); print('1' if d.get('can_capture') else '0')" 2>/dev/null)
        
        if [[ -n "$can_capture" ]]; then
            # Update both in-memory and persistent cache
            _cache[mic_works.result]="$can_capture"
            _cache[mic_works.time]=$now
            # Write to persistent cache file (survives across script invocations)
            echo "${now}:${can_capture}" > "$MIC_CACHE_FILE" 2>/dev/null || true
            
            [[ "$can_capture" == "1" ]] && return 0 || return 1
        else
            # JSON parsing failed - cache negative result
            _cache[mic_works.result]="0"
            _cache[mic_works.time]=$now
            echo "${now}:0" > "$MIC_CACHE_FILE" 2>/dev/null || true
            return 1
        fi
    else
        # Test failed - cache negative result
        _cache[mic_works.result]="0"
        _cache[mic_works.time]=$now
        echo "${now}:0" > "$MIC_CACHE_FILE" 2>/dev/null || true
        return 1
    fi
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

# Function to get audio level visualization
get_audio_level_viz() {
    local level_file="$HOME/.config/hyprwhspr/audio_level"
    
    if [[ ! -f "$level_file" ]]; then
        echo ""
        return
    fi
    
    local level
    level=$(cat "$level_file" 2>/dev/null || echo "0")
    
    # Convert level (0.0-1.0) to multi-segment dot visualization
    # Using smaller Unicode characters for pixel-like appearance
    local num_segments=12
    local inactive_char="·"  # middle dot for inactive segments
    local active_char="▪"    # small square for active segments
    
    # Apply non-linear scaling for better sensitivity to lower levels
    # Using square root curve: makes quiet sounds more visible
    # This maps low levels (0.0-0.3) to more segments for better responsiveness
    local active_segments=$(awk -v l="$level" -v n="$num_segments" 'BEGIN {
        # Apply square root scaling for better sensitivity
        # This makes lower levels fill more segments
        scaled = sqrt(l) * n
        segs = int(scaled + 0.5)  # Round to nearest integer
        if (segs > n) segs = n
        if (segs < 0) segs = 0
        print segs
    }')
    
    # Build the visualization string
    local viz=""
    local i
    for ((i=0; i<num_segments; i++)); do
        if [ $i -lt $active_segments ]; then
            viz="${viz}${active_char}"
        else
            viz="${viz}${inactive_char}"
        fi
    done
    
    echo "$viz"
}

# Function to emit JSON output for waybar with granular error classes
emit_json() {
    local state="$1" reason="${2:-}" custom_tooltip="${3:-}"
    local icon text tooltip class="$state"
    local audio_viz=""
    
    # Get audio visualization if recording
    if [[ "$state" == "recording" ]]; then
        audio_viz=$(get_audio_level_viz)
        [[ -n "$audio_viz" ]] && audio_viz=" $audio_viz"
    fi
    
    case "$state" in
        "recording")
            icon="󰍬"
            text="$icon REC$audio_viz"
            tooltip="hyprwhspr: Currently recording\n\nLeft-click: Stop recording\nRight-click: Restart service"
            ;;
        "error")
            icon="󰆉"
            text="$icon ERR"
            case "$reason" in
                mic_unavailable)
                    tooltip="hyprwhspr: Microphone not available\n\nMicrophone hardware is present but cannot capture audio.\nThis often happens after suspend/resume or boot.\n\nPlease unplug and replug your USB microphone.\n\nLeft-click: Toggle service\nRight-click: Restart service"
                    ;;
                mic_no_audio)
                    tooltip="hyprwhspr: Recording but no audio input\n\nRecording is active but microphone is not providing audio.\nThis indicates the mic needs to be reconnected.\n\nPlease unplug and replug your USB microphone.\n\nLeft-click: Toggle service\nRight-click: Restart service"
                    ;;
                *)
            tooltip="hyprwhspr: Issue detected${reason:+ ($reason)}\n\nLeft-click: Toggle service\nRight-click: Restart service"
;;
            esac
            class="error"
            ;;
        "ready")
            icon="󰍬"
            text="$icon RDY"
            tooltip="hyprwhspr: Ready to record\n\nLeft-click: Start recording\nRight-click: Restart service"
            ;;
        *)
            icon="󰆉"
            text="$icon"
            tooltip="hyprwhspr: Unknown state\n\nLeft-click: Toggle service\nRight-click: Restart service"
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
        else
            reason="service_inactive"
        fi
        echo "error:$reason"; return
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
    
    # Check if mic actually works BEFORE other checks
    # This is the most specific error and should be checked first
    if ! mic_actually_works; then
        # Request recovery from main app (file-based trigger)
        # Only touch file if it doesn't exist (first detection) or is old (>30s, meaning recovery was attempted)
        local recovery_file="$HOME/.config/hyprwhspr/recovery_requested"
        if [[ ! -f "$recovery_file" ]]; then
            # First detection - touch file to request recovery
            touch "$recovery_file" 2>/dev/null || true
        else
            # File exists - check age
            local file_age
            file_age=$(($(date +%s) - $(stat -c %Y "$recovery_file" 2>/dev/null || echo 0)))
            if [[ $file_age -lt 2 ]]; then
                # File is very new (<2s) - main app is likely processing it
                # Don't recreate it to avoid race condition
                :
            elif [[ $file_age -gt 30 ]]; then
                # File is old (>30s) - recovery was likely attempted and failed
                # Don't touch it again, just show error
                :
            fi
        fi
        # Invalidate cache when requesting recovery (mic state may change)
        [[ -f "$MIC_CACHE_FILE" ]] && rm -f "$MIC_CACHE_FILE" 2>/dev/null || true
        echo "error:mic_unavailable"; return
    else
        # Mic works - clear any pending recovery request and force cache refresh
        local recovery_file="$HOME/.config/hyprwhspr/recovery_requested"
        [[ -f "$recovery_file" ]] && rm -f "$recovery_file" 2>/dev/null || true
        # Clear cache to force fresh check next time (helps detect when mic is reseated)
        _cache[mic_works.time]=0
        _cache[mic_works.result]=""
        # Also clear persistent cache file to force fresh check
        [[ -f "$MIC_CACHE_FILE" ]] && rm -f "$MIC_CACHE_FILE" 2>/dev/null || true
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
