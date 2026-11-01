#!/usr/bin/env python3
"""Termux ducking via SSH, optionally persistent and using fast root Android IPC.

stdin is the ducking-hooks control pipe, not a terminal. --persistent requires
a persistent hook worker. --backend android-root uses Magisk su + cmd instead
of the slower Termux:API broadcast bridge. No remote daemon installation needed.
"""

import argparse
import os
import shlex


# A remote lease, not a fixed duck duration: every heartbeat renews it.
# flock prevents overlapping clients from snapshotting an already-ducked volume.
# The journal survives a killed remote shell; the next session recovers it first.
REMOTE_SCRIPT = r'''
set -eu
stream=$1
level=$2
backend=${3:-termux-api}
mode=${4:-session}
case "$stream" in
    music) stream_id=3;; call) stream_id=0;; system) stream_id=1;;
    ring) stream_id=2;; alarm) stream_id=4;; notification) stream_id=5;;
    *) exit 64;;
esac
state_dir="${TMPDIR:-$HOME/.cache}/hyprwhspr-duck"
umask 077
mkdir -p "$state_dir"
exec 9>"$state_dir/$stream.lock"
flock -n 9 || exit 75
state="$state_dir/$stream.volume"
get_volume() {
    if [ "$backend" = android-root ]; then
        output=$(timeout 2 su -c "/system/bin/cmd media_session volume --stream $stream_id --get") || return 1
        [[ "$output" =~ volume\ is\ ([0-9]+)\ in\ range ]] || return 1
        printf '%s\n' "${BASH_REMATCH[1]}"
    else
        timeout 5 termux-volume | jq -er --arg stream "$stream" '.[] | select(.stream == $stream) | .volume'
    fi
}
set_volume() {
    value=$1
    case "$value" in ''|*[!0-9]*) return 1;; esac
    if [ "$backend" = android-root ]; then
        output=$(timeout 2 su -c "/system/bin/cmd media_session volume --stream $stream_id --set $value --get") || return 1
        [[ "$output" =~ volume\ is\ ([0-9]+)\ in\ range ]] || return 1
        [ "${BASH_REMATCH[1]}" = "$value" ]
    else
        timeout 5 termux-volume "$stream" "$value"
    fi
}
restore() {
    if [ -f "$state" ]; then
        saved=$(cat "$state")
        if set_volume "$saved"; then
            rm -f "$state"
        else
            return 1
        fi
    fi
}
trap restore EXIT
trap 'exit 0' HUP INT TERM
restore
duck() {
    # Repeated heartbeats must not overwrite the original snapshot.
    [ ! -f "$state" ] || return 0
    saved=$(get_volume)
    case "$saved" in ''|*[!0-9]*) return 1;; esac
    printf '%s\n' "$saved" > "$state.tmp"
    mv -f "$state.tmp" "$state"
    # Never raise a stream already quieter than the requested duck level.
    target=$(( saved < level ? saved : level ))
    set_volume "$target"
}
if [ "$mode" = persistent ]; then
    printf 'ready\n'
    while IFS= read -r -t 6 message; do
        case "$message" in
            recording) duck;;
            idle) restore;;
            *) exit 64;;
        esac
        printf 'ok\n'
    done
else
    # Do not change volume if recording ended before SSH finished connecting.
    IFS= read -r -t 6 message || exit 0
    [ "$message" = recording ] || exit 0
    duck
    while IFS= read -r -t 6 message; do
        [ "$message" = recording ] || break
    done
fi
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', required=True, help='SSH alias or user@hostname')
    parser.add_argument('--port', type=int, default=8022)
    parser.add_argument('--volume', type=int, default=2)
    parser.add_argument('--stream', choices=('music', 'call', 'system', 'ring', 'alarm', 'notification'), default='music')
    parser.add_argument('--persistent', action='store_true', help='Warm idle/recording protocol with ready/ok acknowledgements')
    parser.add_argument('--backend', choices=('termux-api', 'android-root'), default='termux-api')
    args = parser.parse_args()
    if args.host.startswith('-') or not 1 <= args.port <= 65535 or args.volume < 0:
        parser.error('invalid host, port, or volume')
    remote_command = shlex.join([
        'bash', '-c', REMOTE_SCRIPT, 'hyprwhspr-duck', args.stream,
        str(args.volume), args.backend, 'persistent' if args.persistent else 'session',
    ])
    # exec preserves the heartbeat stdin and process group owned by the worker.
    os.execvp('ssh', [
        'ssh', '-T', '-p', str(args.port),
        '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes',
        '-o', 'ConnectTimeout=5', '-o', 'ServerAliveInterval=2',
        '-o', 'ServerAliveCountMax=2', args.host, remote_command,
    ])


if __name__ == '__main__':
    main()
