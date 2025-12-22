This is a project which I've forked and have my own private repo here: https://github.com/keithschacht/hyprwhspr

It's like Whisper Flow, a popular Mac app, but it's a copy for arch.

## Installation (Running from Repo)

This installation runs directly from the repo rather than the pacman package. This allows for easy development and testing of changes.

### How it's set up:

1. **Systemd service** runs from the repo:
   - Service file: `~/.config/systemd/user/hyprwhspr.service`
   - Points to: `/home/keithschacht/repos/active/hyprwhspr/`
   - Uses the venv at: `~/.local/share/hyprwhspr/venv/`

2. **Waybar module** references the repo:
   - Module config: `~/.config/waybar/hyprwhspr-module.jsonc`
   - Style import: `~/.config/waybar/style.css` imports from repo
   - Tray script: `/home/keithschacht/repos/active/hyprwhspr/config/hyprland/hyprwhspr-tray.sh`

3. **Config files** remain in standard locations:
   - `~/.config/hyprwhspr/config.json` - main configuration
   - `~/.config/hyprwhspr/sounds/` - custom notification sounds

### Making changes:

1. Edit files in this repo (`/home/keithschacht/repos/active/hyprwhspr/`)
2. Restart the service: `systemctl --user restart hyprwhspr`
3. Changes take effect immediately

### Key commands:

```bash
# Restart service after code changes
systemctl --user restart hyprwhspr

# Check service status/logs
systemctl --user status hyprwhspr
journalctl --user -u hyprwhspr -f

# Reload waybar after waybar-related changes
pkill -SIGUSR2 waybar
```

### Pulling upstream changes

When I say "pull in latest changes from upstream":

1. **Check for local changes** - Run `git status` and alert me if there are uncommitted changes that need to be stashed or resolved before proceeding
2. **Wait for clean state** - Don't proceed until main is clean
3. **Pull and merge upstream**:
   ```bash
   git checkout upstream
   git pull
   git checkout main
   git merge upstream
   ```
4. **Restart the service** - Run `systemctl --user restart hyprwhspr` to apply the changes
5. **Summarize** - Tell me what changed in the new commits

### Custom notification sounds

The custom sounds in `~/.config/hyprwhspr/sounds/` are soft sine wave tones:

**Generation commands:**

```bash
cd ~/.config/hyprwhspr/sounds

# Start sound (higher pitch - 600Hz)
ffmpeg -y -f lavfi -i "sine=frequency=600:duration=0.15" \
  -af "afade=t=in:st=0:d=0.02,afade=t=out:st=0.10:d=0.05,volume=0.4" \
  start-soft.ogg

# Stop sound (lower pitch - 400Hz)
ffmpeg -y -f lavfi -i "sine=frequency=400:duration=0.15" \
  -af "afade=t=in:st=0:d=0.02,afade=t=out:st=0.10:d=0.05,volume=0.4" \
  stop-soft.ogg

# Error sound (double-beep at 400Hz with 40ms gap)
ffmpeg -y -f lavfi -i "sine=frequency=400:duration=0.15" \
  -af "afade=t=in:st=0:d=0.02,afade=t=out:st=0.10:d=0.05,volume=0.4" beep.wav
ffmpeg -y -f lavfi -i "anullsrc=r=44100:cl=mono" -t 0.04 gap.wav
ffmpeg -y -i beep.wav -i gap.wav -i beep.wav \
  -filter_complex "[0:a][1:a][2:a]concat=n=3:v=0:a=1" \
  -c:a libvorbis error-soft.ogg
rm -f beep.wav gap.wav
```

**Parameters explained:**
- `sine=frequency=XXX` - Pure sine wave at specified Hz (600=start, 400=stop/error)
- `duration=0.15` - 150ms per tone
- `afade=t=in:st=0:d=0.02` - Fade in over 20ms (prevents click)
- `afade=t=out:st=0.10:d=0.05` - Fade out starting at 100ms, lasting 50ms
- `volume=0.4` - 40% volume (keeps it subtle)
- Error sound: two 150ms beeps with 40ms silence gap = 340ms total

### Important config notes:

The `~/.config/hyprwhspr/config.json` has these custom settings for this machine:

- `"grab_keys": false` - Required because keyd grabs keyboards first
- `"selected_device_path": "/dev/input/event15"` - Points to keyd virtual keyboard (may change on reboot)
- Custom soft notification sounds in `~/.config/hyprwhspr/sounds/`

### If shortcuts stop working after reboot:

The keyd virtual keyboard device path may change. Find the new path:

```bash
cat /proc/bus/input/devices | grep -A5 "keyd virtual keyboard"
```

Then update `selected_device_path` in `~/.config/hyprwhspr/config.json`.
