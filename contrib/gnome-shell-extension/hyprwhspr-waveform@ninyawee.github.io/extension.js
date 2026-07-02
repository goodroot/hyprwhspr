import St from 'gi://St';
import GLib from 'gi://GLib';
import Clutter from 'gi://Clutter';

import {Extension} from 'resource:///org/gnome/shell/extensions/extension.js';
import * as Main from 'resource:///org/gnome/shell/ui/main.js';

const N_BARS = 27;          // number of equalizer bars
const TICK_ACTIVE_MS = 55;  // poll/animate interval while recording
const TICK_IDLE_MS = 200;   // slower poll while idle — fewer wakeups, still <200ms to show
const BAR_MIN = 4;          // px, idle bar height
const BAR_MAX = 56;         // px, loudest bar height
const C1 = [34, 211, 238];  // cyan  (left)
const C2 = [167, 139, 250]; // violet (right)

const CONFIG_DIR = GLib.build_filenamev([GLib.get_user_config_dir(), 'hyprwhspr']);
const REC_FILE = GLib.build_filenamev([CONFIG_DIR, 'recording_status']);
const LEVEL_FILE = GLib.build_filenamev([CONFIG_DIR, 'audio_level']);
const RUNTIME = GLib.getenv('XDG_RUNTIME_DIR') || GLib.get_tmp_dir();
const PREVIEW_FILE = GLib.build_filenamev([RUNTIME, 'hyprwhspr', 'transcript_preview']);

function lerpColor(t) {
    const r = Math.round(C1[0] + (C2[0] - C1[0]) * t);
    const g = Math.round(C1[1] + (C2[1] - C1[1]) * t);
    const b = Math.round(C1[2] + (C2[2] - C1[2]) * t);
    return `rgb(${r},${g},${b})`;
}

function fileExists(p) {
    return GLib.file_test(p, GLib.FileTest.EXISTS);
}

function readLevel() {
    try {
        const [ok, bytes] = GLib.file_get_contents(LEVEL_FILE);
        if (!ok) return 0;
        const v = parseFloat(new TextDecoder().decode(bytes).trim());
        return Number.isFinite(v) ? Math.max(0, Math.min(1, v)) : 0;
    } catch (_e) {
        return 0;
    }
}

function readPreview() {
    try {
        if (!fileExists(PREVIEW_FILE)) return '';
        const [ok, bytes] = GLib.file_get_contents(PREVIEW_FILE);
        if (!ok) return '';
        return new TextDecoder().decode(bytes).trim();
    } catch (_e) {
        return '';
    }
}

export default class HyprwhsprWaveformExtension extends Extension {
    enable() {
        this._visible = false;
        this._phase = 0;
        this._bars = [];

        // The pill container — drawn inside gnome-shell, never grabs focus.
        this._osd = new St.BoxLayout({
            vertical: true,
            style_class: 'hw-osd',
            reactive: false,
            track_hover: false,
            can_focus: false,
            opacity: 0,
            visible: false,
        });

        const row = new St.BoxLayout({vertical: false, style_class: 'hw-osd-row'});

        this._dot = new St.Widget({style_class: 'hw-osd-dot'});
        row.add_child(this._dot);

        const barsBox = new St.BoxLayout({vertical: false, style_class: 'hw-osd-bars'});
        barsBox.set_height(BAR_MAX);
        for (let i = 0; i < N_BARS; i++) {
            const bar = new St.Widget({
                style_class: 'hw-osd-bar',
                y_align: Clutter.ActorAlign.CENTER,
                y_expand: true,
            });
            bar.set_width(4);
            bar.set_height(BAR_MIN);
            bar.set_style(`background-color: ${lerpColor(i / (N_BARS - 1))};`);
            barsBox.add_child(bar);
            this._bars.push(bar);
        }
        row.add_child(barsBox);
        this._osd.add_child(row);

        this._preview = new St.Label({style_class: 'hw-osd-preview', text: ''});
        this._preview.clutter_text.line_wrap = false;
        this._preview.clutter_text.ellipsize = 3; // PANGO_ELLIPSIZE_END
        this._osd.add_child(this._preview);

        // trackFullscreen:false keeps the pill visible when dictating into a
        // fullscreen window (browser video, fullscreen editor) — the common case.
        Main.layoutManager.addChrome(this._osd, {
            affectsInputRegion: false,
            trackFullscreen: false,
        });

        this._reposition();
        this._monitorsId = Main.layoutManager.connect('monitors-changed', () => this._reposition());

        this._tickMs = null;
        this._scheduleTick(TICK_IDLE_MS);
    }

    // (Re)arm the poll timer at the given interval. Called once from enable() and
    // again from _tick() when switching between idle and active rates.
    _scheduleTick(intervalMs) {
        this._tickMs = intervalMs;
        this._timeoutId = GLib.timeout_add(GLib.PRIORITY_DEFAULT, intervalMs, () => this._tick());
    }

    _reposition() {
        const m = Main.layoutManager.primaryMonitor;
        if (!m) return;
        const w = this._osd.width || 320;
        this._osd.set_position(
            m.x + Math.round((m.width - w) / 2),
            m.y + m.height - this._osd.height - 72
        );
    }

    _show() {
        if (this._visible) return;
        this._visible = true;
        this._osd.show();
        this._osd.set_pivot_point(0.5, 1.0);
        this._osd.scale_y = 0.85;
        this._osd.ease({
            opacity: 255,
            scale_y: 1.0,
            duration: 180,
            mode: Clutter.AnimationMode.EASE_OUT_BACK,
        });
        this._reposition();
    }

    _hide() {
        if (!this._visible) return;
        this._visible = false;
        this._osd.ease({
            opacity: 0,
            duration: 220,
            mode: Clutter.AnimationMode.EASE_OUT_QUAD,
            onComplete: () => this._osd.hide(),
        });
        for (const bar of this._bars)
            bar.ease({height: BAR_MIN, duration: 200, mode: Clutter.AnimationMode.EASE_OUT_QUAD});
        this._preview.text = '';
    }

    _tick() {
        const recording = fileExists(REC_FILE);
        if (recording)
            this._show();
        else
            this._hide();

        // Poll fast while the pill is up, slow while idle. When the rate needs to
        // change, re-arm at the new interval and let this (old) source expire.
        const wantMs = this._visible ? TICK_ACTIVE_MS : TICK_IDLE_MS;
        if (wantMs !== this._tickMs) {
            this._scheduleTick(wantMs);
            return GLib.SOURCE_REMOVE;
        }

        if (!this._visible) return GLib.SOURCE_CONTINUE;
        this._reposition();

        const level = readLevel();

        // Pulsing record dot.
        this._phase = (this._phase + 0.32) % (2 * Math.PI);
        this._dot.opacity = Math.round(150 + 105 * (0.5 + 0.5 * Math.sin(this._phase)));

        // Center-weighted, lively bar envelope driven by the live level.
        for (let i = 0; i < N_BARS; i++) {
            const env = Math.sin((Math.PI * i) / (N_BARS - 1)); // 0 at edges, 1 center
            const jitter = 0.55 + 0.45 * Math.random();
            const target = BAR_MIN + (BAR_MAX - BAR_MIN) * level * env * jitter;
            this._bars[i].ease({
                height: Math.max(BAR_MIN, Math.round(target)),
                duration: TICK_ACTIVE_MS + 25,
                mode: Clutter.AnimationMode.EASE_OUT_QUAD,
            });
        }

        const preview = readPreview();
        if (preview && preview !== this._preview.text)
            this._preview.text = preview;

        return GLib.SOURCE_CONTINUE;
    }

    disable() {
        if (this._timeoutId) {
            GLib.source_remove(this._timeoutId);
            this._timeoutId = null;
        }
        if (this._monitorsId) {
            Main.layoutManager.disconnect(this._monitorsId);
            this._monitorsId = null;
        }
        if (this._osd) {
            Main.layoutManager.removeChrome(this._osd);
            this._osd.destroy();
            this._osd = null;
        }
        this._bars = [];
        this._dot = null;
        this._preview = null;
    }
}
