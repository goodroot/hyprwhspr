# Noctalia

[Noctalia](https://noctalia.dev) (v5+) ships its own bar and theming engine. `hyprwhspr setup` offers this integration when Noctalia is detected, or run it directly:

```bash
hyprwhspr noctalia install   # also: status / remove
```

You get:

- **Bar widget** ([`noctwhspr`](https://noctalia.dev/plugins/community/noctwhspr)) — service/recording state as a glyph; left-click records, right-click restarts. Also installable straight from Noctalia's plugin browser (Settings → Plugins).
- **Visualizer theme sync** — the recording overlay follows your live Noctalia palette, including theme switches.

Install enables the plugin, but placing the widget is up to you: **Noctalia Settings → Bar → add widget → noctwhspr** (or add `goodroot/noctwhspr:status` to a bar's widget list in Noctalia's `settings.toml`). Setup reminds you of this only when the widget isn't in your bar yet; reinstalls keep existing placements.

Earlier `goodroot/hyprwhspr` names migrate automatically on reinstall.

Niri users: see [Service starts but doesn't work until restarted](../TROUBLESHOOTING.md#service-starts-but-doesnt-work-until-restarted) for the `NIRI_SOCKET` requirement.
