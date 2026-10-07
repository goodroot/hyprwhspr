# Configuration

Validate with ease:

```bash
hyprwhspr config validate  # Settings
hyprwhspr status --report  # Settings, installation, live state
```

Configure via `hyprwhspr setup`, the CLI, or by editing `~/.config/hyprwhspr/config.json` directly.

The config file uses **sparse storage** — it only contains values you've changed from the defaults, so it stays clean and upstream default changes apply automatically on update.

There is also a `$schema` reference for IDE autocompletion and validation:

```jsonc
{
    "$schema": "https://raw.githubusercontent.com/goodroot/hyprwhspr/main/share/config.schema.json"
}
```

To view your overrides or the full resolved config:

```bash
hyprwhspr config show        # Show your overrides only
hyprwhspr config show --all  # Show all settings including defaults
```

## Minimal configuration

Only 2 essential options:

```jsonc
{
    "primary_shortcut": "SUPER+ALT+D",
    "model": "base"
}
```

## Environment variable substitution

Both `config.json` and `credentials.json` support `${VAR}` tokens.

Tokens are stored as-is on disk and expanded at read time:

```jsonc
{
  "rest_api_key": "${OPENAI_API_KEY}"
}
```
