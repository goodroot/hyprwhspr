--!nonstrict
-- hyprwhspr bar widget for Noctalia.
--
-- Thin shim over the existing tray script (single source of truth for state
-- detection, shared with the Waybar module): polls `hyprwhspr-tray.sh status`
-- and renders its JSON (class + tooltip) as a glyph. Clicks invoke the same
-- script actions the Waybar module binds.

local TRAY_REL = "/config/hyprland/hyprwhspr-tray.sh"

-- state class (from the tray script's JSON) -> glyph + palette color
local STATES = {
  recording = { glyph = "player-record-filled", color = "error" },
  ready     = { glyph = "microphone",           color = "primary" },
  unloaded  = { glyph = "zzz",                  color = "outline" },
  stopped   = { glyph = "microphone-off",       color = "outline" },
  error     = { glyph = "alert-triangle",       color = "error" },
}

local function applyState(state, tooltip)
  barWidget.setGlyph(state.glyph)
  barWidget.setGlyphColor(state.color)
  if tooltip ~= nil and tooltip ~= "" then
    barWidget.setTooltip(tooltip)
  end
end

local function applyError(tooltip)
  applyState(STATES.error, tooltip)
end

-- Resolve the tray script lazily and cache the hit. Noctalia's process is
-- spawned by the compositor, so HYPRWHSPR_ROOT is usually absent from its
-- environment — the widget "root" setting is the override for non-standard
-- install prefixes.
local trayCached = nil
local function trayPath()
  if trayCached ~= nil and noctalia.fileExists(trayCached) then
    return trayCached
  end
  trayCached = nil
  local candidates = {}
  local cfg = noctalia.getConfig("root")
  if type(cfg) == "string" and cfg ~= "" then
    table.insert(candidates, cfg)
  end
  local env = noctalia.getenv("HYPRWHSPR_ROOT")
  if env ~= nil and env ~= "" then
    table.insert(candidates, env)
  end
  table.insert(candidates, "/usr/lib/hyprwhspr")
  for _, root in ipairs(candidates) do
    local path = root .. TRAY_REL
    if noctalia.fileExists(path) then
      trayCached = path
      return path
    end
  end
  return nil
end

local function trayCommand(action)
  local path = trayPath()
  if path == nil then
    return nil
  end
  -- Single-quote the path: install roots may contain spaces.
  return "'" .. path .. "' " .. action
end

local function render(json)
  if type(json) ~= "string" or json == "" then
    applyError("hyprwhspr: tray script returned no status")
    return
  end
  local class = json:match('"class":"([^"]*)"') or "error"
  local tooltip = json:match('"tooltip":"([^"]*)"') or ""

  -- Strip the Waybar cache-buster line, unescape \n
  tooltip = tooltip:gsub("\\n_ts:%d+", "")
  tooltip = tooltip:gsub("\\n", "\n")

  applyState(STATES[class] or STATES.error, tooltip)
end

local function refresh()
  local cmd = trayCommand("status")
  if cmd == nil then
    applyError("hyprwhspr not found\nInstall hyprwhspr or set this widget's root setting")
    return
  end
  noctalia.runAsync(cmd, function(result)
    if type(result) ~= "table" then
      applyError("hyprwhspr: status query failed")
      return
    end
    render(result.stdout)
  end)
end

function update()
  -- The tray script re-derives state via many subprocesses per call; 2s is
  -- responsive enough (the recording overlay gives immediate feedback).
  noctalia.setUpdateInterval(2000)
  refresh()
end

function onClick()
  local cmd = trayCommand("record")
  if cmd ~= nil then
    noctalia.runAsync(cmd, function() refresh() end)
  end
end

function onRightClick()
  local cmd = trayCommand("restart")
  if cmd ~= nil then
    noctalia.runAsync(cmd, function() refresh() end)
  end
end

-- Honest placeholder until the first poll lands.
barWidget.setGlyph("microphone")
barWidget.setGlyphColor("outline")
barWidget.setTooltip("hyprwhspr: waiting for status…")
