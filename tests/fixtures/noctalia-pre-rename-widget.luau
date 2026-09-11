--!nonstrict
-- hyprwhspr bar widget for Noctalia.
--
-- Thin shim over the existing tray script (single source of truth for state
-- detection, shared with the Waybar module): polls `hyprwhspr-tray.sh status`
-- and renders its JSON (class + tooltip) as a glyph. Clicks invoke the same
-- script actions the Waybar module binds.

local root = noctalia.getenv("HYPRWHSPR_ROOT")
if root == nil or root == "" then
  root = "/usr/lib/hyprwhspr"
end
local tray = root .. "/config/hyprland/hyprwhspr-tray.sh"

-- state class (from the tray script's JSON) -> glyph + palette color
local STATES = {
  recording = { glyph = "player-record-filled", color = "error" },
  ready     = { glyph = "microphone",           color = "primary" },
  unloaded  = { glyph = "zzz",                  color = "outline" },
  stopped   = { glyph = "microphone-off",       color = "outline" },
  error     = { glyph = "alert-triangle",       color = "error" },
}

local function render(json)
  if type(json) ~= "string" or json == "" then
    return
  end
  local class = json:match('"class":"([^"]*)"') or "error"
  local tooltip = json:match('"tooltip":"([^"]*)"') or ""

  -- Strip the Waybar cache-buster line, unescape \n
  tooltip = tooltip:gsub("\\n_ts:%d+", "")
  tooltip = tooltip:gsub("\\n", "\n")

  local state = STATES[class] or STATES.error
  barWidget.setGlyph(state.glyph)
  barWidget.setGlyphColor(state.color)
  if tooltip ~= "" then
    barWidget.setTooltip(tooltip)
  end
end

local function refresh()
  noctalia.runAsync(tray .. " status", function(result)
    if type(result) ~= "table" then
      return
    end
    render(result.stdout)
  end)
end

function update()
  noctalia.setUpdateInterval(1000)
  refresh()
end

function onClick()
  noctalia.runAsync(tray .. " record", function() refresh() end)
end

function onRightClick()
  noctalia.runAsync(tray .. " restart", function() refresh() end)
end

render('{"class":"ready","tooltip":"hyprwhspr"}')
