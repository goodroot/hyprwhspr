#!/bin/bash
# Test script for hyprwhspr systemd services

set -e

echo "🧪 Testing hyprwhspr systemd services..."

# Check if services are loaded
echo "📋 Checking service status..."

echo "1. hyprwhspr service:"
if systemctl --user is-active hyprwhspr.service &> /dev/null; then
    echo "   ✅ Running"
elif systemctl --user is-enabled hyprwhspr.service &> /dev/null; then
    echo "   ⚠️  Enabled but not running"
else
    echo "   ❌ Not configured"
fi

echo "2. Ydotool service:"
if systemctl --user is-active ydotool.service &> /dev/null; then
    echo "   ✅ Running"
elif systemctl --user is-enabled ydotool.service &> /dev/null; then
    echo "   ⚠️  Enabled but not running"
else
    echo "   ❌ Not configured"
fi

# Check if Hyprland session target is available
echo "3. Hyprland session target:"
if systemctl --user is-active wayland-session@hyprland.desktop.target &> /dev/null; then
    echo "   ✅ Available and active"
else
    echo "   ⚠️  Not active (Hyprland may not be running)"
fi

# Check ydotool availability
echo "4. Ydotool binary:"
if command -v ydotool &> /dev/null; then
    echo "   ✅ Available at $(which ydotool)"
else
    echo "   ❌ Not found in PATH"
fi

# Check if services can be started
echo "5. Testing service start capability:"
if systemctl --user start hyprwhspr.service 2>/dev/null; then
    echo "   ✅ hyprwhspr can be started"
    systemctl --user stop hyprwhspr.service 2>/dev/null
else
    echo "   ❌ hyprwhspr cannot be started"
fi

if systemctl --user start ydotool.service 2>/dev/null; then
    echo "   ✅ Ydotool can be started"
    systemctl --user stop ydotool.service 2>/dev/null
else
    echo "   ❌ Ydotool cannot be started"
fi

echo ""
echo "🎯 Summary:"
echo "If you see any ❌ marks above, the services may need configuration."
echo "Run './scripts/install-services.sh' to set up the services properly."
echo ""
echo "To start services manually:"
echo "  systemctl --user start hyprwhspr"
echo "  systemctl --user start ydotool"
