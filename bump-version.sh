#!/bin/bash
# Version bump script for hyprwhspr
# Usage: ./bump-version.sh <new_version>
# Example: ./bump-version.sh 1.8.7
#
# This script:
# 1. Creates and pushes git tag in hyprwhspr repo
# 2. Updates PKGBUILD version and SHA256 in aur-aur repo
# 3. Updates .SRCINFO

set -euo pipefail

if [ $# -ne 1 ]; then
    echo "Usage: $0 <new_version>"
    echo "Example: $0 1.8.7"
    exit 1
fi

NEW_VERSION="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HYPRWHSPR_REPO="$SCRIPT_DIR"
AUR_REPO="$SCRIPT_DIR/../aur/hyprwhspr"

# Validate we're in the right directory
if [ ! -f "$HYPRWHSPR_REPO/lib/cli.py" ]; then
    echo "Error: Must run from hyprwhspr repo root"
    exit 1
fi

if [ ! -f "$AUR_REPO/PKGBUILD" ]; then
    echo "Error: AUR repo not found at $AUR_REPO"
    exit 1
fi

# Get current version from PKGBUILD
OLD_VERSION=$(grep '^pkgver=' "$AUR_REPO/PKGBUILD" | cut -d'=' -f2 | tr -d "'")

if [ -z "$OLD_VERSION" ]; then
    echo "Error: Could not determine current version from PKGBUILD"
    exit 1
fi

echo "=========================================="
echo "Bumping version from $OLD_VERSION to $NEW_VERSION"
echo "=========================================="
echo ""

# Step 1: Check git status in hyprwhspr repo
echo "1. Checking hyprwhspr repo status..."
cd "$HYPRWHSPR_REPO"
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo "   Error: Not in a git repository"
    exit 1
fi

# Check for uncommitted changes
if ! git diff-index --quiet HEAD --; then
    echo "   ⚠ Warning: You have uncommitted changes"
    echo "   Consider committing them before creating a release tag"
    read -p "   Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Step 2: Create and push git tag
echo "2. Creating git tag v$NEW_VERSION..."
if git rev-parse "v$NEW_VERSION" >/dev/null 2>&1; then
    echo "   ⚠ Tag v$NEW_VERSION already exists locally"
    read -p "   Delete and recreate? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        git tag -d "v$NEW_VERSION" || true
        git push origin ":refs/tags/v$NEW_VERSION" 2>/dev/null || true
    else
        echo "   Using existing tag"
    fi
fi

if ! git rev-parse "v$NEW_VERSION" >/dev/null 2>&1; then
    git tag -a "v$NEW_VERSION" -m "Release v$NEW_VERSION"
    echo "   ✓ Tag created locally"
    
    echo "3. Pushing tag to origin..."
    git push origin "v$NEW_VERSION"
    echo "   ✓ Tag pushed to origin"
    
    # Wait a moment for GitHub to process
    echo "   Waiting for GitHub to process tag..."
    sleep 3
else
    echo "3. Tag already exists, skipping push"
fi

# Step 4: Get SHA256 from GitHub
echo "4. Fetching SHA256 from GitHub..."
MAX_RETRIES=10
RETRY_COUNT=0
NEW_SHA256=""

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    NEW_SHA256=$(curl -sL "https://github.com/goodroot/hyprwhspr/archive/refs/tags/v$NEW_VERSION.tar.gz" | sha256sum | awk '{print $1}')
    
    if [ -n "$NEW_SHA256" ] && [ "$NEW_SHA256" != "0000000000000000000000000000000000000000000000000000000000000000" ]; then
        break
    fi
    
    RETRY_COUNT=$((RETRY_COUNT + 1))
    if [ $RETRY_COUNT -lt $MAX_RETRIES ]; then
        echo "   Retrying... ($RETRY_COUNT/$MAX_RETRIES)"
        sleep 2
    fi
done

if [ -z "$NEW_SHA256" ] || [ "$NEW_SHA256" = "0000000000000000000000000000000000000000000000000000000000000000" ]; then
    echo "   ✗ Error: Could not fetch SHA256 after $MAX_RETRIES attempts"
    echo "   The tag may not be available on GitHub yet. Try again in a moment."
    exit 1
fi

echo "   ✓ SHA256: $NEW_SHA256"

# Step 5: Update PKGBUILD
echo "5. Updating PKGBUILD..."
cd "$AUR_REPO"
sed -i "s/^pkgver=.*/pkgver=$NEW_VERSION/" PKGBUILD
sed -i "s/v$OLD_VERSION/v$NEW_VERSION/g" PKGBUILD

# Update SHA256 in PKGBUILD
if grep -q "sha256sums=" PKGBUILD; then
    sed -i "s/sha256sums=('.*')/sha256sums=('$NEW_SHA256')/" PKGBUILD
else
    # Add sha256sums line if it doesn't exist
    sed -i "/^source=.*/a sha256sums=('$NEW_SHA256')" PKGBUILD
fi

echo "   ✓ PKGBUILD updated"

# Step 6: Update .SRCINFO
echo "6. Updating .SRCINFO..."
makepkg --printsrcinfo > .SRCINFO
echo "   ✓ .SRCINFO updated"

echo ""
echo "=========================================="
echo "Version bump complete!"
echo "=========================================="
echo "Old version: $OLD_VERSION"
echo "New version: $NEW_VERSION"
echo "SHA256:      $NEW_SHA256"
echo ""
echo "Next steps:"
echo "1. Review changes:"
echo "   cd $AUR_REPO"
echo "   git diff PKGBUILD .SRCINFO"
echo ""
echo "2. Commit and push AUR package:"
echo "   cd $AUR_REPO"
echo "   git add PKGBUILD .SRCINFO"
echo "   git commit -m 'Bump to v$NEW_VERSION'"
echo "   git push"
echo ""
