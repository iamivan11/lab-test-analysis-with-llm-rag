#!/usr/bin/env bash
set -euo pipefail

APP_NAME="Lab Analyzer"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST_DIR="$ROOT_DIR/dist"
APP_PATH="$DIST_DIR/$APP_NAME.app"
DMG_PATH="$DIST_DIR/$APP_NAME.dmg"
STAGING_DIR="$DIST_DIR/dmg"

cd "$ROOT_DIR"

uv run --with pyinstaller pyinstaller packaging/lab_analyzer.spec --noconfirm

codesign --verify --deep --strict "$APP_PATH"

rm -rf "$STAGING_DIR"
mkdir -p "$STAGING_DIR"
cp -R "$APP_PATH" "$STAGING_DIR/"
ln -s /Applications "$STAGING_DIR/Applications"

rm -f "$DMG_PATH"
hdiutil create \
  -volname "$APP_NAME" \
  -srcfolder "$STAGING_DIR" \
  -ov \
  -format UDZO \
  "$DMG_PATH"

# Clean up intermediate build artifacts now that the DMG is in place.
# PyInstaller's COLLECT step writes `dist/Lab Analyzer/` (the
# unwrapped bundle), then BUNDLE wraps it into `dist/Lab Analyzer.app`,
# then we copy that into `dist/dmg/` for hdiutil. None of these are
# needed after the DMG exists — leaving them confuses users who expect
# dist/ to contain only the .dmg.
rm -rf "$APP_PATH" "$STAGING_DIR" "$DIST_DIR/$APP_NAME"

echo "Built: $DMG_PATH"
