#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "[VideoDNA] Installing CLI from: $REPO_DIR"

# Install CLI without forcing reinstall of system/Termux binary wheels
python3 -m pip install --no-build-isolation --no-deps -e "$REPO_DIR" 2>/dev/null \
  || python3 -m pip install --no-deps -e "$REPO_DIR"

# Auto-install Agent Skill for Antigravity, Claude Code, and Codex if their directories exist
install_skill() {
  local target_dir="$1/videodna"
  mkdir -p "$target_dir"
  cp -f "$REPO_DIR/skills/videodna/SKILL.md" "$target_dir/SKILL.md"
  echo "[VideoDNA] Installed Agent Skill -> $target_dir/SKILL.md"
}

[ -d "$HOME/.gemini/config/skills" ] && install_skill "$HOME/.gemini/config/skills"
[ -d "$HOME/.claude/skills" ] && install_skill "$HOME/.claude/skills"
[ -d "$HOME/.codex/skills" ] && install_skill "$HOME/.codex/skills"

# Ensure ~/.gemini/config/skills gets it if using Antigravity
if [ -d "$HOME/.gemini" ] && [ ! -f "$HOME/.gemini/config/skills/videodna/SKILL.md" ]; then
  install_skill "$HOME/.gemini/config/skills"
fi

echo ""
echo "[VideoDNA] Installation complete! Running environment check:"
videodna doctor || python3 -m videodna.cli doctor
