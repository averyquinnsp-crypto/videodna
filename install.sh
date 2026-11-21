#!/usr/bin/env bash
# VideoDNA installer — Termux, Debian/Ubuntu, macOS, and other Linux.
#
# Ubuntu/Debian (incl. Ubuntu 24.04+, where system pip is PEP 668
# externally-managed): installs missing system packages (python3-venv,
# python3-pip, ffmpeg), creates a venv at <repo>/.venv, and installs the
# CLI + URL deps (yt-dlp) into it, with a ~/.local/bin/videodna shim.
# Termux keeps the old system-pip path (its binary wheels are managed).
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$REPO_DIR/.venv"
echo "[VideoDNA] Installing CLI from: $REPO_DIR"

# --- environment detection -------------------------------------------------

is_termux() {
  case "${PREFIX:-}" in *com.termux*) return 0 ;; *) return 1 ;; esac
}

is_debian_like() {
  [ -r /etc/os-release ] || return 1
  # shellcheck disable=SC1091
  . /etc/os-release
  case "${ID:-} ${ID_LIKE:-}" in *debian*|*ubuntu*) return 0 ;; *) return 1 ;; esac
}

# --- Debian/Ubuntu system packages ----------------------------------------

install_system_deps_debian() {
  local missing=()
  python3 -c "import ensurepip" 2>/dev/null || missing+=(python3-venv)
  python3 -m pip --version >/dev/null 2>&1 || missing+=(python3-pip)
  command -v ffmpeg >/dev/null 2>&1 || missing+=(ffmpeg)
  if [ "${#missing[@]}" -eq 0 ]; then
    return 0
  fi
  echo "[VideoDNA] Missing system packages: ${missing[*]}"
  local apt_cmd="sudo apt-get"
  [ "$(id -u)" -eq 0 ] && apt_cmd="apt-get"
  if command -v sudo >/dev/null 2>&1 || [ "$(id -u)" -eq 0 ]; then
    echo "[VideoDNA] Installing system packages (may ask for your sudo password)..."
    if $apt_cmd update && $apt_cmd install -y "${missing[@]}"; then
      return 0
    fi
    echo "[VideoDNA] Automatic install failed — please run manually:"
  else
    echo "[VideoDNA] sudo not found — please install manually:"
  fi
  echo "  sudo apt-get update && sudo apt-get install -y ${missing[*]}"
  exit 1
}

# --- venv install (Debian/Ubuntu default, PEP 668 fallback elsewhere) -----

install_into_venv() {
  if [ ! -x "$VENV_DIR/bin/python" ]; then
    echo "[VideoDNA] Creating virtualenv at $VENV_DIR ..."
    python3 -m venv "$VENV_DIR"
  fi
  "$VENV_DIR/bin/pip" install --upgrade pip >/dev/null 2>&1 || true
  # [url] extra pulls yt-dlp (needed for URL ingest); numpy + Pillow come
  # from the base dependencies. No --no-deps here: a fresh venv needs them.
  "$VENV_DIR/bin/pip" install -e "$REPO_DIR[url]"
  echo "[VideoDNA] Installed videodna into $VENV_DIR"
  # PATH shim so plain `videodna` works (Ubuntu puts ~/.local/bin on PATH).
  mkdir -p "$HOME/.local/bin"
  cat > "$HOME/.local/bin/videodna" <<EOF
#!/bin/sh
exec "$VENV_DIR/bin/videodna" "\$@"
EOF
  chmod +x "$HOME/.local/bin/videodna"
  echo "[VideoDNA] Created shim: $HOME/.local/bin/videodna"
  # PATH shadowing check: an older videodna installed elsewhere (e.g.
  # /usr/local/bin from a previous release) wins over the new shim when it
  # appears earlier on PATH, so plain `videodna` would keep running the OLD
  # version after a "successful" install. The old check below only warned
  # when `videodna` was not found at all, which never happens for upgraders.
  # Warn LOUDLY instead of silently "succeeding". We never overwrite the
  # old binary here — removing it is the user's call.
  local existing
  existing="$(command -v videodna 2>/dev/null || true)"
  if [ -n "$existing" ] && [ "$existing" != "$HOME/.local/bin/videodna" ]; then
    local old_ver="(version unknown)"
    if "$existing" --version >/dev/null 2>&1; then
      old_ver="$("$existing" --version 2>/dev/null | head -n 1)"
    fi
    echo "[VideoDNA] WARNING: an older videodna is already on your PATH and will SHADOW this install!"
    echo "  Found: $existing  ($old_ver)"
    echo "  Plain \`videodna\` will keep running that old copy, NOT the new one at $HOME/.local/bin/videodna."
    echo "  Fix it with one of:"
    echo "    sudo rm \"$existing\"                       # remove the old binary"
    echo "    sudo mv \"$existing\" \"$existing.old\"     # or rename it out of the way"
    echo "    export PATH=\"\$HOME/.local/bin:\$PATH\"    # or put ~/.local/bin first on PATH"
  elif [ -z "$existing" ]; then
    echo "[VideoDNA] Note: ~/.local/bin is not on this shell's PATH yet."
    echo "  Run: export PATH=\"\$HOME/.local/bin:\$PATH\"  (or open a new terminal)"
  fi
}

# --- pre-flight ------------------------------------------------------------

if ! command -v python3 >/dev/null 2>&1; then
  echo "[VideoDNA] python3 not found. Install Python 3.10+ first, e.g.:"
  echo "  Ubuntu/Debian: sudo apt-get install -y python3 python3-venv python3-pip ffmpeg"
  echo "  macOS:         brew install python ffmpeg"
  echo "  Termux:        pkg install python ffmpeg"
  exit 1
fi

# --- install ---------------------------------------------------------------

if is_termux; then
  # Termux: system pip, no venv (binary wheels are managed by Termux).
  python3 -m pip install --no-build-isolation --no-deps -e "$REPO_DIR" 2>/dev/null \
    || python3 -m pip install --no-deps -e "$REPO_DIR"
elif is_debian_like; then
  install_system_deps_debian
  install_into_venv
else
  # macOS / other: try system pip first, fall back to a venv (PEP 668).
  if python3 -m pip install --no-deps -e "$REPO_DIR" 2>/dev/null; then
    echo "[VideoDNA] Installed videodna into system Python."
  else
    echo "[VideoDNA] System pip install failed (likely PEP 668) — installing into a venv instead."
    install_into_venv
  fi
fi

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
if [ -x "$VENV_DIR/bin/videodna" ]; then
  "$VENV_DIR/bin/videodna" doctor
elif command -v videodna >/dev/null 2>&1; then
  videodna doctor
else
  python3 -m videodna.cli doctor
fi
