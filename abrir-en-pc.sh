#!/usr/bin/env bash
# Abre English Buddy en la PC (Linux) con WebGPU activado, sin tocar tu Chrome normal.
# Usa un perfil de Chrome aparte. No actives Vulkan: en Wayland deja la ventana negra.
exec google-chrome \
  --user-data-dir="$HOME/.config/english-buddy-chrome" \
  --enable-unsafe-webgpu \
  --ignore-gpu-blocklist \
  --app=https://lj2755.github.io/english-buddy/ \
  "$@"
