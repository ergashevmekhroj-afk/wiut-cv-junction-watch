#!/usr/bin/env bash
# The detector weights are committed in weights/ (yolo11s.pt, 19 MB), so this
# script is only a fallback if the file is missing. Run once, with internet.
set -euo pipefail
cd "$(dirname "$0")"
[ -f yolo11s.pt ] || curl -L -o yolo11s.pt https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11s.pt
sha256sum yolo11s.pt
