#!/usr/bin/env bash
# JarvisLabs helpers for the 3-Channel Physics-Motivated experiment.
# Mirrors scripts/jarvis.sh (baseline) with the NEW entrypoints.
# Auth first:  jl setup --token YOUR_TOKEN --agents universal --yes
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

usage() {
  cat <<'EOF'
Usage:
  ./scripts/jarvis_physics.sh status            # account + GPU availability
  ./scripts/jarvis_physics.sh create [GPU]      # create instance (default L4)
  ./scripts/jarvis_physics.sh upload <id>       # upload experiment code + creds
  ./scripts/jarvis_physics.sh preprocess <id>   # build processed_dataset/ on instance
  ./scripts/jarvis_physics.sh train <id> [...]  # train p1p2-n on 3-channel data
  ./scripts/jarvis_physics.sh logs <run_id>     # follow run logs
  ./scripts/jarvis_physics.sh pause <id>        # stop compute billing
  ./scripts/jarvis_physics.sh destroy <id>      # delete instance

Examples:
  ./scripts/jarvis_physics.sh create L4
  ./scripts/jarvis_physics.sh upload 12345
  ./scripts/jarvis_physics.sh preprocess 12345 --sync
  ./scripts/jarvis_physics.sh train 12345 --epochs 30 --batch 16
EOF
}

cmd="${1:-}"
shift || true

case "$cmd" in
  status)
    jl status
    jl gpus
    ;;
  create)
    GPU="${1:-L4}"
    shift || true
    # 120GB: room for source dataset + processed_dataset (~17GB+) + checkpoints
    exec jl create --gpu "$GPU" --storage 120 --name "capstone-physics" --template pytorch --yes --json "$@"
    ;;
  upload)
    ID="${1:?machine id required}"
    shift || true
    echo "Uploading experiment code..."
    jl upload "$ID" "$ROOT/space_debris" /home/space_debris
    jl upload "$ID" "$ROOT/preprocessing" /home/preprocessing
    jl upload "$ID" "$ROOT/configs" /home/configs
    jl upload "$ID" "$ROOT/scripts" /home/scripts
    jl upload "$ID" "$ROOT/train_physics_jarvis.py" /home/train_physics_jarvis.py
    jl upload "$ID" "$ROOT/requirements-gpu.txt" /home/requirements-gpu.txt
    if [ -f "$ROOT/credentials.json" ]; then
      echo "Uploading OAuth credentials.json (Drive upload/sync)..."
      jl upload "$ID" "$ROOT/credentials.json" /home/credentials.json
    else
      echo "NOTE: credentials.json not found - Drive sync/upload will be unavailable."
    fi
    if [ -d "$ROOT/yolo_dataset" ]; then
      echo "Uploading local yolo_dataset/ (large - can take a while)..."
      jl upload "$ID" "$ROOT/yolo_dataset" /home/yolo_dataset
    else
      echo "NOTE: no local yolo_dataset/ - use '--sync' to pull from Drive on the instance."
    fi
    echo "Done. Remote layout: /home/{space_debris,preprocessing,configs,scripts,...}"
    ;;
  preprocess)
    ID="${1:?machine id required}"
    shift || true
    # Build processed_dataset/ on the instance (add --sync to pull Drive data first)
    exec jl run . --script train_physics_jarvis.py --on "$ID" --yes -- --preprocess-only "$@"
    ;;
  train)
    ID="${1:?machine id required}"
    shift || true
    # Preprocess if needed (auto), then train p1p2-n with HSV disabled
    exec jl run . --script train_physics_jarvis.py --on "$ID" --yes -- "$@"
    ;;
  logs)
    RUN_ID="${1:?run id required}"
    shift || true
    exec jl run logs "$RUN_ID" --follow "$@"
    ;;
  pause)
    ID="${1:?machine id required}"
    shift || true
    exec jl pause "$ID" --yes
    ;;
  destroy)
    ID="${1:?machine id required}"
    shift || true
    exec jl destroy "$ID" --yes
    ;;
  *)
    usage
    exit 1
    ;;
esac
