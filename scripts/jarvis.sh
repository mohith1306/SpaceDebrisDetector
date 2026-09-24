#!/usr/bin/env bash
# One-shot helpers for JarvisLabs. Run from the Capstone project root.
# Auth first:  jl setup --token YOUR_TOKEN --agents universal --yes
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

usage() {
  cat <<'EOF'
Usage:
  ./scripts/jarvis.sh status          # account + GPU availability
  ./scripts/jarvis.sh create [GPU]    # create instance (default L4), print machine id
  ./scripts/jarvis.sh upload <id>     # upload project + raw data to instance
  ./scripts/jarvis.sh train <id> [...]  # rebuild + train on existing instance
  ./scripts/jarvis.sh logs <run_id>   # follow run logs
  ./scripts/jarvis.sh pause <id>      # stop compute billing
  ./scripts/jarvis.sh destroy <id>    # delete instance

Examples:
  ./scripts/jarvis.sh create L4
  ./scripts/jarvis.sh upload 12345
  ./scripts/jarvis.sh train 12345 --epochs 30 --batch 16
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
    # 120GB: room for raw train (~14GB) + processed (~17GB) + yolo_dataset (~17GB) + checkpoints
    exec jl create --gpu "$GPU" --storage 120 --name "capstone-yolo" --template pytorch --yes --json "$@"
    ;;
  upload)
    ID="${1:?machine id required}"
    shift || true
    echo "Uploading project code..."
    jl upload "$ID" "$ROOT/space_debris" /home/space_debris
    jl upload "$ID" "$ROOT/train_jarvis.py" /home/train_jarvis.py
    jl upload "$ID" "$ROOT/requirements-gpu.txt" /home/requirements-gpu.txt
    echo "Uploading raw dataset (train/ + labels/) — this can take a while..."
    jl upload "$ID" "$ROOT/train" /home/train
    jl upload "$ID" "$ROOT/labels" /home/labels
    echo "Done. Remote layout: /home/{space_debris,train,labels,train_jarvis.py}"
    ;;
  train)
    ID="${1:?machine id required}"
    shift || true
    # Run rebuild+train on existing instance; project synced under /home
    exec jl run . --script train_jarvis.py --on "$ID" --yes -- "$@"
    ;;
  logs)
    RUN_ID="${1:?run id required}"
    shift || true
    exec jl run logs "$RUN_ID" --follow "$@"
    ;;
  pause)
    ID="${1:?machine id required}"
    exec jl pause "$ID" --yes
    ;;
  destroy)
    ID="${1:?machine id required}"
    exec jl destroy "$ID" --yes
    ;;
  *)
    usage
    exit 1
    ;;
esac
