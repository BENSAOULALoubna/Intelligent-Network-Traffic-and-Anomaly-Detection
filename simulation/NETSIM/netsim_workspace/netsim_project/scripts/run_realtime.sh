#!/bin/bash
CONFIG=${1:-Realtime_Hosts20}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_DIR="$SCRIPT_DIR"
WORKSPACE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
INET="$WORKSPACE_DIR/inet-4.5.2/src"

echo "Cleaning old results..."
truncate -s 0 $PROJECT_DIR/results/congestion_stream.jsonl 2>/dev/null || true
truncate -s 0 $PROJECT_DIR/results/portscan_stream.jsonl 2>/dev/null || true
touch $PROJECT_DIR/results/congestion_stream.jsonl
touch $PROJECT_DIR/results/portscan_stream.jsonl

echo "Starting: $CONFIG"
cd $PROJECT_DIR || exit
./netsim \
  -n .:$INET \
  -f configs/omnetpp.ini \
  -c "$CONFIG" \
  -u Cmdenv \
  2>&1 | tee results/sim_output.log
