#!/bin/bash
CONFIG=${1:-Realtime_Hosts20}
PROJECT_DIR=/home/opp_env/default_workspace/netsim
INET=/home/opp_env/default_workspace/inet-4.5.4/src

echo "Cleaning old results..."
truncate -s 0 $PROJECT_DIR/results/congestion_stream.jsonl 2>/dev/null || true
truncate -s 0 $PROJECT_DIR/results/portscan_stream.jsonl 2>/dev/null || true
touch $PROJECT_DIR/results/congestion_stream.jsonl
touch $PROJECT_DIR/results/portscan_stream.jsonl

echo "Starting: $CONFIG"
cd $PROJECT_DIR
./netsim \
  -n .:$INET \
  -f configs/omnetpp.ini \
  -c $CONFIG \
  -u Cmdenv \
  2>&1 | tee results/sim_output.log
