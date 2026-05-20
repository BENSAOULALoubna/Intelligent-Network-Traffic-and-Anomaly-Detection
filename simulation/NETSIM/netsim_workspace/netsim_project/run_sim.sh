#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# run_sim.sh — netsim master run script
# Usage:
#   ./run_sim.sh realtime        # run all host counts simultaneously (dashboard)
#   ./run_sim.sh realtime 20     # run only Hosts20 realtime
#   ./run_sim.sh batch           # run Batch_Hosts20 (default for data gen)
#   ./run_sim.sh combined        # run Combined_Hosts20
#   ./run_sim.sh stop            # kill everything
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NETSIM_DIR="$SCRIPT_DIR"
WORKSPACE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
INET_SRC="$WORKSPACE_DIR/inet-4.5.2/src"
RESULTS="$NETSIM_DIR/results"
DASHBOARD_DIR="$WORKSPACE_DIR/netsim_dashboard"

cd "$NETSIM_DIR" || exit 1

# ── cleanup function ──────────────────────────────────────────────────────────
cleanup() {
    echo ""
    echo "[run_sim] Stopping all simulations..."
    pkill -f "netsim" 2>/dev/null
    sleep 1

    echo "[run_sim] Clearing stream files..."
    for f in "$RESULTS"/stream_h*.jsonl "$RESULTS"/stream.jsonl; do
        [ -f "$f" ] && truncate -s 0 "$f" && echo "  Cleared: $f"
    done

    # Remove stale OMNeT++ result files if any slipped through
    rm -f "$RESULTS"/*.vec "$RESULTS"/*.sca 2>/dev/null
    rm -f "$NETSIM_DIR/configs/results"/*.vec \
          "$NETSIM_DIR/configs/results"/*.sca 2>/dev/null

    echo "[run_sim] Done."
}

# Register cleanup on exit / Ctrl+C / kill
trap cleanup EXIT INT TERM

# ── stop command ──────────────────────────────────────────────────────────────
if [ "$1" = "stop" ]; then
    cleanup
    exit 0
fi

# ── pre-run cleanup ───────────────────────────────────────────────────────────
echo "[run_sim] Pre-run cleanup..."
pkill -f "netsim" 2>/dev/null
sleep 1

mkdir -p "$RESULTS"
for f in "$RESULTS"/stream_h*.jsonl "$RESULTS"/stream.jsonl; do
    [ -f "$f" ] && truncate -s 0 "$f"
done
rm -f "$RESULTS"/*.vec "$RESULTS"/*.sca 2>/dev/null

# Check disk space
FREE_KB=$(df "$RESULTS" | awk 'NR==2{print $4}')
FREE_MB=$((FREE_KB / 1024))
echo "[run_sim] Free disk: ${FREE_MB} MB"
if [ "$FREE_MB" -lt 500 ]; then
    echo "[run_sim] WARNING: less than 500MB free. Aborting."
    exit 1
fi

NETSIM_BIN="$NETSIM_DIR/netsim"
NED_PATH="$NETSIM_DIR:$INET_SRC"

run_config() {
    local CONFIG=$1
    local INI=$2
    local LOGFILE="/tmp/sim_${CONFIG}.log"
    echo "[run_sim] Starting $CONFIG..."
    "$NETSIM_BIN" -n "$NED_PATH" -u Cmdenv -c "$CONFIG" "$INI" \
        > "$LOGFILE" 2>&1 &
    echo "  PID=$! log=$LOGFILE"
}

# ── mode dispatch ─────────────────────────────────────────────────────────────
MODE="${1:-realtime}"
HOSTS="${2:-all}"

case "$MODE" in

    realtime)
        echo "[run_sim] Starting REALTIME mode..."
        if [ "$HOSTS" = "all" ]; then
            # Run all four host counts simultaneously for comparison
            run_config "Realtime_Hosts5"  "$NETSIM_DIR/configs/omnetpp.ini"
            sleep 2   # stagger starts to avoid port conflicts
            run_config "Realtime_Hosts10" "$NETSIM_DIR/configs/omnetpp.ini"
            sleep 2
            run_config "Realtime_Hosts20" "$NETSIM_DIR/configs/omnetpp.ini"
            sleep 2
            run_config "Realtime_Hosts40" "$NETSIM_DIR/configs/omnetpp.ini"
        else
            # Single host count
            run_config "Realtime_Hosts${HOSTS}" "$NETSIM_DIR/configs/omnetpp.ini"
        fi
        echo ""
        echo "[run_sim] Dashboard streams:"
        echo "  H5  -> $RESULTS/stream_h5.jsonl  (port 5001)"
        echo "  H10 -> $RESULTS/stream_h10.jsonl (port 5002)"
        echo "  H20 -> $RESULTS/stream_h20.jsonl (port 5003)"
        echo "  H40 -> $RESULTS/stream_h40.jsonl (port 5004)"
        echo ""
        echo "[run_sim] Start dashboard:"
        echo "  cd $DASHBOARD_DIR && ~/.venv/bin/python dashboard/app.py"
        echo ""
        echo "[run_sim] Press Ctrl+C to stop all simulations and clear files."
        wait
        ;;

    batch)
        BATCH_HOSTS="${HOSTS:-20}"
        run_config "Batch_Hosts${BATCH_HOSTS}" "$NETSIM_DIR/configs/omnetpp.ini"
        echo "[run_sim] Batch running. Waiting for completion..."
        wait
        echo "[run_sim] Batch done."
        ;;

    combined)
        echo "[run_sim] Starting COMBINED mode (normal + congestion)..."
        run_config "Combined_Hosts20" "$NETSIM_DIR/configs/combined.ini"
        echo "[run_sim] Congestion fires at t=300s, clears at t=900s."
        echo "[run_sim] Press Ctrl+C to stop."
        wait
        ;;

    *)
        echo "Usage: $0 [realtime|realtime N|batch|combined|stop]"
        exit 1
        ;;
esac