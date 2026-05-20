import collections
import threading
import time

MAX_POINTS = 1000

# Thread-safe rolling buffer
history = collections.deque(maxlen=MAX_POINTS)
history_lock = threading.Lock()


def add_entry(entry):
    """Add entry with timestamp."""
    entry['_timestamp'] = time.time()
    with history_lock:
        history.append(entry)


def get_history():
    with history_lock:
        return list(history)


def get_recent_data(seconds=120):
    """Get data from last N seconds."""
    cutoff = time.time() - seconds
    with history_lock:
        return [d for d in history if d.get('_timestamp', 0) >= cutoff]


def get_portscan_data(seconds=120):
    """Get portscan records from last N seconds."""
    cutoff = time.time() - seconds
    with history_lock:
        return [d for d in history if d.get('type') == 'portscan' and d.get('_timestamp', 0) >= cutoff]


def get_congestion_data(seconds=120):
    """Get congestion records from last N seconds."""
    cutoff = time.time() - seconds
    with history_lock:
        return [d for d in history if d.get('type') == 'congestion' and d.get('_timestamp', 0) >= cutoff]


def get_portscan_window(timesteps=12):
    """Get last N timesteps for LSTM input (assumes ~10s per timestep)."""
    window = get_portscan_data(seconds=130)
    return window[-timesteps:] if len(window) >= timesteps else window


def get_latest_congestion():
    """Get most recent congestion record."""
    with history_lock:
        congestion = [d for d in history if d.get('type') == 'congestion']
        return congestion[-1] if congestion else None


def get_congestion_history(count=30):
    """Get recent congestion records for rolling features."""
    with history_lock:
        congestion = [d for d in history if d.get('type') == 'congestion']
        return congestion[-count:] if len(congestion) >= count else congestion


def get_all():
    """Alias for get_history() for compatibility with app.py"""
    return get_history()


def clear():
    """Clear all data."""
    with history_lock:
        history.clear()