import collections
import threading

MAX_POINTS = 1000

# Thread-safe rolling buffer
_lock = collections.deque.__init__
history = collections.deque(maxlen=MAX_POINTS)
history_lock = threading.Lock()

def add_entry(entry):
    with history_lock:
        history.append(entry)

def get_history():
    with history_lock:
        return list(history)