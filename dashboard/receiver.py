"""
Connects to DataExporter C++ module on TCP port 5001.
Reads newline-delimited JSON and pushes to shared store.
Run as a background thread inside app.py.
"""

import socket
import json
import time
import threading
from store import add_entry

SIM_HOST = "127.0.0.1"  # change to workstation IP if remote
SIM_PORT = 5001
RECONNECT_DELAY = 3.0    # seconds between reconnect attempts

def receiver_loop():
    buf = ""
    while True:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5.0)
            sock.connect((SIM_HOST, SIM_PORT))
            print(f"[receiver] connected to {SIM_HOST}:{SIM_PORT}")
            sock.settimeout(None)

            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    print("[receiver] server closed connection")
                    break
                buf += chunk.decode("utf-8", errors="replace")
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        parsed = json.loads(line)
                        add_entry(parsed)
                    except json.JSONDecodeError:
                        pass

        except (ConnectionRefusedError, OSError) as e:
            print(f"[receiver] {e}, retrying in {RECONNECT_DELAY}s")
        finally:
            try: sock.close()
            except: pass
        time.sleep(RECONNECT_DELAY)


def start_background():
    t = threading.Thread(target=receiver_loop, daemon=True)
    t.start()