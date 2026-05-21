"""
NetSim Dashboard - Fixed Version
Fixes:
 1. store never replaced — cleared in-place so all threads share one object
 2. scenario→config matching fixed to use scenarioName values from omnetpp.ini
 3. Callback outputs split per page — no missing-ID crashes
 4. TCP server always writes to the live store reference
"""

import os
import json
import time
import socket
import threading
import subprocess
import signal
import collections
from datetime import datetime
import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.graph_objs as go

# ============================================================================
# CONFIGURATION
# ============================================================================
PROJECT_DIR = "/home/opp_env/default_workspace/netsim"
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
SIM_BINARY  = os.path.join(PROJECT_DIR, "netsim")
SIM_INI     = os.path.join(PROJECT_DIR, "configs", "omnetpp.ini")
os.makedirs(RESULTS_DIR, exist_ok=True)

TCP_PORT    = 5001
MAX_HISTORY = 4000
REFRESH_MS  = 3000

ROUTERS = ["boundary", "core0", "core1", "core2", "edge0", "edge1"]
RSHORT  = {"boundary": "BDR", "core0": "C0", "core1": "C1",
           "core2": "C2", "edge0": "E0", "edge1": "E1"}
RCOLOR  = {"boundary": "#3B82F6", "core0": "#8B5CF6", "core1": "#A78BFA",
           "core2": "#C4B5FD", "edge0": "#10B981", "edge1": "#34D399"}

# scenarioName values set in omnetpp.ini  ← KEY FIX: match these exactly
HOST_CONFIGS = [
    {"config": "Realtime_Hosts5",  "scenario": "realtime_h5",
     "label": "5 per LAN (80 total)",   "total_hosts": 80,  "color": "#10B981"},
    {"config": "Realtime_Hosts10", "scenario": "realtime_h10",
     "label": "10 per LAN (160 total)", "total_hosts": 160, "color": "#3B82F6"},
    {"config": "Realtime_Hosts20", "scenario": "realtime_h20",
     "label": "20 per LAN (320 total)", "total_hosts": 320, "color": "#F59E0B"},
    {"config": "Realtime_Hosts40", "scenario": "realtime_h40",
     "label": "40 per LAN (640 total)", "total_hosts": 640, "color": "#EF4444"},
]

CONGESTION_LABELS = {
    0: "Normal", 1: "Queue Congestion", 2: "Incast",
    3: "Burst Congestion", 4: "Periodic Congestion", 5: "Link Failure",
}

# ============================================================================
# DATA STORE  — single shared instance, never replaced
# ============================================================================
class DataStore:
    def __init__(self):
        self._data = collections.deque(maxlen=MAX_HISTORY)
        self._lock = threading.Lock()

    def add(self, record):
        with self._lock:
            self._data.append(record)

    def clear(self):
        with self._lock:
            self._data.clear()

    def get_all(self):
        with self._lock:
            return list(self._data)

store = DataStore()   # ONE instance for the lifetime of the process

# ============================================================================
# TCP SERVER  — always writes to the module-level `store`
# ============================================================================
class TCPServer:
    def __init__(self, port=TCP_PORT):
        self.port = port
        self.connected = False
        self.running   = True
        self._clients  = 0

    def start(self):
        def server_loop():
            while self.running:
                srv = None
                try:
                    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    srv.bind(("0.0.0.0", self.port))
                    srv.listen(5)
                    print(f"[TCP] listening on :{self.port}")
                    while self.running:
                        srv.settimeout(1.0)
                        try:
                            client, addr = srv.accept()
                        except socket.timeout:
                            continue
                        print(f"[TCP] client connected: {addr}")
                        self.connected = True
                        self._clients += 1
                        threading.Thread(
                            target=self._handle_client,
                            args=(client,), daemon=True
                        ).start()
                except Exception as e:
                    print(f"[TCP] server error: {e}")
                    time.sleep(3)
                finally:
                    if srv:
                        try: srv.close()
                        except: pass

        threading.Thread(target=server_loop, daemon=True).start()

    def _handle_client(self, client):
        buf = ""
        try:
            while self.running:
                chunk = client.recv(8192)
                if not chunk:
                    break
                buf += chunk.decode("utf-8", errors="replace")
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        store.add(record)          # always the live store
                    except json.JSONDecodeError:
                        pass
        except Exception as e:
            print(f"[TCP] client error: {e}")
        finally:
            self._clients -= 1
            if self._clients == 0:
                self.connected = False
            try: client.close()
            except: pass

tcp_server = TCPServer()
tcp_server.start()

# ============================================================================
# FILE WATCHER (fallback for JSONL streaming)
# ============================================================================
class FileWatcher:
    def __init__(self):
        self._pos = {}

    def start(self):
        def loop():
            files = [
                os.path.join(RESULTS_DIR, "congestion_stream.jsonl"),
                os.path.join(RESULTS_DIR, "portscan_stream.jsonl"),
            ]
            while True:
                for fpath in files:
                    if not os.path.exists(fpath):
                        continue
                    pos = self._pos.get(fpath, 0)
                    try:
                        with open(fpath, "r") as f:
                            f.seek(pos)
                            for raw in f:
                                raw = raw.strip()
                                if raw:
                                    try:
                                        store.add(json.loads(raw))
                                    except json.JSONDecodeError:
                                        pass
                            self._pos[fpath] = f.tell()
                    except Exception:
                        pass
                time.sleep(2)

        threading.Thread(target=loop, daemon=True).start()

FileWatcher().start()

# ============================================================================
# SIMULATION MANAGER
# ============================================================================
class SimulationManager:
    def __init__(self):
        self.process    = None
        self.config     = None
        self.start_time = None

    def start(self, config_name):
        self.stop()
        store.clear()          # clear in-place — all threads still use same object

        for fname in ("congestion_stream.jsonl", "portscan_stream.jsonl"):
            fpath = os.path.join(RESULTS_DIR, fname)
            try:
                open(fpath, "w").close()   # truncate
            except Exception:
                pass

        ini = SIM_INI if os.path.exists(SIM_INI) else os.path.join(PROJECT_DIR, "omnetpp.ini")
        try:
            self.process = subprocess.Popen(
                [SIM_BINARY, "-f", ini, "-c", config_name, "-u", "Cmdenv"],
                cwd=PROJECT_DIR,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                preexec_fn=os.setsid if hasattr(os, "setsid") else None,
            )
            self.config     = config_name
            self.start_time = datetime.now()
            print(f"[SIM] started: {config_name} (pid {self.process.pid})")
            return True
        except Exception as e:
            print(f"[SIM] failed to start: {e}")
            return False

    def stop(self):
        if self.process and self.process.poll() is None:
            try:
                if hasattr(os, "killpg"):
                    os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                else:
                    self.process.terminate()
                self.process.wait(timeout=5)
            except Exception:
                try: self.process.kill()
                except: pass
        self.process = self.config = self.start_time = None

    def is_running(self):
        return self.process is not None and self.process.poll() is None

    def elapsed(self):
        if not self.start_time:
            return ""
        e = datetime.now() - self.start_time
        m, s = divmod(int(e.total_seconds()), 60)
        return f"{m:02d}:{s:02d}"

    def status_text(self):
        if self.is_running():
            return f"RUNNING — {self.config} — {self.elapsed()}"
        return "STOPPED"

sim = SimulationManager()

# ============================================================================
# DASH APP
# ============================================================================
app = dash.Dash(__name__, title="NetSim Monitor", suppress_callback_exceptions=True)

app.index_string = """
<!DOCTYPE html><html>
<head>{%metas%}<title>{%title%}</title>{%favicon%}{%css%}
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#F8FAFC}
.sidebar{position:fixed;left:0;top:0;height:100vh;width:260px;background:#fff;border-right:1px solid #E2E8F0;overflow-y:auto;display:flex;flex-direction:column}
.main-content{margin-left:260px;padding:24px}
.card{background:#fff;border-radius:12px;padding:16px;border:1px solid #E2E8F0;margin-bottom:20px}
.stat-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:20px}
button{cursor:pointer;padding:8px 16px;border:none;border-radius:8px;font-weight:600;width:100%;margin-bottom:8px}
button:hover{opacity:.9}
.nav-btn{text-align:left;background:transparent;color:#64748B;padding:10px 16px;margin:2px 0}
.nav-btn:hover,.nav-btn.active{background:#EFF6FF;color:#3B82F6}
</style>
</head>
<body>{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer>
</body></html>
"""

# ── sidebar ─────────────────────────────────────────────────────────────────
def sidebar():
    return html.Div(className="sidebar", children=[
        html.Div(style={"padding":"24px 20px","borderBottom":"1px solid #E2E8F0"}, children=[
            html.Div("NetSim",         style={"fontSize":"20px","fontWeight":"700","color":"#1E293B"}),
            html.Div("Network Monitor",style={"fontSize":"12px","color":"#64748B","marginTop":"4px"}),
        ]),
        html.Div(style={"padding":"20px"}, children=[
            html.Div("Simulation Control",
                     style={"fontSize":"12px","fontWeight":"600","color":"#64748B","marginBottom":"12px"}),
            dcc.Dropdown(id="config-select",
                         options=[{"label":c["label"],"value":c["config"]} for c in HOST_CONFIGS],
                         value="Realtime_Hosts20", clearable=False),
            html.Div(style={"height":"12px"}),
            html.Button("▶  Start Simulation", id="start-btn", n_clicks=0,
                        style={"background":"#10B981","color":"#fff"}),
            html.Button("■  Stop Simulation",  id="stop-btn",  n_clicks=0,
                        style={"background":"#EF4444","color":"#fff"}),
            html.Div(id="status-text",
                     style={"marginTop":"12px","fontSize":"11px","color":"#64748B","textAlign":"center"}),
        ]),
        html.Div(style={"padding":"20px","borderTop":"1px solid #E2E8F0"}, children=[
            html.Div("Pages",style={"fontSize":"12px","fontWeight":"600","color":"#64748B","marginBottom":"8px"}),
            html.Button("Scalability Analysis", id="nav-scalability", className="nav-btn", n_clicks=0),
            html.Button("Live Monitor + Alerts",id="nav-live",        className="nav-btn", n_clicks=0),
        ]),
        html.Div(id="sidebar-stats",
                 style={"padding":"20px","borderTop":"1px solid #E2E8F0",
                        "fontSize":"11px","color":"#64748B","marginTop":"auto"}),
    ])

# ── page layouts ─────────────────────────────────────────────────────────────
def scalability_page():
    return html.Div([
        html.Div(className="card", children=[
            html.Div("Total Throughput at Boundary — all host configs",
                     style={"fontSize":"14px","fontWeight":"600","marginBottom":"16px"}),
            dcc.Graph(id="sc-throughput", config={"displayModeBar":False}),
        ]),
        html.Div(className="card", children=[
            html.Div("Mean Bandwidth per Host",
                     style={"fontSize":"14px","fontWeight":"600","marginBottom":"16px"}),
            dcc.Graph(id="sc-perhost", config={"displayModeBar":False}),
        ]),
    ])

def live_page():
    return html.Div([
        html.Div(id="alert-banner", style={"marginBottom":"20px"}),
        html.Div(className="stat-grid", children=[
            html.Div(className="card", children=[
                html.Div("Throughput",  style={"fontSize":"12px","color":"#64748B"}),
                html.Div(id="lv-bw",   style={"fontSize":"24px","fontWeight":"700","color":"#3B82F6"}),
            ]),
            html.Div(className="card", children=[
                html.Div("Loss Rate",   style={"fontSize":"12px","color":"#64748B"}),
                html.Div(id="lv-loss",  style={"fontSize":"24px","fontWeight":"700","color":"#3B82F6"}),
            ]),
            html.Div(className="card", children=[
                html.Div("Active Flows",style={"fontSize":"12px","color":"#64748B"}),
                html.Div(id="lv-flows", style={"fontSize":"24px","fontWeight":"700","color":"#3B82F6"}),
            ]),
            html.Div(className="card", children=[
                html.Div("OSPF Pkts",   style={"fontSize":"12px","color":"#64748B"}),
                html.Div(id="lv-ospf",  style={"fontSize":"24px","fontWeight":"700","color":"#3B82F6"}),
            ]),
        ]),
        html.Div(className="card", children=[
            html.Div("Router Throughput (Mbps)",
                     style={"fontSize":"14px","fontWeight":"600","marginBottom":"16px"}),
            dcc.Graph(id="lv-throughput-graph", config={"displayModeBar":False}),
        ]),
        html.Div(className="card", children=[
            html.Div("Queue Length (bytes)",
                     style={"fontSize":"14px","fontWeight":"600","marginBottom":"16px"}),
            dcc.Graph(id="lv-queue-graph", config={"displayModeBar":False}),
        ]),
        html.Div(className="card", children=[
            html.Div("OSPF Hello Packets",
                     style={"fontSize":"14px","fontWeight":"600","marginBottom":"16px"}),
            dcc.Graph(id="lv-ospf-graph", config={"displayModeBar":False}),
        ]),
        html.Div(className="card", children=[
            html.Div("Recent Alerts",
                     style={"fontSize":"14px","fontWeight":"600","marginBottom":"16px"}),
            html.Div(id="alerts-list"),
        ]),
    ])

# ── app layout ───────────────────────────────────────────────────────────────
app.layout = html.Div([
    dcc.Interval(id="tick", interval=REFRESH_MS),
    dcc.Store(id="current-page", data="live"),
    sidebar(),
    html.Div(id="page-content", className="main-content"),
])

# ── graph builders ────────────────────────────────────────────────────────────
def _empty_fig(msg="No data yet"):
    fig = go.Figure()
    fig.update_layout(
        height=380, template="plotly_white",
        margin=dict(l=50,r=20,t=30,b=50),
        annotations=[dict(text=msg, x=0.5, y=0.5, showarrow=False,
                          xref="paper", yref="paper", font=dict(color="#94A3B8"))]
    )
    return fig

def fig_scalability(by_config):
    fig = go.Figure()
    for cfg in HOST_CONFIGS:
        rows = [d for d in by_config.get(cfg["config"], []) if d.get("node_id") == "boundary"]
        if rows:
            fig.add_trace(go.Scatter(
                x=[d["t"] for d in rows],
                y=[d.get("bw_out_mbps", 0) for d in rows],
                mode="lines", name=cfg["label"],
                line=dict(color=cfg["color"], width=2), connectgaps=True
            ))
    if not fig.data:
        return _empty_fig("Waiting for data — start a simulation")
    fig.update_layout(height=420, hovermode="x unified",
                      template="plotly_white", margin=dict(l=50,r=20,t=30,b=50))
    fig.update_xaxes(title_text="Time (s)", fixedrange=True)
    fig.update_yaxes(title_text="Throughput (Mbps)", fixedrange=True)
    return fig

def fig_perhost(by_config):
    fig = go.Figure()
    for cfg in HOST_CONFIGS:
        rows = [d for d in by_config.get(cfg["config"], []) if d.get("node_id") == "boundary"]
        if rows:
            fig.add_trace(go.Scatter(
                x=[d["t"] for d in rows],
                y=[d.get("bw_out_mbps", 0) / cfg["total_hosts"] for d in rows],
                mode="lines", name=cfg["label"],
                line=dict(color=cfg["color"], width=2), connectgaps=True
            ))
    if not fig.data:
        return _empty_fig("Waiting for data — start a simulation")
    fig.update_layout(height=420, hovermode="x unified",
                      template="plotly_white", margin=dict(l=50,r=20,t=30,b=50))
    fig.update_xaxes(title_text="Time (s)", fixedrange=True)
    fig.update_yaxes(title_text="Mbps / host", fixedrange=True)
    return fig

def _router_fig(data, y_field, ylabel, hlines=None):
    if not data:
        return _empty_fig("Waiting for data — start a simulation")
    fig = go.Figure()
    for router in ROUTERS:
        rows = [d for d in data if d.get("node_id") == router]
        if rows:
            fig.add_trace(go.Scatter(
                x=[d.get("t", 0) for d in rows],
                y=[d.get(y_field, 0) for d in rows],
                mode="lines", name=RSHORT[router],
                line=dict(color=RCOLOR[router], width=1.5), connectgaps=True
            ))
    for y_val, label in (hlines or []):
        fig.add_hline(y=y_val, line_dash="dash", line_color="#EF4444",
                      annotation_text=label)
    fig.update_layout(height=380, hovermode="x unified",
                      template="plotly_white", margin=dict(l=50,r=20,t=30,b=50))
    fig.update_xaxes(title_text="Time (s)", fixedrange=True)
    fig.update_yaxes(title_text=ylabel, fixedrange=True)
    return fig

# ── callbacks ─────────────────────────────────────────────────────────────────

@app.callback(
    Output("page-content",  "children"),
    Output("current-page",  "data"),
    Input("nav-scalability","n_clicks"),
    Input("nav-live",       "n_clicks"),
)
def switch_page(sc, lv):
    ctx = callback_context
    if ctx.triggered and ctx.triggered[0]["prop_id"].split(".")[0] == "nav-scalability":
        return scalability_page(), "scalability"
    return live_page(), "live"


@app.callback(
    Output("status-text",  "children"),
    Output("sidebar-stats","children"),
    Input("tick",       "n_intervals"),
    Input("start-btn",  "n_clicks"),
    Input("stop-btn",   "n_clicks"),
    State("config-select","value"),
    prevent_initial_call=False,
)
def sidebar_update(_, start_n, stop_n, selected):
    ctx = callback_context
    if ctx.triggered:
        trigger = ctx.triggered[0]["prop_id"].split(".")[0]
        if trigger == "start-btn" and start_n:
            sim.start(selected)
        elif trigger == "stop-btn" and stop_n:
            sim.stop()

    all_data = store.get_all()
    status = sim.status_text()
    stats  = html.Div([
        html.Div(f"Records:  {len(all_data)}"),
        html.Div(f"TCP:      {'Connected' if tcp_server.connected else 'Waiting'}"),
        html.Div(f"Updated:  {datetime.now().strftime('%H:%M:%S')}"),
    ], style={"lineHeight":"1.8"})
    return status, stats


@app.callback(
    Output("sc-throughput","figure"),
    Output("sc-perhost",   "figure"),
    Input("tick",          "n_intervals"),
    Input("current-page",  "data"),
)
def update_scalability(_, page):
    if page != "scalability":
        raise dash.exceptions.PreventUpdate

    all_data   = store.get_all()
    by_config  = {cfg["config"]: [] for cfg in HOST_CONFIGS}

    for rec in all_data:
        scenario = rec.get("scenario", "")
        for cfg in HOST_CONFIGS:
            # FIX: compare scenarioName directly
            if scenario == cfg["scenario"]:
                by_config[cfg["config"]].append(rec)
                break

    return fig_scalability(by_config), fig_perhost(by_config)


@app.callback(
    Output("alert-banner",      "children"),
    Output("lv-bw",             "children"),
    Output("lv-loss",           "children"),
    Output("lv-flows",          "children"),
    Output("lv-ospf",           "children"),
    Output("lv-throughput-graph","figure"),
    Output("lv-queue-graph",    "figure"),
    Output("lv-ospf-graph",     "figure"),
    Output("alerts-list",       "children"),
    Input("tick",               "n_intervals"),
    Input("current-page",       "data"),
)
def update_live(_, page):
    if page != "live":
        raise dash.exceptions.PreventUpdate

    all_data = store.get_all()

    # FIX: filter by exact scenarioName from omnetpp.ini
    data_20  = [d for d in all_data if d.get("scenario") == "realtime_h20"]
    alerts   = [d for d in data_20 if d.get("congestion_label", 0) > 0]

    # Quick-stat KPIs
    boundary_recent = [d for d in data_20 if d.get("node_id") == "boundary"][-5:]
    if boundary_recent:
        avg_bw   = sum(d.get("bw_out_mbps",   0) for d in boundary_recent) / len(boundary_recent)
        avg_loss = sum(d.get("pkt_loss_pct",  0) for d in boundary_recent) / len(boundary_recent)
        flows    = boundary_recent[-1].get("active_flows", 0)
        ospf     = boundary_recent[-1].get("ospf_pkts",    0)
        kpi_bw   = f"{avg_bw:.1f} Mbps"
        kpi_loss = f"{avg_loss:.2f}%"
        kpi_flows= str(flows)
        kpi_ospf = str(ospf)
    else:
        kpi_bw = kpi_loss = kpi_flows = kpi_ospf = "—"

    # Alert banner
    if alerts:
        a = alerts[-1]
        banner = html.Div(
            style={"background":"#FEF2F2","borderLeft":"4px solid #EF4444",
                   "borderRadius":"8px","padding":"12px 16px"},
            children=[
                html.Div(f"⚠ {CONGESTION_LABELS.get(a.get('congestion_label',0),'Unknown')}",
                         style={"fontWeight":"700","color":"#EF4444"}),
                html.Div(f"{a.get('node_id','?')}  |  t = {a.get('t',0):.1f} s",
                         style={"fontSize":"12px","marginTop":"4px","color":"#64748B"}),
            ]
        )
    else:
        banner = html.Div()

    # Alerts list
    if alerts:
        alerts_list = [
            html.Div(style={"padding":"8px","borderBottom":"1px solid #E2E8F0"}, children=[
                html.Div(CONGESTION_LABELS.get(a.get("congestion_label",0),"?"),
                         style={"fontWeight":"500"}),
                html.Div(f"{a.get('node_id','?')}  —  t={a.get('t',0):.1f}s",
                         style={"fontSize":"11px","color":"#64748B"}),
            ]) for a in alerts[-10:]
        ]
    else:
        alerts_list = [html.Div("No alerts detected",
                                style={"padding":"20px","textAlign":"center","color":"#94A3B8"})]

    return (
        banner, kpi_bw, kpi_loss, kpi_flows, kpi_ospf,
        _router_fig(data_20, "bw_out_mbps", "Mbps",
                    [(10000,"Core 10Gbps"),(1000,"Edge 1Gbps")]),
        _router_fig(data_20, "queue_bytes", "Bytes"),
        _router_fig(data_20, "ospf_pkts",   "OSPF Packets"),
        alerts_list,
    )

# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    print("\n" + "="*55)
    print("  NetSim Dashboard (fixed) — http://localhost:8051")
    print("="*55 + "\n")
    sim.start("Realtime_Hosts20")
    app.run(debug=False, host="0.0.0.0", port=8051)