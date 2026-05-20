"""
NetSim Dashboard - Fixed Version
Fixes:
 1. store never replaced — cleared in-place so all threads share one object
 2. scenario→config matching fixed to use scenarioName values from omnetpp.ini
 3. Callback outputs split per page — no missing-ID crashes
 4. TCP server always writes to the live store reference
"""

import os
import sys
import json
import time
import threading
import subprocess
import signal
import collections
from datetime import datetime
import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.graph_objs as go

# Import ML inference module
try:
    from models import inference
    INFERENCE_AVAILABLE = inference.load_models()
    print(f"[App] ML Inference loaded: {INFERENCE_AVAILABLE}")
except ImportError:
    INFERENCE_AVAILABLE = False
    print("[App] ML Inference not available")

# ============================================================================
# CONFIGURATION  — all paths relative to project layout
# ============================================================================
DASHBOARD_DIR  = os.path.dirname(os.path.abspath(__file__))
NETSIM_WORKSPACE = os.path.dirname(DASHBOARD_DIR)   # netsim_workspace/
PROJECT_DIR    = os.path.join(NETSIM_WORKSPACE, "netsim_project")
RESULTS_DIR    = os.path.join(PROJECT_DIR, "results")
SIM_BINARY     = os.path.join(PROJECT_DIR, "netsim")
SIM_INI        = os.path.join(PROJECT_DIR, "configs", "omnetpp.ini")
OPP_ENV        = os.path.join(os.path.dirname(sys.executable), "opp_env")
os.makedirs(RESULTS_DIR, exist_ok=True)

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
    {"config": "Realtime_Hosts80", "scenario": "realtime_h80",
     "label": "80 per LAN (1280 total)", "total_hosts": 1280, "color": "#8B5CF6"},
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
        record['_timestamp'] = time.time()
        with self._lock:
            self._data.append(record)

    def clear(self):
        with self._lock:
            self._data.clear()

    def get_all(self):
        with self._lock:
            return list(self._data)

    def _get_typed(self, type_name, seconds=130):
        cutoff = time.time() - seconds
        with self._lock:
            return [d for d in self._data if d.get('type') == type_name and d.get('_timestamp', 0) >= cutoff]

    def get_portscan_window(self, timesteps=12):
        window = self._get_typed('portscan', seconds=130)
        return window[-timesteps:] if len(window) >= timesteps else window

    def get_latest_congestion(self):
        with self._lock:
            congestion = [d for d in self._data if d.get('type') == 'congestion']
            return congestion[-1] if congestion else None

    def get_congestion_history(self, count=30):
        with self._lock:
            congestion = [d for d in self._data if d.get('type') == 'congestion']
            return congestion[-count:] if congestion else congestion

store = DataStore()

# ============================================================================
# FILE WATCHER — reads JSONL files written by DataExporter C++ module
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
        ned_path = f".:../inet-4.5.2/src"
        cmd = (f"{SIM_BINARY} -n {ned_path} -f {ini} -c {config_name} -u Cmdenv")
        logfile = open(os.path.join(PROJECT_DIR, "results", "sim_output.log"), "w")
        try:
            self.process = subprocess.Popen(
                [OPP_ENV, "run", "--no-isolated", "-c", cmd],
                cwd=PROJECT_DIR,
                stdout=logfile,
                stderr=logfile,
                preexec_fn=os.setsid if hasattr(os, "setsid") else None,
            )
            self.config     = config_name
            self.start_time = datetime.now()
            print(f"[SIM] started: {config_name} (pid {self.process.pid})")
            return True
        except Exception as e:
            print(f"[SIM] failed to start: {e}")
            return False
        finally:
            logfile.close()

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
            html.Div(id="detection-alert"),
        ]),
        html.Div(style={"padding":"20px","borderTop":"1px solid #E2E8F0"}, children=[
            html.Div("Pages",style={"fontSize":"12px","fontWeight":"600","color":"#64748B","marginBottom":"8px"}),
            html.Button("Scalability Analysis", id="nav-scalability", className="nav-btn", n_clicks=0),
            html.Button("Live Monitor + Alerts",id="nav-live",        className="nav-btn", n_clicks=0),
            html.Button("Anomaly Detection (80H)",id="nav-anomaly",   className="nav-btn", n_clicks=0),
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
    Input("nav-anomaly",    "n_clicks"),
)
def switch_page(sc, lv, an):
    ctx = dash.callback_context
    if ctx.triggered:
        trigger = ctx.triggered[0]["prop_id"].split(".")[0]
        if trigger == "nav-scalability":
            return scalability_page(), "scalability"
        elif trigger == "nav-anomaly":
            return anomaly_page(), "anomaly"
    return live_page(), "live"


@app.callback(
    Output("status-text",  "children"),
    Output("sidebar-stats","children"),
    Output("detection-alert", "children"),
    Input("tick",       "n_intervals"),
    Input("start-btn",  "n_clicks"),
    Input("stop-btn",   "n_clicks"),
    State("config-select","value"),
    prevent_initial_call=False,
)
def sidebar_update(_, start_n, stop_n, selected):
    ctx = dash.callback_context
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
        html.Div(f"Data:     File-based"),
        html.Div(f"Updated:  {datetime.now().strftime('%H:%M:%S')}"),
    ], style={"lineHeight":"1.8"})

    # ML Detection
    detection_alert = html.Div()
    if INFERENCE_AVAILABLE:
        try:
            portscan_window = store.get_portscan_window(timesteps=12)
            congestion_record = store.get_latest_congestion()
            congestion_history = store.get_congestion_history(count=30)

            if portscan_window and congestion_record:
                det_type, label, confidence = inference.detect_all(
                    portscan_window, congestion_record, congestion_history
                )

                if det_type and label != "Normal" and confidence >= 0.5:
                    # Build alert
                    icon = "🔍" if det_type == "portscan" else "⚠️"
                    type_label = "Port Scanning" if det_type == "portscan" else "Congestion"
                    color = "#EF4444" if det_type == "portscan" else "#F59E0B"

                    detection_alert = html.Div(
                        style={
                            "background": "#FEF2F2" if det_type == "portscan" else "#FFFBEB",
                            "border": f"2px solid {color}",
                            "borderRadius": "8px",
                            "padding": "12px",
                            "marginTop": "12px",
                            "textAlign": "center"
                        },
                        children=[
                            html.Div(
                                f"{icon} ANOMALY DETECTED",
                                style={"fontWeight": "700", "color": color, "fontSize": "12px"}
                            ),
                            html.Div(
                                f"Type: {type_label}",
                                style={"fontSize": "11px", "color": "#64748B", "marginTop": "4px"}
                            ),
                            html.Div(
                                f"Class: {label} ({confidence*100:.1f}%)",
                                style={"fontSize": "11px", "color": "#64748B", "marginTop": "2px"}
                            ),
                        ]
                    )
        except Exception as e:
            print(f"[Detection] Error: {e}")

    return status, stats, detection_alert


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

# ── anomaly detection page ─────────────────────────────────────────────────────
def anomaly_page():
    return html.Div([
        html.Div(className="card", children=[
            html.Div("Anomaly Detection — 80 Hosts per LAN (1280 total)",
                     style={"fontSize":"14px","fontWeight":"600","marginBottom":"16px"}),
            html.Div("This page monitors the high-density scenario for anomaly detection",
                     style={"fontSize":"12px","color":"#64748B","marginBottom":"16px"}),
            dcc.Graph(id="anomaly-throughput", config={"displayModeBar":False}),
        ]),
        html.Div(className="card", children=[
            html.Div("Jitter Analysis", style={"fontSize":"14px","fontWeight":"600","marginBottom":"16px"}),
            dcc.Graph(id="anomaly-jitter", config={"displayModeBar":False}),
        ]),
        html.Div(className="card", children=[
            html.Div("Port Scan Detection (SYN Rate)", style={"fontSize":"14px","fontWeight":"600","marginBottom":"16px"}),
            dcc.Graph(id="anomaly-syn-rate", config={"displayModeBar":False}),
        ]),
        html.Div(className="card", children=[
            html.Div("Connection Anomalies", style={"fontSize":"14px","fontWeight":"600","marginBottom":"16px"}),
            html.Div(id="anomaly-alerts", style={"fontSize":"12px","color":"#64748B"}),
        ]),
    ])


@app.callback(
    Output("anomaly-throughput", "figure"),
    Output("anomaly-jitter",     "figure"),
    Output("anomaly-syn-rate",   "figure"),
    Output("anomaly-alerts",    "children"),
    Input("tick",               "n_intervals"),
    Input("current-page",       "data"),
)
def update_anomaly(_, page):
    if page != "anomaly":
        raise dash.exceptions.PreventUpdate

    all_data = store.get_all()
    data_80 = [d for d in all_data if d.get("scenario") == "realtime_h80"]

    if not data_80:
        return _empty_fig("Waiting for data — start Realtime_Hosts80"), _empty_fig(""), _empty_fig(""), html.Div("No data yet")

    # Throughput by router
    fig_bw = go.Figure()
    for router in ROUTERS:
        router_data = [d for d in data_80 if d.get("node_id") == router]
        if router_data:
            fig_bw.add_trace(go.Scatter(
                x=[d["t"] for d in router_data],
                y=[d.get("bw_out_mbps", 0) for d in router_data],
                mode="lines", name=RSHORT.get(router, router),
                line=dict(color=RCOLOR.get(router, "#888"), width=2), connectgaps=True
            ))
    fig_bw.update_layout(height=380, title="Throughput (Mbps) per Router",
                         template="plotly_white", margin=dict(l=50,r=20,t=30,b=50))
    fig_bw.update_xaxes(title_text="Time (s)", fixedrange=True)
    fig_bw.update_yaxes(title_text="Mbps", fixedrange=True)

    # Jitter
    fig_jitter = go.Figure()
    for router in ROUTERS:
        router_data = [d for d in data_80 if d.get("node_id") == router]
        if router_data:
            fig_jitter.add_trace(go.Scatter(
                x=[d["t"] for d in router_data],
                y=[d.get("jitter_ms", 0) for d in router_data],
                mode="lines", name=RSHORT.get(router, router),
                line=dict(color=RCOLOR.get(router, "#888"), width=2), connectgaps=True
            ))
    fig_jitter.update_layout(height=380, title="Jitter (ms)",
                              template="plotly_white", margin=dict(l=50,r=20,t=30,b=50))
    fig_jitter.update_xaxes(title_text="Time (s)", fixedrange=True)
    fig_jitter.update_yaxes(title_text="Jitter (ms)", fixedrange=True)

    # SYN rate (port scan indicator)
    fig_syn = go.Figure()
    # Get observer data (portscan records)
    scan_data = [d for d in all_data if d.get("type") == "portscan" and d.get("scenario") == "realtime_h80"]
    if scan_data:
        fig_syn.add_trace(go.Scatter(
            x=[d["t"] for d in scan_data],
            y=[d.get("syn_rate_per_sec", 0) for d in scan_data],
            mode="lines", name="SYN Rate",
            line=dict(color="#EF4444", width=2), connectgaps=True
        ))
    fig_syn.update_layout(height=380, title="SYN Rate (port scan detection)",
                          template="plotly_white", margin=dict(l=50,r=20,t=30,b=50))
    fig_syn.update_xaxes(title_text="Time (s)", fixedrange=True)
    fig_syn.update_yaxes(title_text="SYN/s", fixedrange=True)

    # Anomaly alerts
    alerts = []
    # Check for congestion
    cong_alerts = [d for d in data_80 if d.get("congestion_label", 0) > 0]
    if cong_alerts:
        alerts.append(html.Div(f"⚠ Congestion detected: {len(cong_alerts)} events", style={"color":"#EF4444","marginBottom":"8px"}))

    # Check for high jitter
    high_jitter = [d for d in data_80 if d.get("jitter_ms", 0) > 50]
    if high_jitter:
        alerts.append(html.Div(f"⚠ High jitter detected: {len(high_jitter)} events", style={"color":"#F59E0B","marginBottom":"8px"}))

    # Check for high SYN rate (potential scan)
    if scan_data:
        high_syn = [d for d in scan_data if d.get("syn_rate_per_sec", 0) > 10]
        if high_syn:
            alerts.append(html.Div(f"⚠ High SYN rate (possible scan): {len(high_syn)} events", style={"color":"#EF4444","marginBottom":"8px"}))

    if not alerts:
        alerts = [html.Div("✓ No anomalies detected", style={"color":"#10B981"})]

    return fig_bw, fig_jitter, fig_syn, html.Div(alerts)


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    print("\n" + "="*55)
    print("  NetSim Dashboard (fixed) — http://localhost:8051")
    print("="*55 + "\n")
    app.run(debug=False, host="0.0.0.0", port=8051)