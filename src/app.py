"""
IDS Dashboard — src/app.py
===========================
Streamlit dashboard for an AI-powered Intrusion Detection System
trained on the UNSW-NB15 dataset using Random Forest models.

Models
------
  Model A  : OHE (One-Hot Encoding), no SMOTE       → rf_ohe.pkl
  Model B  : Label Encoding, no SMOTE               → rf_le.pkl
  Model C  : OHE + SMOTE (handles class imbalance)  → rf_smote.pkl

Usage
-----
  # From the project root:
  streamlit run src/app.py

Notes
-----
  - Live packet capture requires admin/root privileges.
  - On Windows, install Npcap from https://npcap.com/ before using Scapy.
  - Install plotly:  pip install plotly
"""

# ============================================================
# Step 1: Imports and Configuration
# ============================================================

# ── Standard library ──────────────────────────────────────────────────────────
import os
import sys
import io
import queue
import tempfile
import threading
import time
import warnings
from collections import defaultdict, deque
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# ── Third-party (always available in this project) ───────────────────────────
import joblib
import matplotlib
matplotlib.use("Agg")                   # non-interactive backend — must come first
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
)

warnings.filterwarnings("ignore")

# ============================================================
# Step 2: File System Paths
# ============================================================

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# ============================================================
# Step 3: Feature Schema — Exact Column Lists from prepare_data.py
# ============================================================

# ── Feature schema  (exact column order from prepare_data.py output) ─────────
OHE_COLS: List[str] = [
    "dur", "spkts", "dpkts", "sbytes", "dbytes", "rate", "sttl", "dttl",
    "sload", "dload", "sloss", "dloss", "sinpkt", "dinpkt", "sjit", "djit",
    "swin", "stcpb", "dtcpb", "dwin", "tcprtt", "synack", "ackdat", "smean",
    "dmean", "trans_depth", "response_body_len", "ct_srv_src", "ct_state_ttl",
    "ct_dst_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
    "is_ftp_login", "ct_ftp_cmd", "ct_flw_http_mthd", "ct_src_ltm", "ct_srv_dst",
    "is_sm_ips_ports",
    "proto_arp", "proto_ospf", "proto_other", "proto_sctp", "proto_tcp",
    "proto_udp", "proto_unas",
    "service_dhcp", "service_dns", "service_ftp", "service_ftp-data",
    "service_http", "service_irc", "service_none", "service_pop3",
    "service_radius", "service_smtp", "service_snmp", "service_ssh", "service_ssl",
    "state_CON", "state_FIN", "state_INT", "state_REQ", "state_RST", "state_other",
]
LE_COLS: List[str] = [
    "dur", "proto", "service", "state", "spkts", "dpkts", "sbytes", "dbytes",
    "rate", "sttl", "dttl", "sload", "dload", "sloss", "dloss", "sinpkt", "dinpkt",
    "sjit", "djit", "swin", "stcpb", "dtcpb", "dwin", "tcprtt", "synack", "ackdat",
    "smean", "dmean", "trans_depth", "response_body_len", "ct_srv_src",
    "ct_state_ttl", "ct_dst_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm",
    "ct_dst_src_ltm", "is_ftp_login", "ct_ftp_cmd", "ct_flw_http_mthd",
    "ct_src_ltm", "ct_srv_dst", "is_sm_ips_ports",
]
CAT_COLS = ["proto", "service", "state"]

# ============================================================
# Step 4: Domain Constants (Known Values, Port-to-Service Map, Attack Classes)
# ============================================================

# ── Domain constants ──────────────────────────────────────────────────────────
KNOWN_PROTOS   = {"arp", "ospf", "sctp", "tcp", "udp", "unas"}
KNOWN_SERVICES = {"dhcp","dns","ftp","ftp-data","http","irc","none",
                  "pop3","radius","smtp","snmp","ssh","ssl"}
KNOWN_STATES   = {"CON", "FIN", "INT", "REQ", "RST"}
PORT_SERVICE: Dict[int, str] = {
    20:"ftp-data", 21:"ftp",  22:"ssh",  25:"smtp",  53:"dns",
    67:"dhcp",     68:"dhcp", 80:"http", 110:"pop3", 143:"imap",
    161:"snmp",   162:"snmp", 194:"irc", 443:"ssl",  587:"smtp",
    993:"ssl",    995:"ssl", 1812:"radius", 1813:"radius",
    8080:"http",  8443:"ssl",
}
ATTACK_CLASSES = ["Normal","Generic","Exploits","Fuzzers","DoS",
                  "Reconnaissance","Analysis","Backdoor","Shellcode","Worms"]
MODEL_INFO: Dict[str, Dict] = {
    "Model A \u2014 OHE":        {"file":"rf_ohe.pkl",   "scaler":"scaler_ohe.pkl", "mode":"ohe"},
    "Model B \u2014 Label Enc.": {"file":"rf_le.pkl",    "scaler":"scaler_le.pkl",  "mode":"le"},
    "Model C \u2014 OHE+SMOTE":  {"file":"rf_smote.pkl", "scaler":"scaler_ohe.pkl", "mode":"ohe"},
}
APP_VERSION = "1.0.0"

# ============================================================
# Step 5: Optional Dependency Guards (Plotly, Scapy, SHAP)
# ============================================================

# ── Optional dependencies ─────────────────────────────────────────────────────
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    from scapy.all import ARP, ICMP, IP, TCP, UDP, Ether, conf, rdpcap, sniff
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# ============================================================
# Step 6: Cached Data & Model Loaders
# ============================================================

@st.cache_data(show_spinner="Loading raw dataset…")
def load_overview_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load the original UNSW-NB15 CSVs for the Dataset Overview section."""
    train = pd.read_csv(
        os.path.join(DATA_DIR, "UNSW_NB15_training-set.csv"),
        na_values=["-", " ", ""],
    )
    test = pd.read_csv(
        os.path.join(DATA_DIR, "UNSW_NB15_testing-set.csv"),
        na_values=["-", " ", ""],
    )
    train["attack_cat"] = train["attack_cat"].str.strip()
    test["attack_cat"]  = test["attack_cat"].str.strip()
    return train, test


@st.cache_data(show_spinner="Loading preprocessed features…")
def load_preprocessed() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Load preprocessed test features (output of prepare_data.py)."""
    X_ohe = pd.read_csv(os.path.join(DATA_DIR, "X_test_ohe.csv"))
    X_le  = pd.read_csv(os.path.join(DATA_DIR, "X_test_le.csv"))
    y     = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv")).squeeze()
    return X_ohe, X_le, y


@st.cache_resource(show_spinner="Loading models…")
def load_models() -> Dict:
    """Load all three Random Forest pickle files."""
    models = {}
    for name, info in MODEL_INFO.items():
        path = os.path.join(MODELS_DIR, info["file"])
        if os.path.exists(path):
            models[name] = joblib.load(path)
    return models


@st.cache_resource(show_spinner="Loading scalers & encoders…")
def load_artifacts() -> Tuple[Dict, Dict]:
    """Load MinMaxScalers and LabelEncoders saved during prepare_data.py."""
    scalers, les = {}, {}
    for fname, key in [("scaler_ohe.pkl", "ohe"), ("scaler_le.pkl", "le")]:
        p = os.path.join(MODELS_DIR, fname)
        if os.path.exists(p):
            scalers[key] = joblib.load(p)
    for col in ("proto", "service", "state"):
        p = os.path.join(MODELS_DIR, f"le_{col}.pkl")
        if os.path.exists(p):
            les[col] = joblib.load(p)
    return scalers, les


@st.cache_resource(show_spinner="Loading SHAP artifacts…")
def load_shap_artifacts():
    """Load the pre-saved SHAP TreeExplainer and 200-sample SHAP values."""
    explainer, shap_vals = None, None
    ep = os.path.join(MODELS_DIR, "shap_explainer.pkl")
    vp = os.path.join(MODELS_DIR, "shap_values_sample.npy")
    if os.path.exists(ep):
        explainer = joblib.load(ep)
    if os.path.exists(vp):
        raw = np.load(vp, allow_pickle=True)
        # np.save on a Python list produces a 0-d object array — unwrap it
        shap_vals = raw.item() if raw.ndim == 0 else raw
    return explainer, shap_vals


# ============================================================
# Step 7: Categorical Normalisation Helpers
# ============================================================

def _norm_proto(p: str) -> str:
    v = str(p).lower().strip()
    return v if v in KNOWN_PROTOS else "other"

def _norm_service(s: str) -> str:
    v = str(s).lower().strip()
    return v if v in KNOWN_SERVICES else "none"

def _norm_state(s: str) -> str:
    v = str(s).strip()
    return v if v in KNOWN_STATES else "other"

def infer_service(sport: int, dport: int) -> str:
    for p in (dport, sport):
        svc = PORT_SERVICE.get(p)
        if svc and svc in KNOWN_SERVICES:
            return svc
    return "none"

def infer_state(flags: int, proto: str) -> str:
    if proto != "tcp":
        return "CON"
    FIN, SYN, RST, ACK = 0x01, 0x02, 0x04, 0x10
    if flags & RST: return "RST"
    if flags & FIN: return "FIN"
    if (flags & SYN) and not (flags & ACK): return "INT"
    return "CON"

# ============================================================
# Step 8: Feature Preprocessing Pipeline (mirrors prepare_data.py)
# ============================================================

def preprocess_features(df_raw: pd.DataFrame, mode: str = "ohe") -> np.ndarray:
    """
    Apply the same preprocessing pipeline used in prepare_data.py.

    Parameters
    ----------
    df_raw : DataFrame with raw UNSW-NB15 feature columns (proto/service/state
             may be raw strings).
    mode   : "ohe" → One-Hot Encoding  (Models A & C)
             "le"  → Label Encoding    (Model B)

    Returns
    -------
    np.ndarray  shape (n_samples, n_features),  dtype float32
    """
    scalers, les = load_artifacts()
    df = df_raw.copy()

    # ── Categorical cleanup ─────────────────────────────────────────────────
    df["service"] = df.get("service", pd.Series(["none"] * len(df))).fillna("none").apply(_norm_service)
    df["proto"]   = df.get("proto",   pd.Series(["other"] * len(df))).apply(_norm_proto)
    df["state"]   = df.get("state",   pd.Series(["CON"]   * len(df))).apply(_norm_state)

    # ── Ensure all numeric columns are present ──────────────────────────────
    num_cols = [c for c in LE_COLS if c not in CAT_COLS]
    for col in num_cols:
        if col not in df.columns:
            df[col] = 0

    if mode == "ohe":
        # One-Hot Encode → align → scale
        df = pd.get_dummies(df, columns=CAT_COLS)
        df = df.reindex(columns=OHE_COLS, fill_value=0)
        arr = df.values.astype(np.float64)
        if "ohe" in scalers:
            arr = scalers["ohe"].transform(arr)
    else:
        # Label Encode → align → scale
        for col in CAT_COLS:
            if col in les:
                le = les[col]
                df[col] = df[col].map(
                    lambda x: int(le.transform([x])[0]) if x in le.classes_ else -1  # noqa: B023
                )
            else:
                df[col] = 0
        df = df.reindex(columns=LE_COLS, fill_value=0)
        arr = df.values.astype(np.float64)
        if "le" in scalers:
            arr = scalers["le"].transform(arr)

    # Clip to the training range [0, 1].
    # Live traffic values that exceed the training 99th-percentile cause MinMaxScaler
    # to produce values > 1.0, pushing features out-of-distribution and spreading
    # probability mass across all 10 classes — the primary cause of low confidence scores.
    arr = np.clip(arr, 0.0, 1.0)

    return arr.astype(np.float32)


# ============================================================
# Step 9: Prediction Helpers
# ============================================================

def predict(df_raw: pd.DataFrame, model_name: str) -> np.ndarray:
    """Preprocess raw features and return predicted class labels."""
    models = load_models()
    if model_name not in models:
        raise ValueError(f"Model '{model_name}' not loaded.")
    mode = MODEL_INFO[model_name]["mode"]
    X    = preprocess_features(df_raw, mode=mode)
    return models[model_name].predict(X)


def predict_with_confidence(
    df_raw: pd.DataFrame,
    model_name: str,
    attack_threshold: float = 0.40,
    normal_threshold: float = 0.40,
) -> List[Tuple[str, float]]:
    """
    Return (label, confidence) pairs using symmetric thresholding on both
    Normal and attack classes.

    Decision logic
    --------------
    This is a 10-class problem, so even a "confident" prediction sits around
    40-60% on real-world traffic.

    Both Normal and attack classes require a minimum confidence to be accepted:

    * Normal traffic  — accepted only when P(Normal) >= normal_threshold.
      If the model is below that threshold for Normal, the flow is marked
      "Uncertain" rather than blindly trusted — blindly trusting Normal
      regardless of confidence causes false negatives on attack traffic
      that distributes its probability mass across Normal and attack classes.

    * Attack traffic  — alert only when P(attack_class) >= attack_threshold.
      This eliminates low-signal false positives.

    Anything that doesn't meet either bar is labelled "Uncertain".

    Parameters
    ----------
    df_raw             : raw feature DataFrame (proto/service/state as strings)
    model_name         : key in MODEL_INFO
    attack_threshold   : minimum P(attack_class) required to emit an attack alert
    normal_threshold   : minimum P(Normal) required to accept a Normal prediction
    """
    models = load_models()
    if model_name not in models:
        raise ValueError(f"Model '{model_name}' not loaded.")
    mode    = MODEL_INFO[model_name]["mode"]
    X       = preprocess_features(df_raw, mode=mode)
    model   = models[model_name]
    probas  = model.predict_proba(X)
    classes = model.classes_

    results: List[Tuple[str, float]] = []
    for p in probas:
        idx   = int(np.argmax(p))
        conf  = float(p[idx])
        label = classes[idx]

        if label == "Normal" and conf >= normal_threshold:
            # Normal is most likely AND model is confident enough.
            final = "Normal"
        elif label != "Normal" and conf >= attack_threshold:
            # Attack class is dominant enough to alert.
            final = label
        else:
            # Model is uncertain — do not emit Normal or attack, flag for review.
            final = "Uncertain"

        results.append((final, conf))
    return results


# ============================================================
# Step 10: Connection Tracker — Rolling ct_* Feature Computation
# ============================================================
# UNSW-NB15 is a flow-level dataset.  Features that need app-layer parsing
# (is_ftp_login, ct_ftp_cmd, ct_flw_http_mthd, trans_depth,
#  response_body_len, sloss, dloss) are set to 0.
#
# Previously unfixed root causes of false positives:
#   1. dur ≈ 1e-6  →  rate/sload/dload explode to 10⁶-10⁹  (out-of-dist)
#   2. ct_* set to raw packet count — completely wrong semantics
# Both are fixed: ConnectionTracker uses a 100-flow deque + MIN_DUR floor.

class ConnectionTracker:
    """
    Sliding window of the last WINDOW completed flows.

    Used to compute the 8 UNSW-NB15 ct_* features, which count how many
    recent connections share the same src/dst IP, service, state, or TTL.
    Setting these from a rolling history (instead of a constant) dramatically
    reduces false positives on normal browsing/streaming traffic.
    """
    WINDOW = 100

    def __init__(self):
        self._hist: deque = deque(maxlen=self.WINDOW)

    def record(self, r: dict) -> None:
        """Append the key fields of a completed flow to the history."""
        self._hist.append({
            "src_ip":  r["src_ip"],
            "dst_ip":  r["dst_ip"],
            "sport":   r["sport"],
            "dport":   r["dport"],
            "service": r["service"],
            "state":   r["state"],
            "sttl":    r["sttl"],
        })

    def compute(self, r: dict) -> dict:
        """
        Compute all 8 ct_* features for a flow dict r.
        Call record(r) first so the current flow is included in the count.
        Each value is at least 1 (the flow itself).
        """
        h    = list(self._hist)
        src  = r["src_ip"];  dst   = r["dst_ip"]
        srv  = r["service"]; stt   = r["state"]
        sttl = r["sttl"];    sport = r["sport"]; dport = r["dport"]
        return {
            "ct_srv_src":       max(1, sum(1 for c in h if c["src_ip"] == src and c["service"] == srv)),
            "ct_state_ttl":     max(1, sum(1 for c in h if c["state"]  == stt and c["sttl"]    == sttl)),
            "ct_dst_ltm":       max(1, sum(1 for c in h if c["dst_ip"] == dst)),
            "ct_src_dport_ltm": max(1, sum(1 for c in h if c["src_ip"] == src and c["dport"]   == dport)),
            "ct_dst_sport_ltm": max(1, sum(1 for c in h if c["dst_ip"] == dst and c["sport"]   == sport)),
            "ct_dst_src_ltm":   max(1, sum(1 for c in h if c["dst_ip"] == dst and c["src_ip"]  == src)),
            "ct_src_ltm":       max(1, sum(1 for c in h if c["src_ip"] == src)),
            "ct_srv_dst":       max(1, sum(1 for c in h if c["dst_ip"] == dst and c["service"] == srv)),
        }


# ============================================================
# Step 11: Flow Tracker — Packet-to-Flow Aggregation (Scapy)
# ============================================================

class FlowTracker:
    """
    Aggregates Scapy packets into bidirectional flows, then emits feature dicts
    ready for preprocess_features().

    Key design choices
    ------------------
    * MIN_DUR = 0.5 s  — prevents rate/sload/dload from blowing up for flows
      that complete in microseconds (the #1 cause of false positives).
    * ConnectionTracker — computes ct_* features from a 100-flow rolling
      history instead of a packet-count proxy (was the #2 cause).
    * MIN_PKTS = 6  — flows with fewer packets have unstable rate-based
      features (sload, dload, sinpkt, dinpkt); discarding them raises
      confidence on the flows that are classified.
    """

    FLOW_TIMEOUT = 10.0   # seconds of inactivity → flush
    MAX_FLOW_PKTS = 100   # force-emit after this many packets
    MIN_PKTS      = 6     # discard flows smaller than this (insufficient data)
    MIN_DUR       = 0.50  # seconds — floor for duration to stabilise rate/load

    def __init__(self):
        self.flows: Dict = {}
        self._ct = ConnectionTracker()

    # ── static helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _key(src_ip, dst_ip, sport, dport, proto_num):
        """Canonical 5-tuple; normalised so the lower IP is always first."""
        if src_ip <= dst_ip:
            return (src_ip, dst_ip, sport, dport, proto_num)
        return (dst_ip, src_ip, dport, sport, proto_num)

    @staticmethod
    def _mean_iat_ms(ts: List[float]) -> float:
        if len(ts) < 2: return 0.0
        return float(np.mean(np.diff(sorted(ts)))) * 1000.0

    @staticmethod
    def _jitter_ms(ts: List[float]) -> float:
        if len(ts) < 3: return 0.0
        return float(np.std(np.diff(sorted(ts)))) * 1000.0

    # ── public API ────────────────────────────────────────────────────────────

    def process_packet(self, pkt) -> List[Dict]:
        """
        Add a packet to its flow.  Returns a (possibly empty) list of completed
        flow feature dicts.
        """
        if not SCAPY_AVAILABLE or IP not in pkt:
            return []
        now = time.time()
        try:
            src_ip    = pkt[IP].src
            dst_ip    = pkt[IP].dst
            ttl       = pkt[IP].ttl
            proto_num = pkt[IP].proto
            proto_str = {6: "tcp", 17: "udp"}.get(proto_num, "other")
            if ARP in pkt: proto_str = "arp"
            sport = dport = flags = swin = seq = 0
            if TCP in pkt:
                sport = pkt[TCP].sport; dport = pkt[TCP].dport
                flags = int(pkt[TCP].flags)
                swin  = pkt[TCP].window; seq  = pkt[TCP].seq
            elif UDP in pkt:
                sport = pkt[UDP].sport; dport = pkt[UDP].dport
            pkt_len = len(pkt)
        except Exception:
            return []

        key    = self._key(src_ip, dst_ip, sport, dport, proto_num)
        is_fwd = (src_ip <= dst_ip)   # mirrors key normalisation

        if key not in self.flows:
            self.flows[key] = dict(
                src_ip=src_ip, dst_ip=dst_ip, sport=sport, dport=dport,
                proto=proto_str, service=infer_service(sport, dport),
                state=infer_state(flags, proto_str),
                start_ts=now, last_ts=now,
                spkts=0, dpkts=0, sbytes=0, dbytes=0,
                sttl=ttl, dttl=0, swin=swin, dwin=0, stcpb=seq, dtcpb=0,
                s_ts=[], d_ts=[], s_sz=[], d_sz=[],
                syn_ts=None, synack_ts=None, ack_ts=None,
            )

        rec = self.flows[key]
        rec["last_ts"] = now

        if is_fwd:
            rec["spkts"] += 1; rec["sbytes"] += pkt_len
            rec["sttl"]   = ttl
            rec["swin"]   = swin or rec["swin"]
            rec["stcpb"]  = seq  or rec["stcpb"]
            rec["s_ts"].append(now); rec["s_sz"].append(pkt_len)
        else:
            rec["dpkts"] += 1; rec["dbytes"] += pkt_len
            rec["dttl"]   = ttl
            rec["dwin"]   = swin or rec["dwin"]
            rec["dtcpb"]  = seq  or rec["dtcpb"]
            rec["d_ts"].append(now); rec["d_sz"].append(pkt_len)

        # TCP handshake timing
        SYN, ACK, FIN, RST = 0x02, 0x10, 0x01, 0x04
        if flags & SYN and not (flags & ACK) and rec["syn_ts"]    is None: rec["syn_ts"]    = now
        if flags & SYN and     (flags & ACK) and rec["synack_ts"] is None: rec["synack_ts"] = now
        if flags & ACK and rec["synack_ts"]  and rec["ack_ts"]    is None: rec["ack_ts"]    = now

        # Update state
        if   flags & RST: rec["state"] = "RST"
        elif flags & FIN: rec["state"] = "FIN"
        elif rec["spkts"] > 0 and rec["dpkts"] > 0: rec["state"] = "CON"

        done = bool(flags & (FIN | RST)) or (rec["spkts"] + rec["dpkts"] >= self.MAX_FLOW_PKTS)
        if done:
            result = self._emit(key)
            del self.flows[key]
            return [result] if result is not None else []
        return []

    def flush_expired(self) -> List[Dict]:
        """Emit and remove flows idle for longer than FLOW_TIMEOUT."""
        now     = time.time()
        expired = [k for k, v in self.flows.items() if now - v["last_ts"] > self.FLOW_TIMEOUT]
        results = [self._emit(k) for k in expired]
        for k in expired:
            del self.flows[k]
        return [r for r in results if r is not None]

    def _emit(self, key) -> Optional[Dict]:
        """
        Convert a flow record to a feature dict.
        Returns None if the flow is too small to be meaningful (< MIN_PKTS).
        """
        r   = self.flows[key]
        tot = r["spkts"] + r["dpkts"]

        # ── Gate: discard tiny flows ────────────────────────────────────────
        if tot < self.MIN_PKTS:
            return None

        # ── Fix #1: MIN_DUR floor prevents rate/load from exploding ─────────
        dur = max(r["last_ts"] - r["start_ts"], self.MIN_DUR)

        synack_t = (r["synack_ts"] - r["syn_ts"])    if r["syn_ts"]    and r["synack_ts"] else 0.0
        ackdat_t = (r["ack_ts"]    - r["synack_ts"]) if r["synack_ts"] and r["ack_ts"]    else 0.0

        # ── Fix #2: ct_* from rolling history (not a packet-count proxy) ────
        self._ct.record(r)
        ct = self._ct.compute(r)

        return {
            # display metadata (stripped before model input)
            "_src_ip": r["src_ip"],
            "_dst_ip": r["dst_ip"],
            "_sport":  r["sport"],
            "_dport":  r["dport"],
            "_ts":     datetime.now().strftime("%H:%M:%S"),
            # model features
            "dur":               dur,
            "proto":             _norm_proto(r["proto"]),
            "service":           r["service"],
            "state":             r["state"],
            "spkts":             r["spkts"],
            "dpkts":             r["dpkts"],
            "sbytes":            r["sbytes"],
            "dbytes":            r["dbytes"],
            "rate":              tot / dur,
            "sttl":              r["sttl"],
            "dttl":              r["dttl"],
            "sload":             r["sbytes"] * 8 / dur,
            "dload":             r["dbytes"] * 8 / dur,
            "sloss":             0,
            "dloss":             0,
            "sinpkt":            self._mean_iat_ms(r["s_ts"]),
            "dinpkt":            self._mean_iat_ms(r["d_ts"]),
            "sjit":              self._jitter_ms(r["s_ts"]),
            "djit":              self._jitter_ms(r["d_ts"]),
            "swin":              r["swin"],
            "stcpb":             r["stcpb"],
            "dtcpb":             r["dtcpb"],
            "dwin":              r["dwin"],
            "tcprtt":            synack_t + ackdat_t,
            "synack":            synack_t,
            "ackdat":            ackdat_t,
            "smean":             int(np.mean(r["s_sz"])) if r["s_sz"] else 0,
            "dmean":             int(np.mean(r["d_sz"])) if r["d_sz"] else 0,
            "trans_depth":       0,
            "response_body_len": 0,
            **ct,                           # ct_srv_src, ct_state_ttl, …
            "is_ftp_login":      0,
            "ct_ftp_cmd":        0,
            "ct_flw_http_mthd":  0,
            "is_sm_ips_ports":   1 if r["src_ip"] == r["dst_ip"] else 0,
        }


# ============================================================
# Step 12: PCAP Utilities (flows_to_df, extract_pcap_flows)
# ============================================================

_META_KEYS = {"_src_ip", "_dst_ip", "_sport", "_dport", "_ts"}

def flows_to_df(flow_list: List[Dict]) -> pd.DataFrame:
    """Strip display-metadata keys and return a model-ready DataFrame."""
    rows = [{k: v for k, v in f.items() if k not in _META_KEYS} for f in flow_list]
    return pd.DataFrame(rows)


def extract_pcap_flows(pcap_path: str) -> Tuple[List[Dict], pd.DataFrame]:
    """
    Parse a PCAP file and return (meta_list, feature_df).
    meta_list : list of dicts with src_ip, dst_ip, sport, dport, ts
    feature_df: DataFrame ready for preprocess_features()
    """
    if not SCAPY_AVAILABLE:
        raise RuntimeError("Scapy is not installed (pip install scapy).")

    tracker = FlowTracker()
    tracker.FLOW_TIMEOUT = 5.0
    all_flows: List[Dict] = []

    for pkt in rdpcap(pcap_path):
        all_flows.extend(tracker.process_packet(pkt))
    all_flows.extend(tracker.flush_expired())

    if not all_flows:
        return [], pd.DataFrame()

    meta = [
        {
            "src_ip": f.get("_src_ip", "?"),
            "dst_ip": f.get("_dst_ip", "?"),
            "sport":  f.get("_sport",  "?"),
            "dport":  f.get("_dport",  "?"),
            "ts":     f.get("_ts",     ""),
        }
        for f in all_flows
    ]
    return meta, flows_to_df(all_flows)


# ============================================================
# Step 13: Live Capture Manager (Background Thread + Queue)
# ============================================================

class CaptureManager:
    """
    Background packet capture using Scapy.
    Runs in a daemon thread; completed flow dicts are put into a Queue
    for the Streamlit main thread to consume on each rerun.

    Thread safety
    -------------
    Only get_flows() and is_running should be called from the main thread.
    start() / stop() may also be called from the main thread.
    """

    def __init__(self):
        self._thread:  Optional[threading.Thread] = None
        self._flusher: Optional[threading.Thread] = None
        self._stop    = threading.Event()
        self._q: queue.Queue = queue.Queue(maxsize=500)
        self._tracker = FlowTracker()

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, iface: Optional[str] = None):
        if self.is_running:
            return
        self._stop.clear()
        self._tracker = FlowTracker()
        self._thread  = threading.Thread(target=self._sniff, args=(iface,), daemon=True)
        self._flusher = threading.Thread(target=self._periodic_flush, daemon=True)
        self._thread.start()
        self._flusher.start()

    def stop(self):
        self._stop.set()

    def get_flows(self) -> List[Dict]:
        """Drain and return all pending flow dicts."""
        out: List[Dict] = []
        while True:
            try:
                out.append(self._q.get_nowait())
            except queue.Empty:
                break
        return out

    # ── private ──────────────────────────────────────────────────────────────

    def _on_packet(self, pkt):
        for flow in self._tracker.process_packet(pkt):
            try:
                self._q.put_nowait(flow)
            except queue.Full:
                pass   # drop rather than block

    def _sniff(self, iface):
        try:
            sniff(
                iface=iface,
                prn=self._on_packet,
                stop_filter=lambda _: self._stop.is_set(),
                store=False,
            )
        except Exception as exc:
            try:
                self._q.put_nowait({"_error": str(exc)})
            except queue.Full:
                pass

    def _periodic_flush(self):
        while not self._stop.is_set():
            time.sleep(3)
            for flow in self._tracker.flush_expired():
                try:
                    self._q.put_nowait(flow)
                except queue.Full:
                    pass


# ============================================================
# Step 14: Cached Model Evaluation on Test Set
# ============================================================

@st.cache_data(show_spinner="Evaluating all models on test set…")
def compute_all_metrics() -> Tuple[Dict, pd.Series]:
    """
    Run each model on its test set and return metrics.
    Result is cached — recomputation only happens on code change or cache clear.
    """
    X_ohe, X_le, y_test = load_preprocessed()
    models = load_models()

    results: Dict = {}
    for name, info in MODEL_INFO.items():
        if name not in models:
            continue
        X = X_ohe if info["mode"] == "ohe" else X_le
        y_pred = models[name].predict(X)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        results[name] = {
            "y_pred":    y_pred,
            "accuracy":  accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
            "recall":    recall_score(y_test, y_pred,    average="weighted", zero_division=0),
            "f1":        f1_score(y_test, y_pred,        average="weighted", zero_division=0),
            "f1_macro":  f1_score(y_test, y_pred,        average="macro",    zero_division=0),
            "report":    report,
        }
    return results, y_test


# ============================================================
# Step 15: UI Helpers (_alert, _metric_row, _check_plotly)
# ============================================================

def _alert(msg: str, kind: str = "error") -> None:
    """Render a coloured alert bar."""
    palette = {
        "error":   ("#c0392b", "#fde8e8"),
        "warning": ("#e67e22", "#fef3cd"),
        "info":    ("#2980b9", "#d6eaf8"),
    }
    border, bg = palette.get(kind, palette["error"])
    st.markdown(
        f'<div style="background:{bg};border-left:4px solid {border};'
        f'padding:10px 14px;border-radius:4px;margin:8px 0;">{msg}</div>',
        unsafe_allow_html=True,
    )


def _metric_row(metrics: Dict) -> None:
    """Render a row of st.metric tiles from a label→value dict."""
    cols = st.columns(len(metrics))
    for col, (label, value) in zip(cols, metrics.items()):
        col.metric(label, str(value))


def _check_plotly() -> bool:
    """Return True when Plotly is available; show an error and return False otherwise."""
    if not PLOTLY_AVAILABLE:
        st.error("Charts are unavailable. Please contact the administrator.")
        return False
    return True


# ============================================================
# Step 16: Section 1 — Dataset Overview
# ============================================================

def section_dataset_overview():
    st.title("📊 Dataset Overview")
    st.caption("Explore the UNSW-NB15 dataset used to train the IDS models.")

    if not _check_plotly():
        return

    train, test = load_overview_data()

    feature_cols = [c for c in train.columns if c not in ("id", "attack_cat", "label")]
    num_cols = train[feature_cols].select_dtypes(include="number").columns.tolist()
    cat_cols_raw = train[feature_cols].select_dtypes(exclude="number").columns.tolist()

    _metric_row({
        "Total Samples":   f"{len(train) + len(test):,}",
        "Train Samples":   f"{len(train):,}",
        "Test Samples":    f"{len(test):,}",
        "Features":        str(len(feature_cols)),
        "Attack Classes":  str(train["attack_cat"].nunique()),
    })

    st.divider()

    tab_dist, tab_types, tab_corr, tab_hist = st.tabs(
        ["Class Distribution", "Feature Types", "Correlation Heatmap", "Distributions"]
    )

    # ── Class distribution ────────────────────────────────────────────────────
    with tab_dist:
        c1, c2 = st.columns(2)
        for col, df, title in [
            (c1, train, "Training Set"),
            (c2, test,  "Testing Set"),
        ]:
            vc = df["attack_cat"].value_counts().reset_index()
            vc.columns = ["Category", "Count"]
            fig = px.bar(
                vc, x="Category", y="Count", color="Category",
                title=f"{title} — Class Distribution",
                color_discrete_sequence=px.colors.qualitative.Set3,
            )
            fig.update_layout(showlegend=False, xaxis_tickangle=-30)
            col.plotly_chart(fig, use_container_width=True)

        st.subheader("Class Counts")
        train_vc = train["attack_cat"].value_counts().rename("Train")
        test_vc  = test["attack_cat"].value_counts().rename("Test")
        st.dataframe(
            pd.concat([train_vc, test_vc], axis=1).fillna(0).astype(int),
            use_container_width=True,
        )

    # ── Feature types ─────────────────────────────────────────────────────────
    with tab_types:
        c1, c2, c3 = st.columns(3)
        c1.metric("Numerical", len(num_cols))
        c2.metric("Categorical", len(cat_cols_raw))
        c3.metric("Binary", int((train[num_cols].nunique() == 2).sum()))

        st.subheader("Categorical Features")
        for cc in cat_cols_raw:
            with st.expander(f"`{cc}` — {train[cc].nunique()} unique values"):
                vc = train[cc].value_counts().head(20)
                fig = px.bar(vc, title=f"`{cc}` value counts")
                st.plotly_chart(fig, use_container_width=True)

        st.subheader("Numerical Feature Statistics")
        st.dataframe(train[num_cols].describe().T.round(3), use_container_width=True)

    # ── Correlation heatmap ───────────────────────────────────────────────────
    with tab_corr:
        top_n = st.slider("Number of features to include", 10, min(30, len(num_cols)), 20, key="corr_n")
        sample = train[num_cols[:top_n]].copy()
        corr   = sample.corr()

        fig, ax = plt.subplots(figsize=(max(8, top_n * 0.55), max(6, top_n * 0.45)))
        sns.heatmap(corr, annot=(top_n <= 15), cmap="coolwarm", center=0,
                    linewidths=0.3, ax=ax, fmt=".1f")
        ax.set_title("Pearson Correlation Matrix")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # Highlight highly correlated pairs
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        high  = upper.stack()
        high  = high[high.abs() > 0.9].reset_index()
        high.columns = ["Feature A", "Feature B", "Correlation"]
        if not high.empty:
            st.subheader("Highly Correlated Pairs (|r| > 0.9)")
            st.dataframe(high.sort_values("Correlation", ascending=False), use_container_width=True)

    # ── Distributions ─────────────────────────────────────────────────────────
    with tab_hist:
        feat = st.selectbox("Feature", num_cols[:20], key="dist_feat")
        view = st.radio("Chart type", ["Box plot", "Histogram"], horizontal=True, key="dist_view")
        sample_df = train.sample(min(8000, len(train)), random_state=42)

        if view == "Box plot":
            fig = px.box(
                sample_df, x="attack_cat", y=feat, color="attack_cat",
                title=f"`{feat}` by Attack Category",
            )
            fig.update_layout(showlegend=False, xaxis_tickangle=-30)
        else:
            fig = px.histogram(
                sample_df, x=feat, color="attack_cat", nbins=60,
                title=f"`{feat}` histogram by Attack Category",
                barmode="overlay", opacity=0.7,
            )
            fig.update_layout(xaxis_tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)


# ============================================================
# Step 17: Section 2 — Live Traffic Detection
# ============================================================

def section_live_detection():
    st.title("🔴 Live Traffic Detection")
    st.caption("Real-time network flow classification. Flows below the confidence threshold are marked Uncertain.")

    if not SCAPY_AVAILABLE:
        st.error("Live capture requires Administrator privileges. Please restart the application as Administrator.")
        return

    if not _check_plotly():
        return

    # ── Session state init ────────────────────────────────────────────────────
    if "capture_mgr" not in st.session_state:
        st.session_state.capture_mgr = CaptureManager()
    if "live_log" not in st.session_state:
        st.session_state.live_log = []          # list of display dicts
    if "live_stats" not in st.session_state:
        st.session_state.live_stats = {"total": 0, "attacks": 0, "normal": 0}

    mgr: CaptureManager = st.session_state.capture_mgr

    # ── Controls ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.subheader("Capture Controls")
        try:
            from scapy.all import get_if_list
            ifaces = ["(auto)"] + get_if_list()
        except Exception:
            ifaces = ["(auto)"]
        iface      = st.selectbox("Interface", ifaces, key="iface")
        live_model = st.selectbox("Model", list(MODEL_INFO.keys()), key="live_model")
        attack_threshold = st.slider(
            "Attack Alert Threshold",
            min_value=0.20, max_value=0.80, value=0.40, step=0.05,
            key="live_conf",
            help=(
                "Minimum model confidence required to raise an attack alert. "
                "Flows below this threshold for the top attack class are marked Uncertain."
            ),
        )
        normal_threshold = st.slider(
            "Normal Confidence Threshold",
            min_value=0.20, max_value=0.80, value=0.40, step=0.05,
            key="live_normal_conf",
            help=(
                "Minimum model confidence required to accept a Normal prediction. "
                "Flows where Normal is the top class but below this threshold are "
                "marked Uncertain instead — this prevents attacks whose probability "
                "mass leaks into Normal from being silently ignored."
            ),
        )

    sel_iface = None if iface == "(auto)" else iface

    c1, c2, c3 = st.columns(3)
    if not mgr.is_running:
        if c1.button("▶  Start Capture", type="primary", use_container_width=True):
            try:
                mgr.start(iface=sel_iface)
                st.toast("Capture started!", icon="✅")
            except Exception as exc:
                st.error(f"Failed to start: {exc}")
    else:
        if c1.button("■  Stop Capture", type="secondary", use_container_width=True):
            mgr.stop()
            st.toast("Capture stopped.", icon="⏹️")

    if c2.button("🗑  Clear Log", use_container_width=True):
        st.session_state.live_log   = []
        st.session_state.live_stats = {"total": 0, "attacks": 0, "normal": 0}

    status_icon = "🟢 Capturing…" if mgr.is_running else "⚫ Idle"
    c3.markdown(f"**Status:** {status_icon}")

    # ── Process pending flows ─────────────────────────────────────────────────
    if mgr.is_running:
        new_flows = mgr.get_flows()
        for flow in new_flows:
            if "_error" in flow:
                st.error(f"Capture error: {flow['_error']}")
                mgr.stop()
                break

            try:
                df_flow = flows_to_df([flow])
                results = predict_with_confidence(df_flow, live_model, attack_threshold, normal_threshold)
                pred, conf = results[0]
            except Exception:
                pred, conf = "Unknown", 0.0

            is_atk      = pred not in ("Normal", "Uncertain", "Unknown")
            is_uncertain = pred == "Uncertain"
            alert_str   = (
                "🚨 ATTACK"    if is_atk       else
                "❓ Uncertain" if is_uncertain  else
                "✅ Normal"
            )
            row = {
                "Time":       flow.get("_ts", ""),
                "Src IP":     flow.get("_src_ip", "?"),
                "Src Port":   flow.get("_sport", "?"),
                "Dst IP":     flow.get("_dst_ip", "?"),
                "Dst Port":   flow.get("_dport", "?"),
                "Proto":      flow.get("proto", "?"),
                "Service":    flow.get("service", "?"),
                "State":      flow.get("state", "?"),
                "Pkts":       flow.get("spkts", 0) + flow.get("dpkts", 0),
                "Prediction": pred,
                "Confidence": f"{conf * 100:.1f}%",
                "Alert":      alert_str,
            }
            st.session_state.live_log.insert(0, row)
            st.session_state.live_log = st.session_state.live_log[:300]
            st.session_state.live_stats["total"]   += 1
            st.session_state.live_stats["attacks"] += int(is_atk)
            st.session_state.live_stats["normal"]  += int(not is_atk and not is_uncertain)

    # ── Stats row ─────────────────────────────────────────────────────────────
    s = st.session_state.live_stats
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Total Flows",  s["total"])
    mc2.metric("Attacks",      s["attacks"])
    mc3.metric("Normal",       s["normal"])

    # ── Attack alert banner ───────────────────────────────────────────────────
    log = st.session_state.live_log
    if log and "ATTACK" in log[0]["Alert"]:
        last = log[0]
        _alert(
            f"🚨 ATTACK DETECTED — {last['Prediction']} | "
            f"{last['Src IP']}:{last['Src Port']} → {last['Dst IP']}:{last['Dst Port']}"
        )

    # ── Live log table ────────────────────────────────────────────────────────
    if log:
        df_log = pd.DataFrame(log)

        def _highlight(row):
            alert = str(row.get("Alert", ""))
            if "ATTACK" in alert:
                return ["background-color:#fde8e8"] * len(row)   # red
            if "Uncertain" in alert:
                return ["background-color:#fef3cd"] * len(row)   # orange/yellow
            return [""] * len(row)

        st.dataframe(
            df_log.style.apply(_highlight, axis=1),
            use_container_width=True, height=420,
        )

        # Mini chart: attack ratio over last 50 flows
        if len(log) > 5:
            recent = pd.DataFrame(log[:50])
            vc = recent["Prediction"].value_counts().reset_index()
            vc.columns = ["Category", "Count"]
            fig = px.bar(
                vc, x="Category", y="Count", color="Category",
                title="Recent 50 Flows — Prediction Breakdown",
                color_discrete_sequence=px.colors.qualitative.Set3,
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No flows captured yet. Press **Start Capture** to begin.")

    # ── Auto-refresh every 2 s while capturing ────────────────────────────────
    if mgr.is_running:
        time.sleep(2)
        st.rerun()


# ============================================================
# Step 18: Section 3 — PCAP File Analysis
# ============================================================

def section_pcap_analysis():
    st.title("📁 PCAP File Analysis")
    st.caption("Upload a `.pcap` or `.pcapng` file to run the IDS on captured traffic.")

    if not SCAPY_AVAILABLE:
        st.error("PCAP analysis requires Administrator privileges. Please restart the application as Administrator.")
        return

    if not _check_plotly():
        return

    pcap_model = st.selectbox("Model", list(MODEL_INFO.keys()), key="pcap_model")
    uploaded   = st.file_uploader("Upload PCAP File", type=["pcap", "pcapng"])

    if uploaded is None:
        st.info("Upload a `.pcap` file to begin analysis.")
        return

    # Write to a temp file (Scapy needs a real file path)
    with tempfile.NamedTemporaryFile(suffix=".pcap", delete=False) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    try:
        with st.spinner("Parsing PCAP and extracting flows…"):
            meta, df_feat = extract_pcap_flows(tmp_path)
    except Exception as exc:
        st.error(f"Failed to parse PCAP: {exc}")
        return
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    if df_feat.empty:
        st.warning("No IP flows could be extracted from this file.")
        return

    st.success(f"Extracted **{len(df_feat):,}** flows from the PCAP.")

    with st.spinner("Running predictions…"):
        try:
            preds = predict(df_feat, pcap_model)
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            return

    # ── Build results table ───────────────────────────────────────────────────
    rows = [
        {
            "Flow #":     i + 1,
            "Time":       m["ts"],
            "Src IP":     m["src_ip"],
            "Src Port":   m["sport"],
            "Dst IP":     m["dst_ip"],
            "Dst Port":   m["dport"],
            "Prediction": p,
            "Is Attack":  p != "Normal",
        }
        for i, (m, p) in enumerate(zip(meta, preds))
    ]
    df_res = pd.DataFrame(rows)

    # ── Summary metrics ───────────────────────────────────────────────────────
    total   = len(preds)
    attacks = int((pd.Series(preds) != "Normal").sum())
    _metric_row({
        "Total Flows":      total,
        "Attacks Detected": attacks,
        "Normal Traffic":   total - attacks,
        "Attack Rate":      f"{attacks / total * 100:.1f}%",
    })

    # ── Pie chart ─────────────────────────────────────────────────────────────
    vc  = pd.Series(preds).value_counts()
    fig = px.pie(
        values=vc.values, names=vc.index,
        title="Predicted Traffic Categories",
        color_discrete_sequence=px.colors.qualitative.Set3,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Timeline ─────────────────────────────────────────────────────────────
    if attacks > 0:
        st.subheader(f"🚨 Flagged Flows ({attacks} attacks)")
        atk_df = df_res[df_res["Is Attack"]].drop(columns="Is Attack")
        st.dataframe(atk_df, use_container_width=True)

    st.subheader("All Flows")

    def _pcap_hi(row):
        return (
            ["background-color:#fde8e8"] * len(row)
            if row.get("Is Attack", False)
            else [""] * len(row)
        )

    st.dataframe(
        df_res.style.apply(_pcap_hi, axis=1),
        use_container_width=True,
    )

    # ── Download ──────────────────────────────────────────────────────────────
    buf = io.StringIO()
    df_res.to_csv(buf, index=False)
    st.download_button(
        "⬇ Download Results as CSV",
        buf.getvalue(),
        file_name="pcap_ids_results.csv",
        mime="text/csv",
    )


# ============================================================
# Step 19: Section 4 — Model Comparison & Confusion Matrix
# ============================================================

def section_model_comparison():
    st.title("⚖️ Model Comparison")
    st.caption("Performance comparison across the three trained models on the UNSW-NB15 test set.")

    if not _check_plotly():
        return

    models = load_models()
    if not models:
        st.error("No model files found in `models/`.")
        return

    with st.spinner("Evaluating models…"):
        results, y_test = compute_all_metrics()

    # ── Summary table ─────────────────────────────────────────────────────────
    st.subheader("Overall Metrics")
    st.info("**F1-macro** gives equal weight to all 10 classes — including rare attack types. **F1-weighted** reflects the real traffic distribution.", icon="ℹ️")
    summary = {
        name: {
            "Accuracy":   f"{v['accuracy']:.4f}",
            "Precision":  f"{v['precision']:.4f}",
            "Recall":     f"{v['recall']:.4f}",
            "F1-weighted":f"{v['f1']:.4f}",
            "F1-macro ⭐": f"{v['f1_macro']:.4f}",
        }
        for name, v in results.items()
    }
    df_sum = pd.DataFrame(summary).T
    st.dataframe(
        df_sum.style.highlight_max(axis=0, color="#d4f5d4"),
        use_container_width=True,
    )

    # ── Bar chart comparison ──────────────────────────────────────────────────
    st.subheader("Visual Comparison")
    df_num = pd.DataFrame(
        {name: {"Accuracy": v["accuracy"], "Precision": v["precision"],
                "Recall": v["recall"], "F1-weighted": v["f1"], "F1-macro ⭐": v["f1_macro"]}
         for name, v in results.items()}
    ).T.reset_index().melt(id_vars="index", var_name="Metric", value_name="Score")
    df_num.columns = ["Model", "Metric", "Score"]

    fig = px.bar(
        df_num, x="Metric", y="Score", color="Model",
        barmode="group",
        title="Model Performance Comparison",
        range_y=[max(0, df_num["Score"].min() - 0.02), 1.0],
        color_discrete_sequence=px.colors.qualitative.Plotly,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Per-class F1 ──────────────────────────────────────────────────────────
    st.subheader("Per-Class F1-Score")
    classes = sorted(y_test.unique())
    pc_data = {}
    for name, v in results.items():
        report = v["report"]
        pc_data[name] = {cls: report.get(cls, {}).get("f1-score", 0.0) for cls in classes}

    df_pc = pd.DataFrame(pc_data).reset_index().melt(
        id_vars="index", var_name="Model", value_name="F1"
    )
    df_pc.columns = ["Class", "Model", "F1"]

    fig2 = px.bar(
        df_pc, x="Class", y="F1", color="Model",
        barmode="group",
        title="F1-Score per Attack Class",
        color_discrete_sequence=px.colors.qualitative.Plotly,
    )
    fig2.update_layout(xaxis_tickangle=-30)
    st.plotly_chart(fig2, use_container_width=True)

    # ── Confusion Matrix ──────────────────────────────────────────────────────
    st.divider()
    st.subheader("Confusion Matrix")
    cm_model = st.selectbox("Select model", list(results.keys()), key="cm_model")
    y_pred   = results[cm_model]["y_pred"]
    cm       = confusion_matrix(y_test, y_pred, labels=classes)

    fig3, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=classes, yticklabels=classes, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {cm_model}")
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)

    with st.expander("Full Classification Report"):
        report_df = pd.DataFrame(results[cm_model]["report"]).T.round(3)
        st.dataframe(report_df, use_container_width=True)


# ============================================================
# Step 20: Section 5 — Explainability (SHAP)
# ============================================================

def section_explainability():
    st.title("🧠 Explainability — SHAP Analysis")
    st.caption(
        "SHAP (SHapley Additive exPlanations) quantifies each feature's contribution "
        "to individual predictions and reveals global model behaviour."
    )

    if not SHAP_AVAILABLE:
        st.error("Explainability module is not available on this system.")
        return

    if not _check_plotly():
        return

    explainer, shap_vals = load_shap_artifacts()
    X_ohe, _, y_test     = load_preprocessed()
    X_sample = X_ohe.iloc[:200]          # same 200 rows used in train_model.py

    if explainer is None:
        st.error("Explainability data could not be loaded.")
        return

    models = load_models()
    # The SHAP explainer was saved for the best model (OHE-based)
    # Use "Model C — OHE+SMOTE" as the reference model (typically the best)
    ref_model_name = "Model C — OHE+SMOTE"
    if ref_model_name not in models:
        ref_model_name = list(models.keys())[0]

    tab_global, tab_local = st.tabs(["Global Feature Importance", "Local Explanation"])

    # ── Global ────────────────────────────────────────────────────────────────
    with tab_global:
        st.subheader("Mean |SHAP| — Top Features")

        if shap_vals is not None:
            sv = shap_vals
            if isinstance(sv, list):
                # Older SHAP API: list of (n_samples, n_features) arrays, one per class
                mean_abs = np.mean([np.abs(s).mean(axis=0) for s in sv], axis=0)
            elif isinstance(sv, np.ndarray) and sv.ndim == 3:
                # Newer SHAP API: (n_samples, n_features, n_classes)
                mean_abs = np.abs(sv).mean(axis=(0, 2))   # → shape (n_features,)
            elif isinstance(sv, np.ndarray) and sv.ndim == 2:
                mean_abs = np.abs(sv).mean(axis=0)
            else:
                mean_abs = np.abs(sv).ravel()

            n_feat = min(len(OHE_COLS), len(mean_abs))
            imp_df = (
                pd.DataFrame({"Feature": OHE_COLS[:n_feat], "Mean |SHAP|": mean_abs[:n_feat]})
                .sort_values("Mean |SHAP|", ascending=False)
                .head(25)
            )
        else:
            st.info("Computing SHAP values, please wait…")
            with st.spinner():
                try:
                    sv_new   = explainer.shap_values(X_sample.iloc[:100])
                    mean_abs = (
                        np.mean([np.abs(s).mean(axis=0) for s in sv_new], axis=0)
                        if isinstance(sv_new, list)
                        else np.abs(sv_new).mean(axis=0)
                    )
                    n_feat = min(len(X_sample.columns), len(mean_abs))
                    imp_df = (
                        pd.DataFrame({"Feature": X_sample.columns[:n_feat].tolist(),
                                      "Mean |SHAP|": mean_abs[:n_feat]})
                        .sort_values("Mean |SHAP|", ascending=False)
                        .head(25)
                    )
                except Exception as exc:
                    st.error(f"SHAP computation failed: {exc}")
                    return

        fig = px.bar(
            imp_df, x="Mean |SHAP|", y="Feature",
            orientation="h",
            title="Top 25 Features — Mean Absolute SHAP Value",
            color="Mean |SHAP|",
            color_continuous_scale="RdYlBu_r",
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=600)
        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "A higher Mean |SHAP| value means the feature has a stronger average "
            "influence on model predictions across all attack classes."
        )

    # ── Local ─────────────────────────────────────────────────────────────────
    with tab_local:
        st.subheader("Per-Sample Explanation")

        idx = st.slider(
            "Test-set sample index (0 – 199)",
            0, len(X_sample) - 1, 0, key="shap_idx",
        )

        x_row  = X_sample.iloc[[idx]]
        true_l = str(y_test.iloc[idx])
        model  = models[ref_model_name]
        pred   = model.predict(x_row)[0]
        proba  = model.predict_proba(x_row)[0]
        cls    = list(model.classes_)

        c1, c2 = st.columns([1, 3])
        with c1:
            st.markdown(f"**True label:**  `{true_l}`")
            color = "#c0392b" if pred != "Normal" else "#27ae60"
            st.markdown(
                f'<div style="background:{color};padding:8px 12px;border-radius:5px;'
                f'color:#fff;font-weight:700;">Predicted: {pred}</div>',
                unsafe_allow_html=True,
            )
            st.markdown("**Top class probabilities:**")
            prob_df = (
                pd.DataFrame({"Class": cls, "Probability": proba})
                .sort_values("Probability", ascending=False)
                .head(5)
                .reset_index(drop=True)
            )
            st.dataframe(prob_df, use_container_width=True)

        with c2:
            # Try computing local SHAP for the selected row
            sv_local = None
            try:
                with st.spinner("Computing SHAP for this sample…"):
                    sv_local = explainer.shap_values(x_row)
            except Exception as exc:
                st.warning("Using pre-computed SHAP values for this sample.")

            if sv_local is not None:
                class_idx = cls.index(pred) if pred in cls else 0
                if isinstance(sv_local, list):
                    # Older SHAP: list of (1, n_features) per class
                    sv_row = sv_local[class_idx][0]
                elif isinstance(sv_local, np.ndarray) and sv_local.ndim == 3:
                    # Newer SHAP: (1, n_features, n_classes)
                    sv_row = sv_local[0, :, class_idx]
                else:
                    sv_row = sv_local[0]
            elif shap_vals is not None:
                sv = shap_vals
                class_idx = cls.index(pred) if pred in cls else 0
                if isinstance(sv, list):
                    # Older SHAP: list of (200, n_features) per class
                    sv_row = sv[class_idx][idx] if idx < len(sv[class_idx]) else sv[class_idx][0]
                elif isinstance(sv, np.ndarray) and sv.ndim == 3:
                    # Newer SHAP: (200, n_features, n_classes)
                    sv_row = sv[idx, :, class_idx]
                else:
                    sv_row = sv[idx]
            else:
                st.warning("No SHAP values available for local explanation.")
                return

            feat_names = X_sample.columns.tolist()
            n_f = min(len(feat_names), len(sv_row))
            contrib_df = (
                pd.DataFrame({"Feature": feat_names[:n_f], "SHAP": sv_row[:n_f]})
                .sort_values("SHAP", key=abs, ascending=False)
                .head(15)
            )
            contrib_df["Direction"] = contrib_df["SHAP"].apply(
                lambda v: "Pushes toward Attack" if v > 0 else "Pushes toward Normal"
            )

            fig = px.bar(
                contrib_df, x="SHAP", y="Feature",
                orientation="h",
                title=f"SHAP Contributions — Sample #{idx}  (Predicted: {pred})",
                color="Direction",
                color_discrete_map={
                    "Pushes toward Attack":  "#e74c3c",
                    "Pushes toward Normal":  "#2ecc71",
                },
            )
            fig.update_layout(
                yaxis={"categoryorder": "total ascending"},
                legend_title="Direction",
                height=460,
            )
            st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "Red bars increase the attack probability for the predicted class; "
            "green bars decrease it.  The magnitude shows how strongly each feature contributed."
        )


# ============================================================
# Step 21: Sidebar CSS & HTML Constants
# ============================================================

GLOBAL_CSS = """
<style>
/* ── Sidebar background ───────────────────────────────────── */
[data-testid="stSidebar"] > div:first-child {
    background: linear-gradient(180deg, #0d1117 0%, #161b27 100%);
}
[data-testid="stSidebar"] * { color: #e6edf3 !important; }

/* ── Nav radio — hide label, style each option as a card ─── */
[data-testid="stSidebar"] [data-testid="stRadio"] > label {
    display: none;
}
[data-testid="stSidebar"] [data-testid="stRadio"] > div {
    display: flex;
    flex-direction: column;
    gap: 6px;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label {
    display: flex !important;
    align-items: center;
    padding: 9px 14px;
    border-radius: 8px;
    border: 1px solid transparent;
    cursor: pointer;
    font-size: 0.92rem;
    font-weight: 500;
    letter-spacing: 0.3px;
    transition: background 0.15s, border-color 0.15s;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background: rgba(88,166,255,0.10);
    border-color: rgba(88,166,255,0.25);
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
    background: rgba(88,166,255,0.15);
    border-left: 3px solid #58a6ff;
    font-weight: 700;
}

/* ── Metric cards ─────────────────────────────────────────── */
[data-testid="metric-container"] {
    background: #1c2333;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 14px 18px;
}
</style>
"""

SIDEBAR_HEADER_HTML = """
<div style="
    text-align:center;
    padding: 18px 10px 12px;
    border-bottom: 1px solid #30363d;
    margin-bottom: 14px;
">
  <div style="font-size:2rem; line-height:1;">🛡️</div>
  <div style="font-size:1.15rem; font-weight:700; letter-spacing:1px; margin-top:6px;
              background:linear-gradient(90deg,#58a6ff,#79c0ff);
              -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
    AI-IDS
  </div>
  <div style="font-size:0.72rem; color:#8b949e; margin-top:2px; letter-spacing:0.5px;">
    INTRUSION DETECTION SYSTEM
  </div>
</div>
"""

NAV_SECTION_LABEL_HTML = """
<div style="font-size:0.68rem; font-weight:600; letter-spacing:1.2px;
            color:#8b949e; text-transform:uppercase; padding: 4px 6px 6px;">
  Navigation
</div>
"""

SIDEBAR_FOOTER_HTML = f"""
<div style="font-size:0.70rem; color:#8b949e; text-align:center; padding:8px 4px 2px;">
  v{APP_VERSION} &nbsp;·&nbsp; UNSW-NB15 &nbsp;·&nbsp; Random Forest
</div>
"""


def dep_status_html(scapy: bool, shap_ok: bool, plotly: bool) -> str:
    def badge(label: str, ok: bool) -> str:
        color  = "#2ea043" if ok else "#da3633"
        symbol = "●" if ok else "○"
        return (
            f'<span style="display:inline-flex;align-items:center;gap:4px;'
            f'font-size:0.72rem;background:#21262d;border:1px solid #30363d;'
            f'border-radius:12px;padding:2px 9px;margin:2px;">'
            f'<span style="color:{color};">{symbol}</span>{label}</span>'
        )
    return (
        '<div style="text-align:center;padding:4px 0;">'
        + badge("Scapy", scapy)
        + badge("SHAP", shap_ok)
        + badge("Plotly", plotly)
        + "</div>"
    )


def inject_css() -> None:
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


# ============================================================
# Step 22: Main Entry Point
# ============================================================

def main():
    st.set_page_config(
        page_title            = "AI-IDS Dashboard",
        page_icon             = "🛡️",
        layout                = "wide",
        initial_sidebar_state = "expanded",
    )

    # ── Inject global CSS (sidebar theme + metric cards) ─────────────────────
    inject_css()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:

        # Brand header
        st.markdown(SIDEBAR_HEADER_HTML, unsafe_allow_html=True)

        # Navigation label
        st.markdown(NAV_SECTION_LABEL_HTML, unsafe_allow_html=True)

        # Navigation — styled as card items via GLOBAL_CSS
        section = st.radio(
            "Navigation",
            [
                "📊  Dataset Overview",
                "🔴  Live Detection",
                "📁  PCAP Analysis",
                "⚖️   Model Comparison",
                "🧠  Explainability",
            ],
            label_visibility="collapsed",
        )

        st.markdown("<div style='margin-top:10px;'></div>", unsafe_allow_html=True)
        st.divider()

        # Dependency status badges
        st.markdown(
            dep_status_html(SCAPY_AVAILABLE, SHAP_AVAILABLE, PLOTLY_AVAILABLE),
            unsafe_allow_html=True,
        )

        st.divider()

        # Footer metadata
        st.markdown(SIDEBAR_FOOTER_HTML, unsafe_allow_html=True)

    # ── Route ─────────────────────────────────────────────────────────────────
    if   "Dataset"    in section: section_dataset_overview()
    elif "Live"       in section: section_live_detection()
    elif "PCAP"       in section: section_pcap_analysis()
    elif "Comparison" in section: section_model_comparison()
    elif "Explain"    in section: section_explainability()


if __name__ == "__main__":
    main()
