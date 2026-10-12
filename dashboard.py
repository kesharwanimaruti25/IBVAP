import csv
import io
import json
import logging
import os
import threading
import time
from datetime import date, datetime

import ollama
from flask import Flask, jsonify, make_response, render_template_string, request, send_from_directory

from event_integrity import verify_event_log


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ibvap")

app = Flask(__name__)
CHAT_MODEL = os.environ.get("IBVAP_CHAT_MODEL", "llama3.1:8b")
EVENTS_FILE = os.path.expanduser(os.environ.get("IBVAP_EVENTS_FILE", "~/my-ai-app/events.jsonl"))
ALERTS_FILE = os.path.expanduser(os.environ.get("IBVAP_ALERTS_FILE", "~/my-ai-app/alerts.jsonl"))
SNAPSHOT_DIR = os.path.expanduser(os.environ.get("IBVAP_SNAPSHOT_DIR", "~/my-ai-app/snapshots"))
STATE_FILE = os.path.expanduser(os.environ.get("IBVAP_STATE_FILE", "~/my-ai-app/incident_state.json"))
MAX_ALERTS = 1000
MIN_REVIEW_CONFIDENCE = 0.5
INCIDENT_GROUP_WINDOW_SECONDS = 30

ALERTS = []
INCIDENT_STATE = {}
PROCESSED_EVENT_KEYS = set()
alerts_lock = threading.Lock()
background_started = False


def event_key(alert):
    return str(alert.get("hash") or f"{alert.get('timestamp', '')}:{alert.get('track_id', '')}")


def event_time_seconds(alert):
    try:
        timestamp = datetime.fromisoformat(str(alert.get("timestamp", "")).replace("Z", "+00:00"))
        return timestamp.timestamp()
    except (TypeError, ValueError, OverflowError):
        return None


def incident_group_key(alert):
    return (
        str(alert.get("source") or ""),
        str(alert.get("type") or ""),
        str(alert.get("class") or ""),
        str(alert.get("direction") or ""),
    )


def build_incidents(events, states):
    categories = {}
    standalone = []
    for event in events:
        event_time = event_time_seconds(event)
        if event_time is None:
            standalone.append([event])
            continue
        categories.setdefault(incident_group_key(event), []).append((event_time, event))

    groups = list(standalone)
    for category_events in categories.values():
        category_events.sort(key=lambda item: item[0])
        current_group = []
        group_start_time = None
        for event_time, event in category_events:
            if current_group and event_time - group_start_time > INCIDENT_GROUP_WINDOW_SECONDS:
                groups.append(current_group)
                current_group = []
            if not current_group:
                group_start_time = event_time
            current_group.append(event)
        if current_group:
            groups.append(current_group)

    incidents = []
    for group in groups:
        best_event = max(
            group,
            key=lambda event: float(event.get("confidence") or 0)
            if isinstance(event.get("confidence"), (int, float)) else 0,
        )
        best_confidence = best_event.get("confidence")
        if not isinstance(best_confidence, (int, float)) or best_confidence < MIN_REVIEW_CONFIDENCE:
            continue

        ordered = sorted(group, key=lambda event: event_time_seconds(event) or 0)
        event_ids = [event_key(event) for event in ordered]
        member_states = [states.get(key, {}) for key in event_ids]
        reviewed = bool(member_states) and all(state.get("resolved", False) for state in member_states)
        notes = [state.get("note", "") for state in member_states if state.get("note")]
        evidence = [{
            "timestamp": event.get("timestamp"),
            "media_time_seconds": event.get("media_time_seconds"),
            "track_id": event.get("track_id"),
            "confidence": event.get("confidence"),
            "snapshot": event.get("snapshot"),
            "snapshot_sha256": event.get("snapshot_sha256"),
            "hash": event_key(event),
        } for event in ordered]

        incident = {
            **best_event,
            "incident_id": event_ids[0],
            "event_ids": event_ids,
            "observation_count": len(group),
            "evidence": evidence,
            "operator_state": {
                "resolved": reviewed,
                "note": notes[-1] if notes and len(set(notes)) == 1 else "",
            },
        }
        incidents.append(incident)

    incidents.sort(key=lambda incident: event_time_seconds(incident) or 0, reverse=True)
    return incidents


def fallback_summary(event):
    event_class = str(event.get("class", "Object")).lower()
    event_type = str(event.get("type", "crossing"))
    if event_type == "loitering":
        description = f"{event_class} remained in the monitored area beyond the configured dwell threshold"
    elif event_type.startswith("suspicious_"):
        description = f"potentially unusual {event_type.removeprefix('suspicious_').replace('_', ' ')} activity involving a {event_class} was flagged"
    else:
        description = f"{event_class} was detected {event.get('direction', 'near')} the monitored boundary"
    if event.get("night"):
        description += " during low-light conditions"
    return description + "."


def save_alert(alert):
    folder = os.path.dirname(ALERTS_FILE)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(ALERTS_FILE, "a", encoding="utf-8") as file:
        file.write(json.dumps(alert, ensure_ascii=False) + "\n")


def load_existing_alerts():
    if not os.path.exists(ALERTS_FILE):
        return
    loaded = []
    with open(ALERTS_FILE, "r", encoding="utf-8") as file:
        lines = file.readlines()
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            loaded.append(json.loads(line))
        except json.JSONDecodeError:
            logger.warning("Ignoring malformed alert record at %s:%d", ALERTS_FILE, line_number)
    with alerts_lock:
        ALERTS[:] = loaded[-MAX_ALERTS:]
        PROCESSED_EVENT_KEYS.update(event_key(alert) for alert in loaded)


def load_incident_state():
    if not os.path.exists(STATE_FILE):
        return
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as file:
            state = json.load(file)
        if not isinstance(state, dict):
            raise ValueError("Incident state must be a JSON object.")
        with alerts_lock:
            INCIDENT_STATE.update(state)
    except (OSError, json.JSONDecodeError, ValueError):
        logger.exception("Could not load operator incident state from %s", STATE_FILE)


def save_incident_state():
    folder = os.path.dirname(STATE_FILE)
    if folder:
        os.makedirs(folder, exist_ok=True)
    temporary = STATE_FILE + ".tmp"
    with open(temporary, "w", encoding="utf-8") as file:
        json.dump(INCIDENT_STATE, file, ensure_ascii=False, indent=2)
    os.replace(temporary, STATE_FILE)


def process_detector_event(event):
    key = event_key(event)
    with alerts_lock:
        if key in PROCESSED_EVENT_KEYS:
            return
        PROCESSED_EVENT_KEYS.add(key)
        enriched = {
            **event,
            "summary": fallback_summary(event),
            "summary_source": "detector-record",
            "summary_warning": "",
        }
        ALERTS.insert(0, enriched)
        del ALERTS[MAX_ALERTS:]
        try:
            save_alert(enriched)
        except OSError:
            logger.exception("Could not persist detector event as a dashboard alert.")


def watch_events(stop_event=None):
    event_file = None
    while stop_event is None or not stop_event.is_set():
        try:
            if event_file is None:
                event_file = open(EVENTS_FILE, "rb")
            line_start = event_file.tell()
            line = event_file.readline()
            if not line:
                current_stat = os.stat(EVENTS_FILE)
                open_stat = os.fstat(event_file.fileno())
                if (
                    current_stat.st_ino != open_stat.st_ino
                    or current_stat.st_dev != open_stat.st_dev
                    or current_stat.st_size < event_file.tell()
                ):
                    event_file.close()
                    event_file = None
                    continue
                if stop_event is not None:
                    stop_event.wait(0.2)
                else:
                    time.sleep(0.2)
                continue
            if not line.endswith(b"\n"):
                event_file.seek(line_start)
                if stop_event is not None:
                    stop_event.wait(0.2)
                else:
                    time.sleep(0.2)
                continue
            try:
                event = json.loads(line.decode("utf-8"))
                if not isinstance(event, dict):
                    raise ValueError("Event record must be a JSON object.")
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
                logger.exception("Could not process a detector event record.")
                continue
            process_detector_event(event)
        except FileNotFoundError:
            if event_file is not None:
                event_file.close()
                event_file = None
            if stop_event is not None:
                stop_event.wait(0.2)
            else:
                time.sleep(0.2)
        except OSError:
            logger.exception("Could not read detector event log %s.", EVENTS_FILE)
            if event_file is not None:
                event_file.close()
                event_file = None
            if stop_event is not None:
                stop_event.wait(0.2)
            else:
                time.sleep(0.2)
    if event_file is not None:
        event_file.close()


def filtered_alerts(args=None):
    args = args or request.args
    event_type = args.get("type", "").strip()
    query = args.get("q", "").strip().casefold()
    resolution = args.get("resolution", "all").strip()
    date_from = args.get("from", "").strip()
    date_to = args.get("to", "").strip()
    try:
        if date_from:
            date.fromisoformat(date_from)
        if date_to:
            date.fromisoformat(date_to)
    except ValueError as error:
        raise ValueError("Dates must use YYYY-MM-DD format.") from error
    if resolution not in ("all", "open", "resolved"):
        raise ValueError("Resolution must be all, open, or resolved.")

    with alerts_lock:
        source = list(ALERTS)
        states = dict(INCIDENT_STATE)
    matching_events = []
    for alert in source:
        timestamp = str(alert.get("timestamp", ""))
        event_date = timestamp[:10]
        if event_type and alert.get("type") != event_type:
            continue
        if date_from and event_date < date_from:
            continue
        if date_to and event_date > date_to:
            continue
        if query:
            searchable = " ".join(
                str(alert.get(key, "")) for key in ("summary", "class", "direction", "type", "track_id")
            ).casefold()
            if query not in searchable:
                continue
        matching_events.append(alert)

    results = build_incidents(matching_events, states)
    if resolution == "open":
        return [incident for incident in results if not incident["operator_state"]["resolved"]]
    if resolution == "resolved":
        return [incident for incident in results if incident["operator_state"]["resolved"]]
    return results


HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="theme-color" content="#f1f1e9">
  <title>IBVAP — Incident Desk</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #eeeee8; --paper: #f8f8f3; --paper-raised: #fffefa; --line: #d8d9d0;
      --text: #202822; --muted: #6e776f; --blue: #516d80; --cyan: #416953;
      --amber: #966224; --red: #a64e3a; --green: #3d7150; --quiet: #899188;
    }
    * { box-sizing: border-box; }
    body { margin: 0; color: var(--text); background: var(--bg); font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; font-optical-sizing: auto; }
    button, input, select, textarea { font: inherit; }
    button { cursor: pointer; }
    .shell { max-width: 1440px; margin: auto; padding: 24px 34px 48px; }
    .topbar { display: flex; align-items: center; justify-content: space-between; gap: 20px; padding-bottom: 16px; border-bottom: 1px solid var(--line); }
    .brand { display: flex; align-items: center; gap: 12px; }
    .brand-mark { display: grid; place-items: center; width: 38px; height: 38px; border: 1px solid #bdc6b9; border-radius: 5px; color: var(--paper); background: #435d4b; font-size: 18px; font-weight: 750; }
    .eyebrow { color: var(--cyan); font-size: 10px; font-weight: 700; letter-spacing: .13em; text-transform: uppercase; }
    h1 { margin: 1px 0 0; font-size: 17px; letter-spacing: -.015em; }
    .top-actions { display: flex; align-items: center; gap: 10px; }
    .privacy-pill, .status-pill, .tag { display: inline-flex; align-items: center; gap: 7px; border: 1px solid var(--line); border-radius: 4px; padding: 5px 8px; color: var(--muted); background: var(--paper); font-size: 10px; }
    .privacy-pill { color: #315840; border-color: #bdcabe; background: #e4ebe1; white-space: nowrap; }
    .dot { width: 7px; height: 7px; flex: 0 0 7px; border-radius: 50%; background: var(--quiet); }
    .dot.ok { background: var(--green); }
    .dot.warn { background: var(--amber); }
    .hero { display: flex; justify-content: space-between; align-items: end; gap: 24px; padding: 30px 0 22px; }
    .hero h2 { margin: 7px 0 5px; font-family: Georgia, "Times New Roman", serif; font-size: clamp(29px, 3.5vw, 42px); font-weight: 400; line-height: 1.06; letter-spacing: -.035em; }
    .hero p { max-width: 660px; margin: 0; color: var(--muted); }
    .hero-note { max-width: 260px; padding: 8px 0 8px 12px; border-left: 2px solid #c28d5c; color: #616a61; font-size: 11px; text-align: left; }
    .metrics { display: grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap: 1px; border: 1px solid var(--line); background: var(--line); }
    .metric, .panel { border: 1px solid var(--line); background: var(--paper); }
    .metric { min-height: 99px; padding: 15px 18px; background: var(--paper); }
    .metric-label { color: var(--muted); font-size: 10px; font-weight: 650; letter-spacing: .045em; }
    .metric-value { margin-top: 5px; font-size: 29px; line-height: 1.1; font-weight: 560; letter-spacing: -.045em; font-variant-numeric: tabular-nums; }
    .metric-hint { margin-top: 5px; color: var(--quiet); font-size: 10px; }
    .main-grid { display: grid; grid-template-columns: minmax(0,1fr) 320px; gap: 15px; margin-top: 15px; align-items: start; }
    .panel { min-width: 0; padding: 17px; border-radius: 5px; }
    .panel-head { display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-bottom: 14px; }
    .panel-title { margin: 0; font-size: 14px; font-weight: 680; letter-spacing: -.01em; }
    .panel-subtitle { margin: 3px 0 0; color: var(--muted); font-size: 11px; }
    .live { display: inline-flex; align-items: center; gap: 6px; color: var(--green); font-size: 9px; font-weight: 700; letter-spacing: .1em; text-transform: uppercase; }
    .toolbar { display: grid; grid-template-columns: minmax(150px,1.5fr) repeat(4,minmax(105px,1fr)) auto auto; gap: 7px; margin-bottom: 12px; }
    input, select, textarea { width: 100%; min-width: 0; border: 1px solid #c9cec4; border-radius: 4px; outline: 0; color: var(--text); background: var(--paper-raised); }
    input, select { height: 36px; padding: 0 9px; font-size: 11px; }
    textarea { padding: 9px; resize: vertical; font-size: 12px; line-height: 1.5; }
    input:focus-visible, select:focus-visible, textarea:focus-visible { border-color: #54735c; outline: 2px solid #54735c55; outline-offset: 1px; }
    .btn { min-height: 35px; padding: 0 11px; border: 1px solid #bdc4b9; border-radius: 4px; color: var(--text); background: #efefe9; font-size: 11px; font-weight: 620; white-space: nowrap; transition: background-color 120ms ease, color 120ms ease, transform 100ms ease; }
    .btn:hover { background: #e4e7de; }
    .btn:active, .icon-btn:active { transform: scale(.98); }
    .btn:focus-visible, .icon-btn:focus-visible { outline: 2px solid #54735c; outline-offset: 2px; }
    .btn:disabled { cursor: wait; opacity: .62; }
    .btn.primary { color: #fff; border-color: #405d48; background: #405d48; }
    .btn.primary:hover { background: #334c3a; }
    .btn.ghost { color: var(--muted); background: transparent; }
    .list-meta { display: flex; justify-content: space-between; gap: 10px; padding: 0 1px 8px; color: var(--muted); font-size: 10px; }
    .event-list { display: flex; flex-direction: column; gap: 6px; }
    .event { display: grid; grid-template-columns: 3px minmax(0,1fr) auto; gap: 11px; align-items: stretch; padding: 11px 12px; border: 1px solid var(--line); border-radius: 4px; background: var(--paper-raised); transition: border-color 140ms ease, background-color 140ms ease; }
    .event:hover { border-color: #aeb9aa; background: #fff; }
    .event-bar { border-radius: 2px; background: var(--blue); }
    .event-bar.suspicious { background: var(--red); }
    .event-bar.loitering { background: var(--amber); }
    .event-main { min-width: 0; }
    .event-topline { display: flex; align-items: center; flex-wrap: wrap; gap: 7px; }
    .event-type { font-size: 12px; font-weight: 680; text-transform: capitalize; }
    .tag { padding: 2px 6px; font-size: 9px; font-weight: 600; }
    .tag.alert { color: #87412f; border-color: #e2c2b8; background: #f5e9e4; }
    .tag.good { color: #345a3f; border-color: #c0d1c2; background: #e7eee5; }
    .event-summary { margin: 5px 0 6px; color: #3d463f; font-size: 11px; overflow-wrap: anywhere; }
    .event-meta { display: flex; flex-wrap: wrap; gap: 8px 12px; color: var(--muted); font-size: 10px; font-variant-numeric: tabular-nums; }
    .event-actions { display: flex; align-items: center; gap: 6px; }
    .icon-btn { min-height: 30px; padding: 5px 8px; border: 1px solid #c9cec4; border-radius: 4px; color: #435047; background: #f1f2ec; font-size: 10px; font-weight: 600; white-space: nowrap; transition: background-color 120ms ease, transform 100ms ease; }
    .icon-btn:hover { background: #e4e8df; }
    .empty { padding: 36px 15px; border: 1px dashed #c6ccc1; border-radius: 4px; color: var(--muted); background: #f5f5ef; text-align: center; }
    .side-stack { display: flex; flex-direction: column; gap: 12px; }
    .health-list { display: flex; flex-direction: column; gap: 10px; }
    .health-row { display: flex; align-items: center; justify-content: space-between; gap: 10px; color: #354038; font-size: 11px; }
    .health-value { display: flex; align-items: center; gap: 7px; color: var(--muted); font-size: 10px; text-align: right; }
    .divider { height: 1px; margin: 13px 0; background: var(--line); }
    .assistant-label { margin: 0 0 8px; color: var(--muted); font-size: 10px; }
    .assistant-answer { display: none; margin-top: 10px; padding: 10px; border: 1px solid var(--line); border-radius: 4px; color: #3d463f; background: #f0f1eb; font-size: 11px; white-space: pre-wrap; }
    .assistant-foot { margin-top: 8px; color: var(--muted); font-size: 9px; line-height: 1.55; }
    .privacy-note { color: var(--muted); font-size: 10px; line-height: 1.65; }
    footer { display: flex; justify-content: space-between; gap: 12px; padding: 17px 2px 0; color: var(--muted); font-size: 10px; }
    .modal { position: fixed; inset: 0; z-index: 5; display: grid; place-items: center; padding: 20px; background: #1c241dcc; opacity: 0; visibility: hidden; transition: opacity 160ms ease, visibility 160ms ease; }
    .modal.open { opacity: 1; visibility: visible; }
    .modal-card { display: grid; grid-template-columns: minmax(250px,1fr) minmax(250px,.85fr); width: min(920px,100%); max-height: 90vh; overflow: auto; border: 1px solid #c8cec3; border-radius: 6px; background: var(--paper); box-shadow: 0 18px 60px #151c1840; transform: translateY(7px); transition: transform 180ms cubic-bezier(.2,.75,.25,1); }
    .modal.open .modal-card { transform: translateY(0); }
    .modal-image { display: grid; place-items: center; min-height: 280px; padding: 12px; background: #202820; }
    .modal-image img { display: block; max-width: 100%; max-height: 70vh; object-fit: contain; border-radius: 2px; }
    .evidence-gallery { grid-column: 1 / -1; display: flex; gap: 8px; overflow-x: auto; padding: 10px; border-top: 1px solid var(--line); background: var(--paper); }
    .evidence-item { display: grid; flex: 0 0 120px; gap: 4px; padding: 4px; border: 1px solid var(--line); border-radius: 4px; color: var(--text); background: var(--paper-raised); text-align: left; }
    .evidence-item.selected { border-color: var(--green); outline: 1px solid var(--green); }
    .evidence-item img { width: 110px; height: 68px; object-fit: cover; background: #202820; }
    .evidence-item span { color: var(--muted); font-size: 9px; overflow-wrap: anywhere; }
    .modal-info { padding: 20px; }
    .modal-info h3 { margin: 0 30px 12px 0; font-family: Georgia, "Times New Roman", serif; font-size: 21px; font-weight: 400; }
    .detail-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 7px; margin: 14px 0; }
    .detail-cell { padding: 8px; border: 1px solid var(--line); border-radius: 3px; background: #f2f3ed; }
    .detail-cell small { display: block; color: var(--muted); font-size: 9px; text-transform: uppercase; letter-spacing: .06em; }
    .detail-cell strong { display: block; margin-top: 3px; font-size: 10px; font-weight: 600; overflow-wrap: anywhere; }
    .modal-close { float: right; }
    .integrity-result { margin-top: 9px; color: var(--muted); font-size: 10px; line-height: 1.55; }
    @media (prefers-reduced-motion: reduce) {
      *, *::before, *::after { scroll-behavior: auto !important; transition-duration: .01ms !important; animation-duration: .01ms !important; animation-iteration-count: 1 !important; }
      .modal-card { transform: none; }
    }
    @media (prefers-contrast: more) {
      :root { --line: #737a71; --muted: #465047; }
      .panel, .metric, .event, input, select, textarea, .btn, .icon-btn { border-width: 2px; }
    }
    @media (max-width: 1000px) { .main-grid { grid-template-columns: 1fr; } .side-stack { display: grid; grid-template-columns: repeat(2,minmax(0,1fr)); } .toolbar { grid-template-columns: repeat(3,minmax(0,1fr)); } }
    @media (max-width: 650px) { .shell { padding: 17px 13px 32px; } .topbar, .hero { align-items: flex-start; flex-direction: column; } .topbar { gap: 12px; } .brand { max-width: 100%; } .brand .eyebrow { font-size: 9px; letter-spacing: .09em; } .top-actions { width: 100%; justify-content: space-between; } .privacy-pill { font-size: 9px; } .hero { gap: 12px; padding: 24px 0 18px; } .hero-note { text-align: left; } .metrics { grid-template-columns: repeat(2,minmax(0,1fr)); gap: 1px; } .metric { min-height: 89px; padding: 12px; } .metric-value { font-size: 25px; } .toolbar { grid-template-columns: repeat(2,minmax(0,1fr)); } .toolbar .search { grid-column: 1/-1; } .side-stack { display: flex; } .event { grid-template-columns: 3px minmax(0,1fr); } .event-actions { grid-column: 2; justify-content: flex-end; } .modal-card { grid-template-columns: 1fr; } .modal-image { min-height: 160px; } footer { flex-direction: column; } }
    body { background: radial-gradient(ellipse at 12% 0%, #e5e9dc 0, transparent 34rem), #f1f1e9; }
    .shell { max-width: 1520px; padding-top: 20px; }
    .topbar { padding: 0 0 17px; border-color: #d4d8ce; }
    .brand { gap: 13px; }
    .brand-mark { width: 42px; height: 42px; border: 0; border-radius: 12px 12px 12px 4px; color: #f5f5ed; background: #284b3c; box-shadow: 0 4px 10px #284b3c22; }
    .brand h1 { font-size: 18px; letter-spacing: -.035em; }
    .eyebrow { letter-spacing: .16em; }
    .privacy-pill { border-radius: 999px; padding: 7px 11px; }
    .hero { position: relative; min-height: 196px; margin: 22px 0 18px; padding: 29px 34px; align-items: center; overflow: hidden; border: 1px solid #284b3c; border-radius: 18px; color: #f5f5ed; background: linear-gradient(118deg, #254738 0%, #315a47 62%, #3c664e 100%); box-shadow: 0 12px 30px #294b3b18; }
    .hero::after { position: absolute; top: -84px; right: 26%; width: 270px; height: 270px; border: 1px solid #ffffff12; border-radius: 50%; box-shadow: 0 0 0 35px #ffffff08, 0 0 0 70px #ffffff06; content: ""; pointer-events: none; }
    .hero > div { position: relative; z-index: 1; }
    .hero .eyebrow { color: #d6c595; }
    .hero h2 { margin-top: 8px; color: #fbfbf3; font-size: clamp(34px, 4vw, 49px); letter-spacing: -.045em; }
    .hero p { max-width: 590px; color: #d4dfd5; font-size: 13px; }
    .hero-note { max-width: 270px; padding: 12px 14px; border: 1px solid #ffffff24; border-left: 2px solid #d6c595; border-radius: 0 10px 10px 0; color: #e2e9df; background: #ffffff0a; font-size: 11px; line-height: 1.6; }
    .metrics { gap: 12px; border: 0; background: transparent; }
    .metric, .panel { border: 1px solid #e0e2d9; border-radius: 12px; background: #fbfbf6; box-shadow: 0 4px 15px #34433108; }
    .metric { position: relative; min-height: 112px; padding: 17px 19px; overflow: hidden; }
    .metric::before { position: absolute; top: 0; left: 0; width: 3px; height: 100%; background: #66846e; content: ""; }
    .metric:nth-child(2)::before { background: #b08148; }
    .metric:nth-child(3)::before { background: #b76550; }
    .metric:nth-child(4)::before { background: #708596; }
    .metric-label { letter-spacing: .11em; }
    .metric-value { margin-top: 7px; color: #263c31; font-size: 32px; font-weight: 620; letter-spacing: -.055em; }
    .metric-hint { font-size: 11px; }
    .main-grid { gap: 16px; margin-top: 16px; grid-template-columns: minmax(0,1fr) 330px; }
    .panel { padding: 19px; }
    .panel-head { margin-bottom: 16px; }
    .panel-title { font-size: 15px; letter-spacing: -.025em; }
    .panel-subtitle { font-size: 11px; }
    .toolbar { gap: 8px; margin-bottom: 14px; }
    input, select, textarea { border-color: #d5d9cf; border-radius: 8px; background: #fffefa; }
    input, select { height: 39px; padding: 0 11px; font-size: 11px; }
    textarea { padding: 11px; }
    input:focus-visible, select:focus-visible, textarea:focus-visible { border-color: #52745d; outline-color: #52745d55; }
    .btn, .icon-btn { border-radius: 8px; }
    .btn { min-height: 38px; padding: 0 13px; border-color: #d1d7cc; background: #f0f2eb; }
    .btn:hover, .icon-btn:hover { border-color: #b6c2b4; background: #e8ede5; }
    .btn.primary { border-color: #284b3c; background: #284b3c; }
    .btn.primary:hover { background: #1f3d30; }
    .list-meta { padding-bottom: 10px; font-size: 10px; }
    .event-list { gap: 8px; }
    .event { grid-template-columns: 4px minmax(0,1fr) auto; gap: 13px; padding: 13px 14px; border-color: #e1e4db; border-radius: 9px; }
    .event:hover { border-color: #aebfad; background: #fffefa; box-shadow: 0 5px 14px #3443310a; }
    .event-bar { border-radius: 99px; }
    .event-type { font-size: 13px; }
    .tag { border-radius: 999px; padding: 3px 8px; letter-spacing: .035em; }
    .event-summary { margin: 7px 0; color: #37443b; font-size: 12px; line-height: 1.55; }
    .event-meta { gap: 7px 13px; font-size: 10px; }
    .icon-btn { min-height: 34px; padding: 6px 10px; border-color: #d3d9cf; background: #f3f5ef; }
    .empty { padding: 46px 18px; border-radius: 10px; background: #f6f7f1; line-height: 1.7; }
    .side-stack { gap: 12px; }
    .health-list { gap: 12px; }
    .health-row { font-size: 12px; }
    .health-value { font-size: 11px; }
    .divider { margin: 16px 0; }
    .assistant-answer { padding: 12px; border-radius: 8px; line-height: 1.6; }
    .privacy-note { font-size: 11px; }
    footer { padding-top: 20px; font-size: 10px; }
    .modal { backdrop-filter: blur(7px); }
    .modal-card { border: 1px solid #d8ddd2; border-radius: 14px; box-shadow: 0 24px 80px #15251c50; }
    .modal-image { background: #1e2d25; }
    .modal-info { padding: 24px; }
    .modal-info h3 { font-size: 23px; }
    .detail-cell { padding: 10px; border-radius: 7px; background: #f3f5ef; }
    @media (max-width: 1000px) {
      .main-grid { grid-template-columns: 1fr; }
      .side-stack { display: grid; grid-template-columns: repeat(2,minmax(0,1fr)); }
      .toolbar { grid-template-columns: repeat(3,minmax(0,1fr)); }
    }
    @media (max-width: 650px) {
      .shell { padding: 13px 12px 28px; }
      .brand .eyebrow { white-space: nowrap; }
      .hero { min-height: 0; margin: 16px 0; padding: 23px 20px; border-radius: 14px; }
      .hero h2 { font-size: 35px; }
      .hero-note { max-width: none; }
      .metrics { gap: 8px; }
      .metric { min-height: 96px; padding: 13px 12px 12px 15px; border-radius: 10px; }
      .metric-label { font-size: 9px; }
      .metric-value { font-size: 27px; }
      .panel { padding: 14px; border-radius: 11px; }
      .toolbar { grid-template-columns: repeat(2,minmax(0,1fr)); }
      .toolbar .search { grid-column: 1/-1; }
      .event { gap: 10px; padding: 11px; border-radius: 8px; }
      .event-actions { grid-column: 2; justify-content: flex-end; }
      .side-stack { display: flex; }
      .modal { padding: 10px; }
      .modal-card { max-height: 94vh; border-radius: 11px; }
      .modal-info { padding: 17px; }
    }
    :root {
      color-scheme: dark;
      --bg: #081216; --paper: #101d22; --paper-raised: #14242a; --line: #263a40;
      --text: #edf5ef; --muted: #a2b5af; --blue: #79c6dd; --cyan: #9edcbb;
      --amber: #f0bb68; --red: #ff947d; --green: #b8ed83; --quiet: #728982;
    }
    body {
      min-height: 100vh;
      color: var(--text);
      background:
        radial-gradient(ellipse at 82% 1%, #12312c 0, transparent 31rem),
        radial-gradient(ellipse at 8% 40%, #0e2227 0, transparent 38rem),
        var(--bg);
    }
    .shell { max-width: 1600px; padding-top: 22px; }
    .topbar { border-color: #263a40; }
    .brand-mark {
      border: 1px solid #81a991;
      color: #0a1715;
      background: linear-gradient(145deg, #c4f087, #82c9a4);
      box-shadow: 0 0 30px #a9e78b20;
    }
    .eyebrow { color: #abd990; }
    .privacy-pill {
      color: #c9f3a3;
      border-color: #42644c;
      background: #162820;
    }
    .dot.ok { background: var(--green); box-shadow: 0 0 10px #b8ed8360; }
    .dot.warn { background: var(--amber); }
    .hero {
      min-height: 220px;
      margin: 24px 0 18px;
      padding: 30px 34px;
      border: 1px solid #31534b;
      border-radius: 16px;
      color: var(--text);
      background:
        linear-gradient(110deg, #102820 0%, #102b2c 56%, #10242b 100%);
      box-shadow: 0 18px 55px #00000020, inset 0 1px #ffffff08;
    }
    .hero::before {
      position: absolute;
      inset: 0;
      background-image: linear-gradient(#c4edc208 1px, transparent 1px), linear-gradient(90deg, #c4edc208 1px, transparent 1px);
      background-size: 34px 34px;
      mask-image: linear-gradient(90deg, transparent 20%, #000 100%);
      content: "";
      pointer-events: none;
    }
    .hero::after {
      top: -116px;
      right: 14%;
      width: 320px;
      height: 320px;
      border-color: #b9ee8b16;
      box-shadow: 0 0 0 28px #b9ee8b09, 0 0 0 58px #b9ee8b07, 0 0 0 88px #b9ee8b05;
    }
    .hero .eyebrow { color: #bfe78f; }
    .hero h2 { color: #f0f7ef; }
    .hero p { color: #b3c9c0; }
    .hero-note {
      border-color: #ffffff18;
      border-left-color: #c1ed84;
      border-radius: 0 10px 10px 0;
      color: #c0d2c8;
      background: #06130f50;
    }
    .demo-path {
      display: grid;
      grid-template-columns: 135px repeat(3, minmax(0, 1fr));
      align-items: center;
      gap: 12px;
      margin: 0 0 17px;
      padding: 12px 15px;
      border: 1px solid #263c40;
      border-radius: 12px;
      background: linear-gradient(105deg, #102025, #101b20);
    }
    .demo-path-label { color: #c1ed84; font-size: 9px; font-weight: 760; letter-spacing: .14em; }
    .demo-path-label span { display: block; margin-top: 3px; color: var(--quiet); font-size: 8px; letter-spacing: .09em; }
    .demo-step { display: flex; align-items: center; gap: 9px; min-width: 0; color: #dce9e1; font-size: 10px; }
    .demo-step > span { display: grid; place-items: center; width: 23px; height: 23px; flex: 0 0 23px; border: 1px solid #48664e; border-radius: 50%; color: #c1ed84; font-size: 9px; font-weight: 700; }
    .demo-step-copy { min-width: 0; }
    .demo-step strong { display: block; font-size: 10px; font-weight: 650; }
    .demo-step code { color: #9bb2aa; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 9px; }
    .metrics { gap: 10px; }
    .metric, .panel {
      border: 1px solid #263a40;
      border-radius: 12px;
      background: linear-gradient(145deg, #132228, #101b20);
      box-shadow: 0 8px 24px #00000016, inset 0 1px #ffffff05;
    }
    .metric { min-height: 108px; padding: 16px 18px; }
    .metric::before { width: 3px; background: #8dd7c4; }
    .metric:nth-child(2)::before { background: #8acde0; }
    .metric:nth-child(3)::before { background: #f0bb68; }
    .metric:nth-child(4)::before { background: #bda9ef; }
    .metric-label { color: #9cb2aa; }
    .metric-value { color: #f0f7ef; }
    .metric-hint { color: #80968e; }
    .main-grid { grid-template-columns: minmax(0, 1fr) 330px; }
    .panel { padding: 18px; }
    .panel-title { color: #edf5ef; }
    .panel-subtitle { color: #91a69e; }
    .live { color: #c1ed84; }
    input, select, textarea {
      border-color: #30464b;
      color: #e8f1eb;
      background: #0d191e;
    }
    input::placeholder, textarea::placeholder { color: #748a83; }
    input:focus-visible, select:focus-visible, textarea:focus-visible {
      border-color: #a7df7a;
      outline-color: #a7df7a55;
    }
    .btn, .icon-btn {
      border-color: #34494d;
      color: #dce9e1;
      background: #17272c;
    }
    .btn:hover, .icon-btn:hover { border-color: #55726a; background: #1b3034; }
    .btn:focus-visible, .icon-btn:focus-visible { outline-color: #b8ed83; }
    .btn.primary { border-color: #a9d881; color: #0b1713; background: #b8ed83; }
    .btn.primary:hover { border-color: #c9f29b; background: #c9f29b; }
    .btn.ghost { color: #a9bbb4; }
    .list-meta { color: #91a69e; }
    .event-list { gap: 9px; }
    .event {
      grid-template-columns: 3px 112px minmax(0, 1fr) auto;
      gap: 12px;
      padding: 10px;
      border-color: #2b3f44;
      border-radius: 10px;
      background: #122126;
    }
    .event:hover { border-color: #577568; background: #16282d; box-shadow: 0 9px 22px #00000022; }
    .event-thumb {
      position: relative;
      display: grid;
      place-items: center;
      min-height: 76px;
      overflow: hidden;
      border: 1px solid #2b4044;
      border-radius: 7px;
      color: #79918a;
      background: radial-gradient(circle at 50% 45%, #25423e, #101a1e 70%);
      font-size: 8px;
      font-weight: 700;
      letter-spacing: .1em;
    }
    .event-thumb::after {
      position: absolute;
      inset: 0;
      background: linear-gradient(180deg, transparent 55%, #0610127a);
      content: "";
      pointer-events: none;
    }
    .event-thumb img { width: 100%; height: 100%; min-height: 76px; object-fit: cover; }
    .event-type { color: #edf5ef; }
    .tag { border-color: #30464a; color: #b3c7bf; background: #19292e; }
    .tag.alert { color: #ffc0ae; border-color: #70493e; background: #37231f; }
    .tag.good { color: #caf19e; border-color: #456344; background: #1b3022; }
    .event-summary { color: #c3d2cb; }
    .event-meta { color: #8fa59c; }
    .empty {
      padding: 30px 20px;
      border-color: #3a5352;
      border-radius: 10px;
      color: #a7bab2;
      background: #101e23;
      text-align: left;
    }
    .empty-state-title { display: block; margin-bottom: 5px; color: #e8f2eb; font-size: 14px; font-weight: 680; }
    .empty-state-copy { max-width: 520px; margin: 0; color: #9bb0a8; font-size: 11px; line-height: 1.7; }
    .empty-state-command { display: inline-block; margin-top: 12px; padding: 8px 10px; border: 1px solid #385347; border-radius: 6px; color: #c7ef9c; background: #14271f; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 11px; }
    .empty-state-command button { margin-left: 10px; border: 0; color: #e3f4d2; background: transparent; font: inherit; cursor: pointer; }
    .empty-state-command button:hover { text-decoration: underline; }
    .health-row { color: #d6e3dc; }
    .health-value { color: #a1b4ac; }
    .assistant-label { color: #a1b4ac; }
    .assistant-answer { border-color: #33494c; color: #d6e3dc; background: #14242a; }
    .assistant-foot, .privacy-note, footer { color: #91a69e; }
    .divider { background: #2a3d42; }
    .modal { background: #030b0de6; }
    .modal-card { border-color: #33494c; background: #101d22; box-shadow: 0 28px 90px #00000080; }
    .modal-image { background: #061013; }
    .evidence-gallery { border-color: #293d42; background: #101d22; }
    .evidence-gallery[hidden] { display: none; }
    .snapshot-empty { display: grid; gap: 6px; max-width: 300px; padding: 20px; color: #a3b7ae; text-align: center; }
    .snapshot-empty[hidden] { display: none; }
    .snapshot-empty strong { color: #dce9e1; font-size: 13px; }
    .snapshot-empty span { font-size: 10px; line-height: 1.6; }
    .evidence-item { border-color: #33494c; color: #dce9e1; background: #14242a; }
    .evidence-item.selected { border-color: #b8ed83; outline-color: #b8ed83; }
    .evidence-item span, .detail-cell small { color: #9aafa7; }
    .detail-cell { border-color: #304449; background: #15252a; }
    .detail-cell strong { color: #e5eee8; }
    .integrity-result { color: #a1b4ac; }
    @media (max-width: 1000px) {
      .demo-path { grid-template-columns: 110px repeat(3, minmax(0, 1fr)); gap: 8px; }
      .main-grid { grid-template-columns: minmax(0, 1fr); }
    }
    @media (max-width: 650px) {
      .shell { padding: 13px 12px 28px; }
      .hero { min-height: 0; padding: 22px 18px; }
      .hero-note { max-width: none; }
      .demo-path { grid-template-columns: 1fr; gap: 10px; }
      .demo-path-label span { display: inline; margin-left: 6px; }
      .main-grid { grid-template-columns: minmax(0, 1fr); }
      .demo-step { padding-top: 2px; }
      .event { grid-template-columns: 3px 78px minmax(0, 1fr); gap: 9px; padding: 9px; }
      .event-thumb, .event-thumb img { min-height: 68px; }
      .event-actions { grid-column: 3; justify-content: flex-end; }
    }
    @media (prefers-contrast: more) {
      :root { --line: #58736d; --muted: #c4d3cd; --quiet: #aebfb7; }
      .panel, .metric, .event, input, select, textarea, .btn, .icon-btn { border-width: 2px; }
    }
  </style>
</head>
<body>
<div class="shell">
  <header class="topbar">
    <div class="brand"><div class="brand-mark">I</div><div><div class="eyebrow">IBVAP / LOCAL OPERATIONS</div><h1>Incident desk</h1></div></div>
    <div class="top-actions"><span class="privacy-pill"><span class="dot ok"></span>LOCAL-FIRST · NO CLOUD AI</span><button id="refreshBtn" class="btn ghost" type="button">Refresh data</button></div>
  </header>
  <section class="hero">
    <div><div class="eyebrow">LOCAL VIDEO ANALYTICS · HUMAN REVIEW</div><h2>See the signal.<br>Review the evidence.</h2><p>Turn video observations into a clear review queue, with snapshots and a verifiable event history on this device.</p></div>
    <div class="hero-note">Computer vision flags observations. Alerts are not proof of intent or wrongdoing.</div>
  </section>
  <section class="demo-path" aria-label="Three-step demonstration guide">
    <div class="demo-path-label">JUDGE DEMO FLOW<span>REAL EVENTS · NO MOCK DATA</span></div>
    <div class="demo-step"><span>01</span><div class="demo-step-copy"><strong>Start the detector</strong><code>python detector.py</code></div></div>
    <div class="demo-step"><span>02</span><div class="demo-step-copy"><strong>Set a virtual fence</strong><code>Draw · confirm · review</code></div></div>
    <div class="demo-step"><span>03</span><div class="demo-step-copy"><strong>Inspect &amp; verify</strong><code>Snapshot · event integrity</code></div></div>
  </section>
  <section class="metrics" aria-label="Filtered event metrics">
    <article class="metric"><div class="metric-label">VISIBLE EVENTS</div><div class="metric-value" id="totalMetric">—</div><div class="metric-hint">Matches current filters</div></article>
    <article class="metric"><div class="metric-label">BOUNDARY CROSSINGS</div><div class="metric-value" id="crossingMetric">—</div><div class="metric-hint">Recorded direction changes</div></article>
    <article class="metric"><div class="metric-label">FLAGGED ACTIVITY</div><div class="metric-value" id="suspiciousMetric">—</div><div class="metric-hint">Rule-based observations</div></article>
    <article class="metric"><div class="metric-label">LOW-LIGHT EVENTS</div><div class="metric-value" id="nightMetric">—</div><div class="metric-hint">Detector night-mode records</div></article>
  </section>
  <div class="main-grid">
    <main class="panel">
      <div class="panel-head"><div><h2 class="panel-title">Incident timeline</h2><p class="panel-subtitle">Grouped observations · operator review required</p></div><span class="live" id="syncStatus"><span class="dot ok"></span> AUTO-REFRESH · 5S</span></div>
      <div class="toolbar">
        <input id="searchFilter" class="search" type="search" placeholder="Search summary, class, track…" aria-label="Search incidents">
        <select id="typeFilter" aria-label="Filter by event type"><option value="">All event types</option><option value="crossing">Crossings</option><option value="loitering">Loitering</option><option value="suspicious_running">Suspicious running</option><option value="suspicious_converging">Suspicious converging</option></select>
        <select id="resolutionFilter" aria-label="Filter by review status"><option value="all">All review states</option><option value="open">Needs review</option><option value="resolved">Reviewed</option></select>
        <input id="fromFilter" type="date" aria-label="Start date">
        <input id="toFilter" type="date" aria-label="End date">
        <button id="clearBtn" class="btn" type="button">Clear</button>
        <button id="exportBtn" class="btn" type="button">Export CSV</button>
      </div>
      <div class="list-meta"><span id="resultCount">Loading event records…</span><span id="lastUpdated">Waiting for first sync</span></div>
      <div id="eventList" class="event-list" aria-live="polite"></div>
    </main>
    <aside class="side-stack">
      <section class="panel">
        <div class="panel-head"><div><h2 class="panel-title">System readiness</h2><p class="panel-subtitle">Live checks for this dashboard</p></div><button id="healthRefresh" class="icon-btn" type="button">Check</button></div>
        <div id="healthList" class="health-list"><div class="health-row">Checking local services…</div></div>
        <div class="divider"></div>
        <button id="integrityBtn" class="btn" type="button">Verify event log</button>
        <div id="integrityResult" class="integrity-result" role="status"></div>
      </section>
      <section class="panel">
        <div class="panel-head"><div><h2 class="panel-title">Local security assistant</h2><p class="panel-subtitle">Answers from the latest 20 alert records</p></div><span class="tag">OLLAMA</span></div>
        <p class="assistant-label">Ask a factual question about the visible alert history.</p>
        <textarea id="question" rows="3" maxlength="1000" placeholder="How many crossings were recorded today?"></textarea>
        <button id="askBtn" class="btn primary" type="button" style="width:100%;margin-top:9px">Ask about recent activity</button>
        <div id="answer" class="assistant-answer" role="status"></div>
        <div class="assistant-foot">AI-generated summaries can be inaccurate. Confirm against the event record and evidence.</div>
      </section>
      <section class="panel">
        <div class="panel-head"><div><h2 class="panel-title">Privacy by design</h2><p class="panel-subtitle">On-device processing</p></div></div>
        <div class="privacy-note">Video detection runs locally. Event summaries and Q&amp;A use the configured Ollama model on this machine. No cloud service is required by this dashboard; the model must already be installed and available locally.</div>
      </section>
    </aside>
  </div>
  <footer><span>IBVAP · Intelligent Border Video Analytics Platform</span><span>Decision support only · Review evidence before acting</span></footer>
</div>
<div id="incidentModal" class="modal" role="dialog" aria-modal="true" aria-labelledby="modalTitle">
  <div class="modal-card"><div class="modal-image"><img id="modalSnapshot" alt="Event evidence snapshot"><div id="snapshotUnavailable" class="snapshot-empty" role="status" hidden><strong>No snapshot recorded</strong><span>This observation does not have an attached image.</span></div></div><section class="modal-info">
    <button id="modalClose" class="icon-btn modal-close" type="button">Close</button><h3 id="modalTitle">Incident details</h3>
    <div id="modalSummary" class="event-summary"></div><div id="detailGrid" class="detail-grid"></div>
    <label class="assistant-label" for="operatorNote">Operator note</label><textarea id="operatorNote" rows="3" maxlength="1000" placeholder="Add a review note…"></textarea>
    <div style="display:flex;gap:8px;margin-top:9px"><button id="saveNoteBtn" class="btn primary" type="button">Save review</button><button id="resolveBtn" class="btn" type="button">Mark reviewed</button></div>
    <div id="noteStatus" class="integrity-result" role="status"></div>
  </section><div id="evidenceGallery" class="evidence-gallery" aria-label="Incident evidence snapshots"></div></div>
</div>
<script>
  const list = document.getElementById('eventList');
  let alerts = [];
  let selectedIncident = null;
  let returnFocus = null;
  let pollingAlerts = false;
  const escapeFilename = value => encodeURIComponent(value || '');
  const labelFor = value => (value || 'event').replace(/^suspicious_/, '').replaceAll('_', ' ');
  const hasActiveFilters = () => (
    document.getElementById('searchFilter').value.trim() ||
    document.getElementById('typeFilter').value ||
    document.getElementById('resolutionFilter').value !== 'all' ||
    document.getElementById('fromFilter').value ||
    document.getElementById('toFilter').value
  );

  function queryString() {
    const params = new URLSearchParams();
    const values = {q: 'searchFilter', type: 'typeFilter', resolution: 'resolutionFilter', from: 'fromFilter', to: 'toFilter'};
    for (const [key, id] of Object.entries(values)) {
      const value = document.getElementById(id).value.trim();
      if (value) params.set(key, value);
    }
    return params.toString();
  }

  function addText(parent, tag, className, text) {
    const element = document.createElement(tag);
    if (className) element.className = className;
    element.textContent = text == null ? '' : String(text);
    parent.appendChild(element);
    return element;
  }

  function renderAlerts(items) {
    alerts = items;
    list.replaceChildren();
    document.getElementById('resultCount').textContent = `${items.length} review incident${items.length === 1 ? '' : 's'} · groups by source/type/class/direction within 30 seconds · 50%+ model score`;
    document.getElementById('totalMetric').textContent = items.length;
    document.getElementById('crossingMetric').textContent = items.filter(a => a.type === 'crossing').length;
    document.getElementById('suspiciousMetric').textContent = items.filter(a => String(a.type || '').startsWith('suspicious_')).length;
    document.getElementById('nightMetric').textContent = items.filter(a => a.night === true).length;
    if (!items.length) {
      const empty = addText(list, 'div', 'empty', '');
      if (hasActiveFilters()) {
        addText(empty, 'strong', 'empty-state-title', 'No incidents match these filters');
        addText(empty, 'p', 'empty-state-copy', 'Clear a filter or widen the date range to see more of the available review queue.');
        return;
      }
      addText(empty, 'strong', 'empty-state-title', 'Your review queue is clear');
      addText(empty, 'p', 'empty-state-copy', 'There are no reviewable incidents in the current event history. Start the local detector in another terminal, draw a virtual fence, and qualifying events will appear here automatically. This dashboard shows real event records only—no sample incidents are fabricated.');
      const command = addText(empty, 'div', 'empty-state-command', '');
      addText(command, 'code', '', 'python detector.py');
      const copyButton = addText(command, 'button', '', 'Copy command');
      copyButton.type = 'button';
      copyButton.addEventListener('click', async () => {
        try {
          await navigator.clipboard.writeText('python detector.py');
          copyButton.textContent = 'Copied';
        } catch (error) {
          copyButton.textContent = 'Copy unavailable';
          copyButton.title = 'Open a terminal in the project folder and run: python detector.py';
        }
      });
      return;
    }
    for (const alert of items) {
      const card = document.createElement('article');
      card.className = 'event';
      const eventType = String(alert.type || 'crossing');
      const bar = document.createElement('div');
      bar.className = `event-bar ${eventType.startsWith('suspicious_') ? 'suspicious' : eventType === 'loitering' ? 'loitering' : ''}`;
      card.appendChild(bar);
      const thumbnailFrame = addText(card, 'div', 'event-thumb', alert.snapshot ? '' : 'NO SNAPSHOT');
      if (alert.snapshot) {
        const thumbnail = document.createElement('img');
        thumbnail.src = `/snapshots/${escapeFilename(alert.snapshot)}`;
        thumbnail.alt = `Evidence snapshot for ${labelFor(eventType)}`;
        thumbnail.loading = 'lazy';
        thumbnail.addEventListener('error', () => {
          thumbnail.remove();
          thumbnailFrame.textContent = 'SNAPSHOT UNAVAILABLE';
        }, {once: true});
        thumbnailFrame.appendChild(thumbnail);
      }
      const main = document.createElement('div');
      main.className = 'event-main';
      const topline = document.createElement('div');
      topline.className = 'event-topline';
      addText(topline, 'span', 'event-type', labelFor(eventType));
      const review = alert.operator_state && alert.operator_state.resolved;
      addText(topline, 'span', `tag ${review ? 'good' : 'alert'}`, review ? 'REVIEWED' : 'NEEDS REVIEW');
      if (alert.night) addText(topline, 'span', 'tag', 'LOW LIGHT');
      main.appendChild(topline);
      addText(main, 'p', 'event-summary', alert.summary || `${alert.class || 'Object'} detected ${alert.direction || 'near'} monitored area.`);
      const meta = document.createElement('div');
      meta.className = 'event-meta';
      addText(meta, 'span', '', new Date(alert.timestamp).toLocaleString());
      addText(meta, 'span', '', alert.class || 'Unknown class');
      addText(meta, 'span', '', `Track #${alert.track_id ?? '—'}`);
      if (alert.source) addText(meta, 'span', '', alert.source);
      if (Number.isFinite(Number(alert.media_time_seconds))) addText(meta, 'span', '', `Video ${Number(alert.media_time_seconds).toFixed(1)}s`);
      if (Number.isFinite(Number(alert.confidence))) addText(meta, 'span', '', `${Math.round(Number(alert.confidence) * 100)}% model score`);
      if (alert.observation_count > 1) addText(meta, 'span', '', `${alert.observation_count} observations · highest model score`);
      if (alert.summary_source === 'detector-record') addText(meta, 'span', '', 'AI summary unavailable');
      main.appendChild(meta);
      card.appendChild(main);
      const actions = document.createElement('div');
      actions.className = 'event-actions';
      const details = document.createElement('button');
      details.className = 'icon-btn';
      details.type = 'button';
      details.textContent = 'Evidence';
      details.addEventListener('click', () => openIncident(alert));
      actions.appendChild(details);
      card.appendChild(actions);
      list.appendChild(card);
    }
  }

  async function pollAlerts() {
    if (pollingAlerts) return;
    pollingAlerts = true;
    try {
      const response = await fetch(`/api/alerts?${queryString()}`);
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || 'Unable to load incidents.');
      renderAlerts(data.alerts);
      document.getElementById('lastUpdated').textContent = `Updated ${new Date().toLocaleTimeString()}`;
      document.getElementById('syncStatus').querySelector('.dot').className = 'dot ok';
    } catch (error) {
      list.replaceChildren();
      addText(list, 'div', 'empty', `Unable to load event records: ${error.message}`);
      document.getElementById('resultCount').textContent = 'Dashboard data unavailable';
      document.getElementById('lastUpdated').textContent = 'Could not sync';
      document.getElementById('syncStatus').querySelector('.dot').className = 'dot warn';
    } finally {
      pollingAlerts = false;
    }
  }

  async function refreshHealth() {
    const healthList = document.getElementById('healthList');
    try {
      const response = await fetch('/api/system-status');
      const status = await response.json();
      if (!response.ok) throw new Error(status.error || 'Health check failed.');
      healthList.replaceChildren();
      const rows = [
        ['Local AI service', status.ollama.available ? (status.ollama.model_available ? `${status.model} ready` : `Connected · ${status.model} missing`) : 'Unavailable', status.ollama.model_available],
        ['Event log', status.event_log.readable ? `${status.event_log.count} records` : 'Unavailable', status.event_log.readable],
        ['Evidence folder', status.snapshot_dir.readable ? 'Readable' : 'Unavailable', status.snapshot_dir.readable]
      ];
      for (const [name, value, ok] of rows) {
        const row = document.createElement('div');
        row.className = 'health-row';
        addText(row, 'span', '', name);
        const detail = document.createElement('span');
        detail.className = 'health-value';
        const dot = document.createElement('i');
        dot.className = `dot ${ok ? 'ok' : 'warn'}`;
        detail.appendChild(dot);
        addText(detail, 'span', '', value);
        row.appendChild(detail);
        healthList.appendChild(row);
      }
    } catch (error) {
      healthList.replaceChildren();
      addText(healthList, 'div', 'health-row', `Health check failed: ${error.message}`);
    }
  }

  function openIncident(alert) {
    returnFocus = document.activeElement;
    selectedIncident = alert;
    const modal = document.getElementById('incidentModal');
    document.getElementById('modalTitle').textContent = `${labelFor(alert.type)} · Track #${alert.track_id ?? '—'}${alert.observation_count > 1 ? ` · ${alert.observation_count} observations` : ''}`;
    document.getElementById('modalSummary').textContent = alert.summary || 'No summary was recorded.';
    document.getElementById('operatorNote').value = alert.operator_state?.note || '';
    document.getElementById('noteStatus').textContent = alert.operator_state?.resolved ? 'This incident is marked reviewed.' : '';
    document.getElementById('resolveBtn').textContent = alert.operator_state?.resolved ? 'Reopen review' : 'Mark reviewed';
    const image = document.getElementById('modalSnapshot');
    const snapshotUnavailable = document.getElementById('snapshotUnavailable');
    image.onload = () => {
      image.style.display = 'block';
      snapshotUnavailable.hidden = true;
    };
    image.onerror = () => {
      image.style.display = 'none';
      snapshotUnavailable.hidden = false;
      snapshotUnavailable.querySelector('strong').textContent = 'Snapshot could not be loaded';
      snapshotUnavailable.querySelector('span').textContent = 'Check that the evidence image still exists in the configured snapshot folder.';
    };
    if (alert.snapshot) {
      image.src = `/snapshots/${escapeFilename(alert.snapshot)}`;
      image.alt = `Evidence snapshot for ${labelFor(alert.type)} at ${alert.timestamp}`;
      image.style.display = 'block';
      snapshotUnavailable.hidden = true;
    } else {
      image.removeAttribute('src');
      image.style.display = 'none';
      snapshotUnavailable.hidden = false;
      snapshotUnavailable.querySelector('strong').textContent = 'No snapshot recorded';
      snapshotUnavailable.querySelector('span').textContent = 'This observation does not have an attached image.';
    }
    const gallery = document.getElementById('evidenceGallery');
    gallery.replaceChildren();
    const evidence = alert.evidence || [];
    gallery.hidden = evidence.filter(item => item.snapshot).length < 2;
    for (const item of evidence) {
      if (!item.snapshot) continue;
      const button = document.createElement('button');
      button.type = 'button';
      button.className = `evidence-item${item.snapshot === alert.snapshot ? ' selected' : ''}`;
      const thumbnail = document.createElement('img');
      thumbnail.src = `/snapshots/${escapeFilename(item.snapshot)}`;
      thumbnail.alt = `Evidence snapshot at ${item.timestamp || 'unknown time'}`;
      button.appendChild(thumbnail);
      addText(button, 'span', '', item.timestamp ? new Date(item.timestamp).toLocaleString() : 'Time unavailable');
      if (Number.isFinite(Number(item.media_time_seconds))) {
        addText(button, 'span', '', `Video ${Number(item.media_time_seconds).toFixed(1)}s`);
      }
      addText(button, 'span', '', `${Math.round(Number(item.confidence || 0) * 100)}% score · Track #${item.track_id ?? '—'}`);
      button.addEventListener('click', () => {
        image.src = thumbnail.src;
        image.alt = thumbnail.alt;
        image.style.display = 'block';
        snapshotUnavailable.hidden = true;
        gallery.querySelectorAll('.evidence-item').forEach(element => element.classList.remove('selected'));
        button.classList.add('selected');
      });
      gallery.appendChild(button);
    }
    const grid = document.getElementById('detailGrid');
    grid.replaceChildren();
    for (const [label, value] of [
      ['Timestamp', alert.timestamp], ['Event class', alert.class], ['Direction', alert.direction],
      ['Detector model score · not a probability', Number.isFinite(Number(alert.confidence)) ? `${Math.round(Number(alert.confidence) * 100)}%` : 'Not recorded'],
      ['Lighting', alert.night ? 'Low light' : 'Daylight'], ['Summary source', alert.summary_source || 'Legacy record'],
      ['Video source', alert.source || 'Not recorded'],
      ['Video time', Number.isFinite(Number(alert.media_time_seconds)) ? `${Number(alert.media_time_seconds).toFixed(3)} seconds` : 'Not recorded'],
      ['Grouped observations', alert.observation_count || 1],
      ['Event hash', alert.hash || 'Not available'], ['Evidence hash', alert.snapshot_sha256 || 'Not recorded']
    ]) {
      const cell = document.createElement('div');
      cell.className = 'detail-cell';
      addText(cell, 'small', '', label);
      addText(cell, 'strong', '', value || '—');
      grid.appendChild(cell);
    }
    modal.classList.add('open');
    setTimeout(() => {
      if (modal.classList.contains('open')) document.getElementById('modalClose').focus();
    }, 25);
  }

  function closeIncident() {
    document.getElementById('incidentModal').classList.remove('open');
    if (returnFocus instanceof HTMLElement) returnFocus.focus();
  }

  async function updateIncidentState(toggleReview = false) {
    if (!selectedIncident) return;
    const note = document.getElementById('operatorNote').value.trim();
    const isResolved = Boolean(selectedIncident.operator_state?.resolved);
    const resolve = toggleReview ? !isResolved : isResolved;
    const status = document.getElementById('noteStatus');
    status.textContent = 'Saving review…';
    try {
      const response = await fetch(`/api/alerts/${encodeURIComponent(selectedIncident.incident_id || selectedIncident.hash || `${selectedIncident.timestamp}:${selectedIncident.track_id}`)}/state`, {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({note, resolved: resolve, event_ids: selectedIncident.event_ids || []})
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || 'Could not save review.');
      selectedIncident.operator_state = data.operator_state;
      document.getElementById('resolveBtn').textContent = resolve ? 'Reopen review' : 'Mark reviewed';
      status.textContent = 'Review saved on this device.';
      await pollAlerts();
    } catch (error) {
      status.textContent = `Could not save: ${error.message}`;
    }
  }

  async function askAssistant() {
    const input = document.getElementById('question');
    const button = document.getElementById('askBtn');
    const answer = document.getElementById('answer');
    const prompt = input.value.trim();
    if (!prompt) return;
    button.disabled = true;
    answer.style.display = 'block';
    answer.textContent = 'Checking the recent event records…';
    try {
      const response = await fetch('/ask-system', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({prompt})
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || 'Assistant request failed.');
      answer.textContent = data.response;
    } catch (error) {
      answer.textContent = `Assistant unavailable: ${error.message}`;
    } finally {
      button.disabled = false;
    }
  }

  async function verifyIntegrity() {
    const button = document.getElementById('integrityBtn');
    const result = document.getElementById('integrityResult');
    button.disabled = true;
    result.textContent = 'Verifying event hashes and available snapshots…';
    try {
      const response = await fetch('/api/integrity');
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || 'Verification failed.');
      const snapshotText = `${data.snapshots_verified} snapshots verified · ${data.snapshots_unverifiable} without a recorded digest`;
      result.textContent = `${data.chain_valid ? 'Event chain valid' : 'Integrity issues found'} · ${data.events_checked} events · ${snapshotText}${data.issues.length ? ` · ${data.issues.length} issue(s)` : ''}`;
      result.style.color = data.valid ? 'var(--green)' : 'var(--amber)';
    } catch (error) {
      result.textContent = `Verification unavailable: ${error.message}`;
      result.style.color = 'var(--red)';
    } finally {
      button.disabled = false;
    }
  }

  for (const id of ['searchFilter', 'typeFilter', 'resolutionFilter', 'fromFilter', 'toFilter']) {
    document.getElementById(id).addEventListener(id === 'searchFilter' ? 'input' : 'change', () => {
      clearTimeout(window.filterTimer);
      window.filterTimer = setTimeout(pollAlerts, id === 'searchFilter' ? 250 : 0);
    });
  }
  document.getElementById('clearBtn').addEventListener('click', () => {
    for (const id of ['searchFilter', 'typeFilter', 'resolutionFilter', 'fromFilter', 'toFilter']) document.getElementById(id).value = id === 'resolutionFilter' ? 'all' : '';
    pollAlerts();
  });
  document.getElementById('exportBtn').addEventListener('click', () => { window.location.href = `/api/alerts.csv?${queryString()}`; });
  document.getElementById('refreshBtn').addEventListener('click', () => { pollAlerts(); refreshHealth(); });
  document.getElementById('healthRefresh').addEventListener('click', refreshHealth);
  document.getElementById('askBtn').addEventListener('click', askAssistant);
  document.getElementById('integrityBtn').addEventListener('click', verifyIntegrity);
  document.getElementById('saveNoteBtn').addEventListener('click', () => updateIncidentState());
  document.getElementById('resolveBtn').addEventListener('click', () => updateIncidentState(true));
  document.getElementById('modalClose').addEventListener('click', closeIncident);
  document.getElementById('incidentModal').addEventListener('click', event => { if (event.target.id === 'incidentModal') closeIncident(); });
  document.addEventListener('keydown', event => {
    const modal = document.getElementById('incidentModal');
    if (!modal.classList.contains('open')) return;
    if (event.key === 'Escape') {
      closeIncident();
      return;
    }
    if (event.key !== 'Tab') return;
    const focusable = [...modal.querySelectorAll('button:not(:disabled), textarea:not(:disabled)')]
      .filter(element => !element.hidden && element.getClientRects().length > 0);
    if (!focusable.length) return;
    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    const focusIsOutsideDialog = !modal.contains(document.activeElement);
    if (event.shiftKey && (document.activeElement === first || focusIsOutsideDialog)) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && (document.activeElement === last || focusIsOutsideDialog)) {
      event.preventDefault();
      first.focus();
    }
  });
  pollAlerts();
  refreshHealth();
  setInterval(pollAlerts, 5000);
</script>
</body>
</html>
"""


@app.route("/")
@app.route("/incidents")
def home():
    return render_template_string(HTML)


@app.route("/snapshots/<path:filename>")
def get_snapshot(filename):
    return send_from_directory(SNAPSHOT_DIR, filename)


@app.route("/api/alerts")
def get_alerts():
    try:
        return jsonify({"alerts": filtered_alerts()})
    except ValueError as error:
        return jsonify({"error": str(error)}), 400


@app.route("/api/alerts.csv")
def export_alerts():
    try:
        records = filtered_alerts()
    except ValueError as error:
        return jsonify({"error": str(error)}), 400
    output = io.StringIO()
    fields = (
        "timestamp", "source", "media_time_seconds", "type", "class",
        "direction", "confidence", "night", "track_id", "summary", "hash",
        "observation_count", "evidence_timestamps", "evidence_snapshots",
    )
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for record in records:
        safe_record = dict(record)
        evidence = record.get("evidence", [])
        safe_record["evidence_timestamps"] = " | ".join(
            str(item.get("timestamp") or "") for item in evidence
        )
        safe_record["evidence_snapshots"] = " | ".join(
            str(item.get("snapshot") or "") for item in evidence
        )
        for field in fields:
            value = safe_record.get(field)
            if isinstance(value, str) and value.startswith(("=", "+", "-", "@", "\t", "\r")):
                safe_record[field] = "'" + value
        writer.writerow(safe_record)
    response = make_response(output.getvalue())
    response.headers["Content-Type"] = "text/csv; charset=utf-8"
    response.headers["Content-Disposition"] = "attachment; filename=ibvap-incidents.csv"
    return response


@app.route("/api/alerts/<path:incident_id>/state", methods=["POST"])
def update_alert_state(incident_id):
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400
    note = payload.get("note", "")
    resolved = payload.get("resolved", False)
    requested_event_ids = payload.get("event_ids", [incident_id])
    if not isinstance(note, str) or len(note) > 1000:
        return jsonify({"error": "Operator notes must be text up to 1,000 characters."}), 400
    if not isinstance(resolved, bool):
        return jsonify({"error": "Resolved state must be true or false."}), 400
    if (
        not isinstance(requested_event_ids, list)
        or not requested_event_ids
        or len(requested_event_ids) > MAX_ALERTS
        or any(not isinstance(key, str) or not key for key in requested_event_ids)
        or incident_id not in requested_event_ids
    ):
        return jsonify({"error": "Incident observations are missing or invalid."}), 400
    with alerts_lock:
        known_keys = {event_key(alert) for alert in ALERTS}
        if any(key not in known_keys for key in requested_event_ids):
            return jsonify({"error": "Incident not found in the current event history."}), 404
        updated_at = datetime.now().isoformat()
        for key in requested_event_ids:
            INCIDENT_STATE[key] = {"note": note, "resolved": resolved, "updated_at": updated_at}
        try:
            save_incident_state()
        except OSError:
            logger.exception("Could not save operator review state.")
            return jsonify({"error": "Could not persist the operator review to disk."}), 500
    return jsonify({"status": "saved", "operator_state": INCIDENT_STATE[incident_id]})


@app.route("/api/system-status")
def system_status():
    ollama_status = {"available": False, "model_available": False}
    try:
        response = ollama.list()
        models = response.get("models", []) if isinstance(response, dict) else getattr(response, "models", [])
        model_names = [
            str(model.get("name") or model.get("model") or "") if isinstance(model, dict)
            else str(getattr(model, "model", "") or getattr(model, "name", ""))
            for model in models
        ]
        ollama_status = {
            "available": True,
            "model_available": CHAT_MODEL in model_names or f"{CHAT_MODEL}:latest" in model_names,
        }
    except Exception as error:
        ollama_status["error"] = str(error)
    event_log_ok = os.path.isfile(EVENTS_FILE) and os.access(EVENTS_FILE, os.R_OK)
    snapshot_ok = os.path.isdir(SNAPSHOT_DIR) and os.access(SNAPSHOT_DIR, os.R_OK)
    event_count = 0
    if event_log_ok:
        try:
            with open(EVENTS_FILE, "r", encoding="utf-8") as file:
                event_count = sum(1 for line in file if line.strip())
        except OSError as error:
            event_log_ok = False
            logger.warning("Could not read event log for health status: %s", error)
    return jsonify({
        "model": CHAT_MODEL,
        "ollama": ollama_status,
        "event_log": {"readable": event_log_ok, "count": event_count},
        "snapshot_dir": {"readable": snapshot_ok},
    })


@app.route("/api/integrity")
def integrity_status():
    try:
        return jsonify(verify_event_log(EVENTS_FILE, SNAPSHOT_DIR))
    except OSError as error:
        logger.exception("Could not verify the event log.")
        return jsonify({"error": f"Could not read integrity data: {error}"}), 500


@app.route("/ask-system", methods=["POST"])
def ask_system():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("prompt"), str):
        return jsonify({"error": "Provide a text prompt in the JSON request body."}), 400
    question = payload["prompt"].strip()
    if not question:
        return jsonify({"error": "Question cannot be empty."}), 400
    if len(question) > 1000:
        return jsonify({"error": "Question must be 1,000 characters or fewer."}), 400
    with alerts_lock:
        recent = list(ALERTS[:20])
    records = [{
        "timestamp": alert.get("timestamp"),
        "type": alert.get("type"),
        "class": alert.get("class"),
        "direction": alert.get("direction"),
        "confidence": alert.get("confidence"),
        "summary": alert.get("summary"),
    } for alert in recent]
    prompt = f"""You are an assistant for the IBVAP local video-analytics dashboard.
Answer using only the JSON event records below. Do not infer identity, intent, or wrongdoing.
If these records do not contain the answer, say that the available alert history does not establish it.
Records: {json.dumps(records, ensure_ascii=False)}
Operator question: {question}"""
    try:
        result = ollama.chat(model=CHAT_MODEL, messages=[{"role": "user", "content": prompt}])
        response = result["message"]["content"].strip()
        if not response:
            raise RuntimeError("The local model returned an empty answer.")
        return jsonify({"response": response})
    except Exception as error:
        logger.exception("Local AI assistant request failed.")
        return jsonify({"error": f"Local AI request failed: {error}"}), 503


def start_background_services():
    global background_started
    if background_started:
        return
    load_existing_alerts()
    load_incident_state()
    threading.Thread(target=watch_events, daemon=True, name="ibvap-event-watcher").start()
    background_started = True
