# IBVAP — Intelligent Video Analytics Platform

[![Tests](https://github.com/kesharwanimaruti25/IBVAP/actions/workflows/tests.yml/badge.svg)](https://github.com/kesharwanimaruti25/IBVAP/actions/workflows/tests.yml)

**An evidence-led, local-first video analytics prototype.** IBVAP processes video on-device, records detector events with a hash chain, and provides a dashboard for reviewing snapshots and asking a local Ollama model about recent alert records.

> Detector alerts are observations for human review. They are not proof of intent, identity, or wrongdoing.

## Product workflow

1. Start the detector and, optionally, the local Ollama service.
2. Use the detector's OpenCV window to select a video and draw its virtual fence.
3. Keep the incident desk open to see detector events appear in its live-updating feed, with snapshots and operator notes.
4. Ask the local assistant questions grounded in recent alert records, export filtered records, or verify the event chain.

## Run locally

The full detector stack has been smoke-tested on macOS arm64 with Python 3.9.6. The lightweight unit-test workflow also runs on Python 3.11, but that does not validate the full computer-vision stack on Python 3.11 or other operating systems. Install the dependencies in a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
ollama pull llama3.1:8b
ollama serve
```

Keep Ollama running. In a second terminal, activate the same environment and run the detector with an authorized video file, or omit the path to use the default camera:

```bash
source .venv/bin/activate
python detector.py /path/to/authorized-video.mp4
```

In another terminal, launch the dashboard:

```bash
source .venv/bin/activate
python app1.py
```

Open `http://127.0.0.1:5001` for the incident desk. To start live webcam monitoring, run `python detector.py`; the live webcam preview and detector view are mirrored, the fence setup stays on the live feed, and detection starts as soon as the fence is confirmed. For recorded clips, use the authorized video file path; the detector keeps its paused fence-setup and pre-start review screens, without mirroring, for that clip. The assistant is optional and requires the configured local Ollama model.

The YOLO model weights are not included in this repository; Ultralytics downloads `yolov8s.pt` when the detector first runs. BoT-SORT's `lap` dependency is installed directly from `requirements.txt` to avoid an implicit package install at first inference. Sample videos and captured footage are also excluded—use only video you are authorized to process and publish.

`anpr.py` is a separate, experimental webcam prototype and is not connected to the dashboard. To install its additional dependency, run `python -m pip install -r requirements-anpr.txt`.

The detector and dashboard must use the same `IBVAP_EVENTS_FILE` and `IBVAP_SNAPSHOT_DIR` values. If you override either path, export the same values in both terminal windows before starting `python app1.py` and `python detector.py`. The dashboard watches the event log and refreshes its feed every five seconds; detector events are added immediately with a deterministic summary, without waiting for Ollama. The suspicious-convergence rule requires people to approach each other over time, rather than merely stand near each other, and limits repeat alerts for the same tracks.

`IBVAP_DEVICE` selects the Ultralytics inference device. It defaults to `mps` for the tested Apple Silicon setup; use `cpu` on a CPU-only host or a supported CUDA device such as `0` on a configured NVIDIA host. These alternatives are configurable but have not been validated as complete deployment targets.

During interactive detection, click a visible person box to mark or unmark that tracker as friendly for the current detector session. Friendly boxes remain visible with a distinct label, while crossing, loitering, running, and convergence alerts involving that track are suppressed. This is manual, temporary operator control—not uniform or face recognition. Tracker IDs can switch or be reassigned in crowds, so verify the marked box and remove the friendly mark if it no longer follows the intended person. Existing recorded events are not changed.

## Dashboard capabilities

- Live event feed with keyword, event-type, date, and review-state filters
- Incident queue groups same-source/type/class/direction observations within 30 seconds, shows their snapshots together, and headlines the highest-confidence observation; groups below a 50% detector-score threshold stay out of the review queue. Grouping reduces review volume but does not identify whether observations came from the same individual.
- Detector events appear in the dashboard feed as they are appended to the shared event log
- Event detail view with snapshot, confidence, event hash, and snapshot digest
- Persistent per-event operator notes and reviewed/open state
- CSV export using the currently selected filters
- On-demand hash-chain and snapshot verification
- Evidence thumbnails directly in the incident queue, with grouped snapshots in the detail view
- Clear local service/sync status, a truthful zero-state, and a three-step judge-demo guide (no fabricated events)
- Local-model summaries and Q&A with explicit unavailable/error states
- Responsive command-center dashboard with keyboard-accessible controls and a clear review/evidence hierarchy
- Detector video-end controls and tracking labels sized to stay readable within the video frame
- Video source and media-relative event time on new detector records

Detector scores are model outputs, not calibrated probabilities. The dashboard labels them as model scores and keeps them for human review; it does not claim that a score proves intent or identity.

## Configuration

The defaults match the current workspace layout. Override paths/model without editing source:

| Environment variable | Default |
| --- | --- |
| `IBVAP_CHAT_MODEL` | `llama3.1:8b` |
| `IBVAP_DEVICE` | `mps` |
| `IBVAP_EVENTS_FILE` | `~/my-ai-app/events.jsonl` |
| `IBVAP_ALERTS_FILE` | `~/my-ai-app/alerts.jsonl` |
| `IBVAP_SNAPSHOT_DIR` | `~/my-ai-app/snapshots` |
| `IBVAP_STATE_FILE` | `~/my-ai-app/incident_state.json` |

## Integrity and event compatibility

New detector events include a SHA-256 digest of the saved snapshot and a chained event hash. The dashboard's **Verify event log** button and `python verify_chain.py` check event links, event content, and snapshots that have recorded digests. Older records without a snapshot digest remain chain-checkable but are explicitly counted as unverifiable evidence; they are not represented as snapshot-verified.

Operator review state is kept separately in `incident_state.json`, so annotations do not mutate the hash-chained detector evidence.

## Measure event-level detection performance

New detector records include the source clip name and video-relative event time. To evaluate a run, prepare an authorized, manually labeled CSV using [evaluation/labels_template.csv](./evaluation/labels_template.csv). Add one expected event per row:

```csv
video,event_type,time_seconds,direction,event_class
vid01.mp4,crossing,12.4,entering,Person
vid01.mp4,loitering,35.0,loitering,Person
```

You can create labels interactively instead of editing CSV by hand:

```bash
python annotate_events.py /path/to/authorized/vid01.mp4 \
  --output evaluation/run-01/labels.csv
```

The video opens paused. Use SPACE to play/pause, A/D to seek one second, [/] to step one frame, and 1–4 to record the displayed timestamp as crossing, loitering, suspicious-running, or suspicious-converging. Q saves and exits. The tool records only the label you choose; direction and class remain blank unless you add them after checking the footage. It refuses to overwrite an existing CSV unless `--append` is supplied. For additional clips, append to the same CSV and use `--source-name` if detector records were given a custom `--source-label`. Review every label against the original video before evaluating; this is human annotation, not automatic ground truth.

Use a fresh output directory for each measured detector run so records from earlier runs cannot inflate the false-positive count:

```bash
mkdir -p evaluation/run-01
IBVAP_EVENTS_FILE="$PWD/evaluation/run-01/events.jsonl" \
IBVAP_SNAPSHOT_DIR="$PWD/evaluation/run-01/snapshots" \
python detector.py /path/to/authorized-video.mp4
```

After reviewing the labels for that exact clip in `evaluation/run-01/labels.csv`, run:

```bash
python evaluate_events.py --labels evaluation/run-01/labels.csv --events evaluation/run-01/events.jsonl --tolerance 2
```

For a run containing only a deliberately event-free negative clip, leave the labels CSV header-only and explicitly opt in:

```bash
python evaluate_events.py \
  --labels evaluation/run-01/negative-labels.csv \
  --events evaluation/run-01/events.jsonl \
  --allow-empty-labels
```

This reports any detector records in the event log as false positives. The flag is required so an accidentally empty annotation file is not silently treated as a valid negative set.

The evaluator reports one-to-one event-level precision, recall, and F1 overall, per event type, and per video, plus false positives and misses. Optional direction and event-class labels must match; leave them blank to ignore those fields. The tolerance is the allowed difference in video-relative seconds. This is not frame-level accuracy, and it is only meaningful when the labels cover the evaluated clips. Existing historical records without video timing metadata are excluded and counted as unscorable; they are not silently matched using wall-clock timestamps.

For a defensible measurement, manually label every expected event in each selected clip, including clips with no expected events, and review the labels against the video before scoring. Keep threshold/rule tuning clips separate from a held-out set used for the result you present. Do not use detector output as ground truth. Report the number and conditions of the labeled clips together with precision, recall, false positives, and misses; a small or convenient sample is not evidence of field accuracy.

For a reviewable evaluation artifact, add `--json-report`:

```bash
python evaluate_events.py \
  --labels evaluation/run-01/labels.csv \
  --events evaluation/run-01/events.jsonl \
  --tolerance 2 \
  --json-report evaluation/run-01/report.json
```

The JSON report records the matching method and tolerance, SHA-256 digests of the input label and event files and evaluator, metrics overall/per type/per video, every one-to-one match with its time error, and the unmatched labels and detections. Keep labels authorized, document which clips and camera conditions they cover, and review every miss and false alert. A report makes a run auditable; it does not establish that the selected clips represent real deployments or record all model/hardware configuration.

Run the automated tests with `python -m unittest discover -s tests -v`. A lightweight test-only install is available with `python -m pip install -r requirements-test.txt`; GitHub Actions runs this suite on pushes and pull requests.

## Deployment validation status

The full dependency set has been installed and exercised in a fresh Python 3.9.6 environment on macOS arm64 (Apple M4). Headless inference has completed on the available local clips using MPS, and the configurable CPU device has also completed a smoke run on this same computer. The dashboard returned HTTP 200 from that environment, and a generated event and snapshot passed `verify_chain.py`. These are installation and workflow smoke tests only: the clips were not manually labeled, so they do not establish detection accuracy. One packaging caveat remains: `pip 26 check` flags the MediaPipe 0.10.14 wheel's Intel-only internal platform tag on Apple Silicon, although its native extension and the detector's face-detection initialization/inference path ran on the M4; `pip 21.2.4 check` reports no broken requirements. Keep MediaPipe pinned until a replacement retains the legacy `mp.solutions.face_detection` API and is verified on this hardware.

The automated GitHub Actions job installs only `requirements-test.txt` on Python 3.11; it does not validate the full vision stack on Python 3.11. Webcam capture, IP-camera streams, Windows, Linux, CUDA, long-duration operation, concurrent cameras, and production access controls have not been validated. The supported demonstrated scope remains a single local stream on the tested Apple Silicon setup. Before claiming support for another deployment target, run the full install, unit tests, detector smoke test, dashboard/evidence check, and integrity verifier on that target.

## Pitch/demo outline

1. Show the local-service status, the auto-refresh timestamp, and the no-cloud processing statement.
2. Run a prepared, authorized sample video and trigger a visible event.
3. Open the event evidence and point out the recorded class, timestamp, model score, and hashes.
4. Ask a question answerable from the recent alert records; show the local-model status.
5. Add an operator note, mark the event reviewed, filter to reviewed items, and export CSV.
6. Run integrity verification. For a tamper demonstration, use a disposable copy of the event log and snapshots—never alter the working evidence.

## Current limitations

- Event types are heuristic detections based on the current detector logic and must be validated for the intended environment.
- The event evaluator requires manually labeled clips; no accuracy figure is meaningful until representative labels have been created and checked.
- Confidence scores are model outputs, not calibrated probabilities of an incident.
- The local language model can produce incorrect summaries or answers; check the source record and snapshot.
- No user authentication or multi-user access control is included. Keep the dashboard bound to localhost unless access controls are added.
- The current detector is an interactive OpenCV program and needs a display for boundary setup and review.
- Boundary-crossing events require the tracked ground-point path to intersect the finite fence segment, not just cross its infinite extension. Synthetic geometry tests cover endpoints and out-of-segment movement; they do not establish real-world detector accuracy.
- Recorded-video dwell, movement, and event cooldown timing use media time, rather than inference wall time, so slow model processing does not make short clips look like longer loitering events.
- Before presenting accuracy claims, compare detections against manually labeled authorized footage from the intended camera setup, and report false positives and missed events alongside successful detections.
