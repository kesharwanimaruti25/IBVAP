import json
import os
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import dashboard


class DashboardApiTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        root = self.temporary_directory.name
        dashboard.ALERTS_FILE = os.path.join(root, "alerts.jsonl")
        dashboard.STATE_FILE = os.path.join(root, "incident_state.json")
        dashboard.EVENTS_FILE = os.path.join(root, "events.jsonl")
        dashboard.SNAPSHOT_DIR = os.path.join(root, "snapshots")
        os.makedirs(dashboard.SNAPSHOT_DIR)
        dashboard.ALERTS[:] = [
            {
                "timestamp": "2026-10-10T10:00:00",
                "track_id": 12,
                "class": "Person",
                "direction": "entering",
                "confidence": 0.91,
                "night": False,
                "type": "crossing",
                "summary": "Person crossed the boundary.",
                "hash": "a" * 64,
            },
            {
                "timestamp": "2026-10-09T10:00:00",
                "track_id": 18,
                "class": "Person",
                "direction": "running",
                "confidence": 0.77,
                "night": True,
                "type": "suspicious_running",
                "summary": "=HYPERLINK(\"https://example.invalid\")",
                "hash": "b" * 64,
            },
        ]
        dashboard.INCIDENT_STATE.clear()
        dashboard.PROCESSED_EVENT_KEYS.clear()
        dashboard.PROCESSED_EVENT_KEYS.update(dashboard.event_key(alert) for alert in dashboard.ALERTS)
        self.client = dashboard.app.test_client()

    def tearDown(self):
        dashboard.ALERTS.clear()
        dashboard.INCIDENT_STATE.clear()
        dashboard.PROCESSED_EVENT_KEYS.clear()
        self.temporary_directory.cleanup()

    def test_alert_filters_apply_type_and_date(self):
        response = self.client.get("/api/alerts?type=crossing&from=2026-10-10")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(len(payload["alerts"]), 1)
        self.assertEqual(payload["alerts"][0]["track_id"], 12)

    def test_nearby_related_observations_form_one_highest_confidence_incident(self):
        events = [
            {
                "timestamp": f"2026-10-10T10:00:0{second}",
                "track_id": track_id,
                "class": "Person",
                "direction": "loitering",
                "confidence": confidence,
                "night": False,
                "type": "loitering",
                "source": "Live Camera",
                "snapshot": f"loitering_{track_id}.jpg",
                "snapshot_sha256": f"{track_id:064d}",
                "summary": "Person remained in the monitored area.",
                "hash": f"{track_id:064d}",
            }
            for second, track_id, confidence in ((1, 1, 0.42), (5, 2, 0.69), (9, 3, 0.91))
        ]
        dashboard.ALERTS.extend(events)

        response = self.client.get("/api/alerts?type=loitering")

        self.assertEqual(response.status_code, 200)
        incidents = response.get_json()["alerts"]
        self.assertEqual(len(incidents), 1)
        incident = incidents[0]
        self.assertEqual(incident["track_id"], 3)
        self.assertEqual(incident["confidence"], 0.91)
        self.assertEqual(incident["observation_count"], 3)
        self.assertEqual(len(incident["evidence"]), 3)
        self.assertEqual(incident["event_ids"], [events[0]["hash"], events[1]["hash"], events[2]["hash"]])

    def test_low_confidence_only_group_is_not_put_in_review_queue(self):
        dashboard.ALERTS.append({
            "timestamp": "2026-10-10T10:00:01",
            "track_id": 31,
            "class": "Person",
            "direction": "loitering",
            "confidence": 0.49,
            "type": "loitering",
            "source": "Live Camera",
            "hash": "9" * 64,
        })

        response = self.client.get("/api/alerts?type=loitering")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["alerts"], [])

    def test_reviewing_group_updates_all_observation_states(self):
        events = [
            {
                "timestamp": f"2026-10-10T10:00:0{second}",
                "track_id": track_id,
                "class": "Person",
                "direction": "loitering",
                "confidence": 0.7 + second / 100,
                "type": "loitering",
                "source": "Live Camera",
                "hash": f"{track_id:064d}",
            }
            for second, track_id in ((1, 1), (5, 2))
        ]
        dashboard.ALERTS.extend(events)
        incident_id = events[0]["hash"]

        response = self.client.post(
            f"/api/alerts/{incident_id}/state",
            json={"note": "Reviewed as one burst.", "resolved": True, "event_ids": [event["hash"] for event in events]},
        )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(all(dashboard.INCIDENT_STATE[event["hash"]]["resolved"] for event in events))
        incidents = self.client.get("/api/alerts?type=loitering&resolution=resolved").get_json()["alerts"]
        self.assertEqual(len(incidents), 1)
        self.assertEqual(incidents[0]["observation_count"], 2)

    def test_invalid_date_is_rejected(self):
        response = self.client.get("/api/alerts?from=not-a-date")

        self.assertEqual(response.status_code, 400)
        self.assertIn("YYYY-MM-DD", response.get_json()["error"])

    def test_csv_export_neutralizes_formula_cells(self):
        response = self.client.get("/api/alerts.csv?type=suspicious_running")

        self.assertEqual(response.status_code, 200)
        self.assertIn("text/csv", response.content_type)
        self.assertIn("'=HYPERLINK", response.get_data(as_text=True))

    def test_operator_review_state_persists_outside_event_record(self):
        response = self.client.post(
            f"/api/alerts/{'a' * 64}/state",
            json={"note": "Reviewed against snapshot.", "resolved": True},
        )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.get_json()["operator_state"]["resolved"])
        self.assertTrue(os.path.exists(dashboard.STATE_FILE))
        self.assertEqual(dashboard.ALERTS[0]["summary"], "Person crossed the boundary.")

    def test_operator_review_state_validates_input(self):
        response = self.client.post(
            f"/api/alerts/{'a' * 64}/state",
            json={"note": "x" * 1001, "resolved": True},
        )

        self.assertEqual(response.status_code, 400)

    def test_detector_events_appear_in_dashboard_feed_without_waiting_for_local_ai(self):
        event = {
            "timestamp": "2026-10-10T10:01:00",
            "track_id": 24,
            "class": "Person",
            "direction": "entering",
            "confidence": 0.88,
            "night": False,
            "type": "crossing",
            "snapshot": "crossing_24.jpg",
            "snapshot_sha256": "c" * 64,
            "source": "webcam",
            "media_time_seconds": 12.5,
            "prev_hash": "d" * 64,
            "hash": "e" * 64,
        }

        with patch("dashboard.ollama.chat", side_effect=AssertionError("Live feed must not wait for AI")):
            dashboard.process_detector_event(event)
            dashboard.process_detector_event(event)

        response = self.client.get("/api/alerts?resolution=all")

        self.assertEqual(response.status_code, 200)
        alerts = response.get_json()["alerts"]
        self.assertEqual(len(alerts), 3)
        self.assertEqual(alerts[0]["hash"], event["hash"])
        self.assertEqual(alerts[0]["source"], "webcam")
        self.assertEqual(alerts[0]["snapshot_sha256"], event["snapshot_sha256"])
        self.assertEqual(alerts[0]["summary_source"], "detector-record")
        with open(dashboard.ALERTS_FILE, "r", encoding="utf-8") as file:
            persisted = [json.loads(line) for line in file if line.strip()]
        self.assertEqual(len(persisted), 1)

    def test_event_watcher_streams_new_detector_log_entries_to_dashboard(self):
        with open(dashboard.EVENTS_FILE, "w", encoding="utf-8"):
            pass
        stop_event = threading.Event()
        watcher = threading.Thread(target=dashboard.watch_events, args=(stop_event,), daemon=True)
        watcher.start()
        event = {
            "timestamp": "2026-10-10T10:02:00",
            "track_id": 25,
            "class": "Car",
            "direction": "entering",
            "confidence": 0.82,
            "night": False,
            "type": "crossing",
            "source": "live-camera",
            "hash": "f" * 64,
        }
        try:
            with open(dashboard.EVENTS_FILE, "a", encoding="utf-8") as event_file:
                event_file.write(json.dumps(event) + "\n")
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                if any(alert.get("hash") == event["hash"] for alert in dashboard.ALERTS):
                    break
                time.sleep(0.02)
            response = self.client.get("/api/alerts?resolution=all")
            alerts = response.get_json()["alerts"]
            self.assertTrue(any(alert.get("hash") == event["hash"] for alert in alerts))
        finally:
            stop_event.set()
            watcher.join(timeout=1)
        self.assertFalse(watcher.is_alive())

    def test_assistant_returns_explicit_service_failure(self):
        with patch("dashboard.ollama.chat", side_effect=OSError("Ollama is offline")):
            response = self.client.post("/ask-system", json={"prompt": "How many events?"})

        self.assertEqual(response.status_code, 503)
        self.assertIn("Local AI request failed", response.get_json()["error"])

    def test_dashboard_is_the_default_page_and_keeps_incidents_alias(self):
        response = self.client.get("/")
        page = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn("Incident timeline", page)
        self.assertNotIn("Video studio", page)
        self.assertIn("JUDGE DEMO FLOW", page)
        self.assertIn("AUTO-REFRESH · 5S", page)
        self.assertIn("real event records only", page)
        self.assertIn("prefers-contrast: more", page)
        self.assertIn("focusIsOutsideDialog", page)
        self.assertIn("python detector.py", page)
        self.assertIn("event-thumb", page)
        self.assertIn("model score", page)
        self.assertIn("Incident timeline", self.client.get("/incidents").get_data(as_text=True))

    def test_video_studio_endpoints_are_removed(self):
        self.assertEqual(self.client.get("/api/studio/media").status_code, 404)
        self.assertEqual(self.client.post("/api/studio/analyze").status_code, 404)


if __name__ == "__main__":
    unittest.main()
