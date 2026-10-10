import hashlib
import json
import os
import tempfile
import unittest

from event_integrity import GENESIS_HASH, compute_hash, verify_event_log


class VerifyEventLogTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = self.temporary_directory.name
        self.events_file = os.path.join(self.root, "events.jsonl")
        self.snapshot_dir = os.path.join(self.root, "snapshots")
        os.makedirs(self.snapshot_dir)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def write_events(self, events):
        with open(self.events_file, "w", encoding="utf-8") as file:
            for event in events:
                file.write(json.dumps(event) + "\n")

    def make_event(self, prev_hash=GENESIS_HASH, snapshot=None):
        event = {
            "timestamp": "2026-01-01T12:00:00",
            "track_id": 1,
            "type": "crossing",
            "prev_hash": prev_hash,
        }
        if snapshot:
            event["snapshot"] = snapshot
            event["snapshot_sha256"] = hashlib.sha256(b"evidence").hexdigest()
        event["hash"] = compute_hash(prev_hash, event)
        return event

    def test_valid_chain_and_snapshot_verify(self):
        with open(os.path.join(self.snapshot_dir, "evidence.jpg"), "wb") as file:
            file.write(b"evidence")
        self.write_events([self.make_event(snapshot="evidence.jpg")])

        result = verify_event_log(self.events_file, self.snapshot_dir)

        self.assertTrue(result["valid"])
        self.assertEqual(result["snapshots_verified"], 1)
        self.assertEqual(result["issues"], [])

    def test_modified_event_is_detected(self):
        event = self.make_event()
        event["type"] = "loitering"
        self.write_events([event])

        result = verify_event_log(self.events_file, self.snapshot_dir)

        self.assertFalse(result["chain_valid"])
        self.assertTrue(any("content hash" in issue["reason"] for issue in result["issues"]))

    def test_modified_snapshot_is_detected(self):
        with open(os.path.join(self.snapshot_dir, "evidence.jpg"), "wb") as file:
            file.write(b"modified")
        self.write_events([self.make_event(snapshot="evidence.jpg")])

        result = verify_event_log(self.events_file, self.snapshot_dir)

        self.assertFalse(result["valid"])
        self.assertTrue(any("Snapshot content hash" in issue["reason"] for issue in result["issues"]))

    def test_legacy_event_without_snapshot_digest_is_reported_unverifiable(self):
        event = self.make_event()
        event["snapshot"] = "legacy.jpg"
        event.pop("hash")
        event["hash"] = compute_hash(GENESIS_HASH, event)
        self.write_events([event])

        result = verify_event_log(self.events_file, self.snapshot_dir)

        self.assertFalse(result["valid"])
        self.assertTrue(result["chain_valid"])
        self.assertEqual(result["snapshots_unverifiable"], 1)
        self.assertTrue(any("no recorded digest" in issue["reason"] for issue in result["issues"]))

    def test_broken_chain_link_is_detected(self):
        first = self.make_event()
        second = self.make_event(prev_hash="f" * 64)
        self.write_events([first, second])

        result = verify_event_log(self.events_file, self.snapshot_dir)

        self.assertFalse(result["chain_valid"])
        self.assertTrue(any("Broken hash-chain link" in issue["reason"] for issue in result["issues"]))

    def test_malformed_line_is_reported(self):
        with open(self.events_file, "w", encoding="utf-8") as file:
            file.write("{not json}\n")

        result = verify_event_log(self.events_file, self.snapshot_dir)

        self.assertFalse(result["valid"])
        self.assertEqual(result["events_checked"], 0)
        self.assertIn("Malformed event record", result["issues"][0]["reason"])


if __name__ == "__main__":
    unittest.main()
