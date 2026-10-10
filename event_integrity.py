import hashlib
import json
import os


GENESIS_HASH = "0" * 64


def compute_hash(prev_hash, event_data):
    payload = prev_hash + json.dumps(event_data, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def verify_event_log(events_file, snapshot_dir):
    if not os.path.exists(events_file):
        return {
            "valid": False,
            "chain_valid": False,
            "events_checked": 0,
            "snapshots_verified": 0,
            "snapshots_unverifiable": 0,
            "issues": [{"entry": None, "reason": "Event log file is missing."}],
        }
    with open(events_file, "r", encoding="utf-8") as file:
        lines = [line for line in file if line.strip()]

    expected_prev = GENESIS_HASH
    chain_valid = True
    snapshots_verified = 0
    snapshots_unverifiable = 0
    issues = []
    checked = 0

    for line_number, line in enumerate(lines, 1):
        try:
            event = json.loads(line)
            if not isinstance(event, dict):
                raise ValueError("Event record is not a JSON object.")
        except (json.JSONDecodeError, ValueError) as error:
            chain_valid = False
            issues.append({"entry": line_number, "reason": f"Malformed event record: {error}"})
            continue
        checked += 1
        stored_prev = event.get("prev_hash")
        stored_hash = event.get("hash")
        if stored_prev != expected_prev:
            chain_valid = False
            issues.append({"entry": line_number, "reason": "Broken hash-chain link."})
        recompute_data = {key: value for key, value in event.items() if key != "hash"}
        if not stored_hash or compute_hash(stored_prev or "", recompute_data) != stored_hash:
            chain_valid = False
            issues.append({"entry": line_number, "reason": "Event content hash does not match."})
        expected_prev = stored_hash

        snapshot = event.get("snapshot")
        snapshot_hash = event.get("snapshot_sha256")
        if not snapshot:
            snapshots_unverifiable += 1
        elif not snapshot_hash:
            snapshots_unverifiable += 1
            issues.append({"entry": line_number, "reason": "Snapshot exists in the record but has no recorded digest."})
        elif os.path.basename(snapshot) != snapshot:
            issues.append({"entry": line_number, "reason": "Snapshot path is not a filename."})
        else:
            snapshot_path = os.path.join(snapshot_dir, snapshot)
            if not os.path.isfile(snapshot_path):
                issues.append({"entry": line_number, "reason": "Snapshot file is missing."})
            else:
                with open(snapshot_path, "rb") as image:
                    actual_hash = hashlib.sha256(image.read()).hexdigest()
                if actual_hash != snapshot_hash:
                    issues.append({"entry": line_number, "reason": "Snapshot content hash does not match."})
                else:
                    snapshots_verified += 1

    return {
        "valid": chain_valid and not issues,
        "chain_valid": chain_valid,
        "events_checked": checked,
        "snapshots_verified": snapshots_verified,
        "snapshots_unverifiable": snapshots_unverifiable,
        "issues": issues,
    }
