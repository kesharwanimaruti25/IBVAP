import os

from event_integrity import verify_event_log


EVENTS_FILE = os.path.expanduser(os.environ.get("IBVAP_EVENTS_FILE", "~/my-ai-app/events.jsonl"))
SNAPSHOT_DIR = os.path.expanduser(os.environ.get("IBVAP_SNAPSHOT_DIR", "~/my-ai-app/snapshots"))


def verify():
    result = verify_event_log(EVENTS_FILE, SNAPSHOT_DIR)
    if result["valid"]:
        print(
            f"Chain integrity verified: {result['events_checked']} events; "
            f"{result['snapshots_verified']} snapshots verified; "
            f"{result['snapshots_unverifiable']} snapshots without a recorded digest."
        )
    else:
        print(
            f"Integrity check FAILED: {result['events_checked']} events checked; "
            f"{len(result['issues'])} issue(s)."
        )
        for issue in result["issues"]:
            entry = issue["entry"] if issue["entry"] is not None else "log"
            print(f"Entry {entry}: {issue['reason']}")
    return result

if __name__ == "__main__":
    verify()