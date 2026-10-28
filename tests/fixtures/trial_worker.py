"""Subprocess worker for the subprocess sweep tests."""

from __future__ import annotations

import json
import os
import sys


def main() -> int:
    if os.environ.get("DLEF_FAIL") == "1":
        print("intentional failure", flush=True)
        return 3

    gpu_ids = [
        int(item)
        for item in os.environ.get("DLEF_GPU_IDS", "").split(",")
        if item
    ]
    payload = {
        "trial_id": os.environ["DLEF_TRIAL_ID"],
        "status": "completed",
        "run_id": f"run-{os.environ.get('DLEF_TRIAL_INDEX', '0')}",
        "metrics": {
            "gpu": gpu_ids,
            "index": int(os.environ.get("DLEF_TRIAL_INDEX", "0")),
        },
        "error": None,
        "started_at": "",
        "ended_at": "",
        "gpu_ids": gpu_ids,
        "log_path": None,
    }
    with open(os.environ["DLEF_TRIAL_RESULT"], "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    print(f"worker done {payload['trial_id']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
