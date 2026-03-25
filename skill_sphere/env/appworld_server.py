#!/usr/bin/env python3
"""AppWorld subprocess server.

Runs in the appworld_env conda environment (pydantic v1) and communicates
with the main process via JSON-line protocol over stdin/stdout.

Protocol:
  → {"cmd": "reset", "task_id": "amazon_1", "experiment_name": "sphere_eval", "max_interactions": 50}
  ← {"ok": true, "instruction": "...", "supervisor": {...}, "allowed_apps": [...], "output": ""}

  → {"cmd": "step", "code": "print('hello')"}
  ← {"ok": true, "output": "hello", "task_completed": false, "num_interactions": 1}

  → {"cmd": "evaluate"}
  ← {"ok": true, "success": true, "pass_count": 3, "fail_count": 0, "pass_percentage": 100.0,
      "passes": [...], "failures": [...]}

  → {"cmd": "close"}
  ← {"ok": true}

  → {"cmd": "task_ids", "split": "test_normal"}
  ← {"ok": true, "task_ids": ["amazon_1", ...]}
"""

import json
import sys
import traceback


def main():
    env = None

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            msg = json.loads(line)
        except json.JSONDecodeError as e:
            _respond({"ok": False, "error": f"JSON parse error: {e}"})
            continue

        cmd = msg.get("cmd")

        try:
            if cmd == "task_ids":
                from appworld import load_task_ids
                split = msg.get("split", "test_normal")
                ids = load_task_ids(split)
                _respond({"ok": True, "task_ids": ids})

            elif cmd == "reset":
                # Close previous env if any
                if env is not None:
                    try:
                        env.close()
                    except Exception:
                        pass

                from appworld import AppWorld
                task_id = msg["task_id"]
                experiment_name = msg.get("experiment_name", "sphere_eval")
                max_interactions = msg.get("max_interactions", 50)

                env = AppWorld(
                    task_id=task_id,
                    experiment_name=experiment_name,
                    max_interactions=max_interactions,
                    timeout_seconds=60,
                    raise_on_failure=False,
                    load_ground_truth=True,
                    ground_truth_mode="minimal",
                    show_api_response_schemas=True,
                )

                task = env.task
                supervisor = {
                    "first_name": task.supervisor.first_name,
                    "last_name": task.supervisor.last_name,
                    "email": task.supervisor.email,
                    "phone_number": getattr(task.supervisor, "phone_number", ""),
                }

                # Extract per-app account passwords
                try:
                    account_passwords = {
                        ap.account_name: ap.password
                        for ap in task.supervisor.account_passwords
                    }
                    supervisor["account_passwords"] = account_passwords
                except Exception:
                    supervisor["account_passwords"] = {}

                _respond({
                    "ok": True,
                    "instruction": task.instruction,
                    "supervisor": supervisor,
                    "allowed_apps": list(task.allowed_apps),
                    "output": "",
                    "task_id": task_id,
                })

            elif cmd == "step":
                if env is None:
                    _respond({"ok": False, "error": "No environment initialized. Call reset first."})
                    continue

                code = msg["code"]
                output = env.execute(code)
                _respond({
                    "ok": True,
                    "output": output,
                    "task_completed": env.task_completed(),
                    "num_interactions": env.num_interactions,
                })

            elif cmd == "evaluate":
                if env is None:
                    _respond({"ok": False, "error": "No environment initialized."})
                    continue

                result = env.evaluate()
                _respond({
                    "ok": True,
                    "success": result.success,
                    "pass_count": result.pass_count,
                    "fail_count": result.fail_count,
                    "pass_percentage": result.pass_percentage,
                    "passes": [str(p) for p in result.passes],
                    "failures": [str(f) for f in result.failures],
                })

            elif cmd == "close":
                if env is not None:
                    try:
                        env.close()
                    except Exception:
                        pass
                    env = None
                _respond({"ok": True})

            elif cmd == "ping":
                _respond({"ok": True, "msg": "pong"})

            else:
                _respond({"ok": False, "error": f"Unknown command: {cmd}"})

        except Exception as e:
            tb = traceback.format_exc()
            _respond({"ok": False, "error": str(e), "traceback": tb})


def _respond(data: dict):
    """Write JSON response to stdout."""
    sys.stdout.write(json.dumps(data) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
