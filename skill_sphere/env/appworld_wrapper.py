"""AppWorld environment wrapper for the Skill Sphere framework.

Communicates with AppWorld via a subprocess running in a separate conda
environment (appworld_env) to avoid pydantic v1/v2 conflicts.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AppWorldStepResult:
    """Result of a single AppWorld step."""

    observation: str
    task_completed: bool
    num_interactions: int
    instruction: str = ""
    supervisor: dict = field(default_factory=dict)
    allowed_apps: list[str] = field(default_factory=list)


@dataclass
class AppWorldEvalResult:
    """Result of AppWorld ground-truth evaluation."""

    success: bool
    pass_count: int
    fail_count: int
    pass_percentage: float
    passes: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)


class AppWorldEnv:
    """Wrapper for AppWorld that runs the environment in a subprocess.

    The subprocess uses the appworld_env conda environment which has
    pydantic v1 (required by AppWorld's sqlmodel dependency).
    """

    CONDA_ENV = "appworld_env"

    def __init__(
        self,
        experiment_name: str = "sphere_eval",
        max_interactions: int = 50,
        conda_env: str | None = None,
    ):
        self.experiment_name = experiment_name
        self.max_interactions = max_interactions
        self.conda_env = conda_env or self.CONDA_ENV
        self._python_path = self._find_conda_python()
        self._proc: subprocess.Popen | None = None
        self._instruction = ""
        self._supervisor: dict = {}
        self._allowed_apps: list[str] = []
        self._task_id = ""
        self._num_interactions = 0

    def _find_conda_python(self) -> str:
        """Find the Python binary in the conda env."""
        result = subprocess.run(
            ["conda", "run", "-n", self.conda_env, "which", "python"],
            capture_output=True, text=True,
        )
        python_path = result.stdout.strip()
        if not python_path or not Path(python_path).exists():
            # Fallback: construct path directly
            conda_prefix = subprocess.run(
                ["conda", "info", "--base"], capture_output=True, text=True,
            ).stdout.strip()
            python_path = str(Path(conda_prefix) / "envs" / self.conda_env / "bin" / "python")
        return python_path

    def _ensure_proc(self):
        """Start the subprocess if not running."""
        if self._proc is not None and self._proc.poll() is None:
            return

        server_script = str(Path(__file__).parent / "appworld_server.py")
        self._proc = subprocess.Popen(
            [self._python_path, "-u", server_script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        # Verify the subprocess is alive
        resp = self._send({"cmd": "ping"})
        if not resp.get("ok"):
            raise RuntimeError(f"AppWorld subprocess failed to start: {resp}")

    def _send(self, msg: dict, timeout: float = 300) -> dict:
        """Send a command and read the response."""
        if self._proc is None or self._proc.poll() is not None:
            raise RuntimeError("AppWorld subprocess is not running")

        line = json.dumps(msg) + "\n"
        try:
            self._proc.stdin.write(line)
            self._proc.stdin.flush()
        except BrokenPipeError:
            stderr = self._proc.stderr.read() if self._proc.stderr else ""
            raise RuntimeError(f"AppWorld subprocess died. stderr: {stderr}")

        # Read response line
        resp_line = self._proc.stdout.readline()
        if not resp_line:
            stderr = self._proc.stderr.read() if self._proc.stderr else ""
            raise RuntimeError(f"AppWorld subprocess returned empty. stderr: {stderr}")

        return json.loads(resp_line.strip())

    def get_task_ids(self, split: str = "test_normal") -> list[str]:
        """Get task IDs for a given split."""
        self._ensure_proc()
        resp = self._send({"cmd": "task_ids", "split": split})
        if not resp["ok"]:
            raise RuntimeError(f"Failed to get task IDs: {resp.get('error')}")
        return resp["task_ids"]

    def reset(self, task_id: str) -> AppWorldStepResult:
        """Initialize a new task episode.

        Args:
            task_id: AppWorld task identifier (e.g., "amazon_1").

        Returns:
            Initial step result with task instruction and metadata.
        """
        self._ensure_proc()
        resp = self._send({
            "cmd": "reset",
            "task_id": task_id,
            "experiment_name": self.experiment_name,
            "max_interactions": self.max_interactions,
        })
        if not resp["ok"]:
            raise RuntimeError(f"Reset failed for {task_id}: {resp.get('error')}\n{resp.get('traceback', '')}")

        self._instruction = resp["instruction"]
        self._supervisor = resp["supervisor"]
        self._allowed_apps = resp["allowed_apps"]
        self._task_id = task_id
        self._num_interactions = 0

        return AppWorldStepResult(
            observation="",
            task_completed=False,
            num_interactions=0,
            instruction=self._instruction,
            supervisor=self._supervisor,
            allowed_apps=self._allowed_apps,
        )

    def step(self, code: str) -> AppWorldStepResult:
        """Execute Python code in the AppWorld sandbox.

        Args:
            code: Python code string to execute.

        Returns:
            Step result with execution output and task status.
        """
        resp = self._send({"cmd": "step", "code": code})
        if not resp["ok"]:
            raise RuntimeError(f"Step failed: {resp.get('error')}\n{resp.get('traceback', '')}")

        self._num_interactions = resp["num_interactions"]

        return AppWorldStepResult(
            observation=resp["output"],
            task_completed=resp["task_completed"],
            num_interactions=resp["num_interactions"],
            instruction=self._instruction,
            supervisor=self._supervisor,
            allowed_apps=self._allowed_apps,
        )

    def evaluate(self) -> AppWorldEvalResult:
        """Run AppWorld's ground-truth evaluation."""
        resp = self._send({"cmd": "evaluate"})
        if not resp["ok"]:
            raise RuntimeError(f"Evaluate failed: {resp.get('error')}\n{resp.get('traceback', '')}")

        return AppWorldEvalResult(
            success=resp["success"],
            pass_count=resp["pass_count"],
            fail_count=resp["fail_count"],
            pass_percentage=resp["pass_percentage"],
            passes=resp.get("passes", []),
            failures=resp.get("failures", []),
        )

    @property
    def task_id(self) -> str:
        return self._task_id

    @property
    def instruction(self) -> str:
        return self._instruction

    @property
    def supervisor(self) -> dict:
        return self._supervisor

    @property
    def allowed_apps(self) -> list[str]:
        return self._allowed_apps

    @property
    def num_interactions(self) -> int:
        return self._num_interactions

    def close(self):
        """Shut down the subprocess."""
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._send({"cmd": "close"})
            except Exception:
                pass
            try:
                self._proc.terminate()
                self._proc.wait(timeout=5)
            except Exception:
                self._proc.kill()
        self._proc = None

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
