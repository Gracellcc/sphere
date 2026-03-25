"""
AppWorld Environment Manager for veRL with ATS+Sphere integration.

Follows the EnvironmentManagerBase interface from veRL-agent (GiGPO):
  __init__(envs, projection_f, config)
  reset(kwargs) -> observations_dict, infos
  step(text_actions) -> observations_dict, rewards, dones, infos
  success_evaluator(**kwargs) -> {"success_rate": np.ndarray}

Key design decisions (aligned with SelfSkill/SkillRL):
- Uses SkillRL's full AppWorld prompt template (example + 16 instructions)
- Includes conversation history in each observation text (veRL's preprocess_single_sample
  only creates one user message from obs['text'], so history must be in the text itself)
- Per-step Sphere skill retrieval + injection
- Anchor observations for GiGPO step-level grouping
"""

import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

# Full Sphere pipeline imports (lazy, guarded by try/except in _init_sphere)
_SPHERE_MODULES_AVAILABLE = False
try:
    from skill_sphere.injection.dynamic_inject import DynamicInjector, InjectionResult
    from skill_sphere.injection.intent_tracker import IntentTracker
    from skill_sphere.injection.confidence import SphereConfidence
    from skill_sphere.geometry.sphere import slerp
    _SPHERE_MODULES_AVAILABLE = True
except ImportError:
    pass


def to_numpy(data):
    """Convert data to numpy array (from veRL base.py)."""
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    elif isinstance(data, np.ndarray):
        pass
    elif isinstance(data, (int, float, bool, tuple, list)):
        data = np.array(data)
    else:
        raise ValueError(f"Unsupported type: {type(data)}")
    return data


# ── Projection function ──────────────────────────────────────────────────

def appworld_ats_projection(text_actions: List[str]) -> Tuple[List[str], List[bool]]:
    """Extract Python code from model outputs.

    Supports both <code>...</code> and ```python...``` formats.
    Returns (actions, valids) matching veRL's projection_f interface.
    """
    actions = []
    valids = []
    for text in text_actions:
        code = _extract_code(text)
        actions.append(code)
        valids.append(len(code.strip()) > 0)
    return actions, valids


def _extract_code(text: str) -> str:
    """Extract Python code from model output."""
    # Try <code> ... </code> blocks (SkillRL format)
    pattern = r"<code>\s*(.*?)</code>"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[-1].strip()

    # Try ```python ... ``` blocks
    pattern = r"```python\s*(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[-1].strip()

    # Try ``` ... ``` blocks
    pattern = r"```\s*(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[-1].strip()

    # Fallback: treat entire text as code
    return text.strip()


# ── Build functions (called by make_envs in env_manager.py) ──────────────

def build_appworld_ats_envs(
    dataset_name: str,
    seed: int,
    env_num: int,
    group_n: int,
    env_config: dict = None,
):
    """Build vectorized AppWorld environments for ATS training."""
    env_config = env_config or {}
    max_steps = env_config.get("max_steps", 30)
    experiment_name = env_config.get("experiment_name", "ats_grpo")

    return AppWorldATSVecEnv(
        n_envs=env_num * group_n,
        dataset_name=dataset_name,
        max_steps=max_steps,
        experiment_name=experiment_name,
        seed=seed,
    )


class AppWorldATSVecEnv:
    """Vectorized AppWorld environment (thin wrapper over AppWorldEnv instances)."""

    def __init__(
        self,
        n_envs: int,
        dataset_name: str,
        max_steps: int = 30,
        experiment_name: str = "ats_grpo",
        seed: int = 0,
    ):
        self.n_envs = n_envs
        self.dataset_name = dataset_name
        self.max_steps = max_steps
        self.experiment_name = experiment_name
        self.seed = seed

        # Lazy-created per reset
        self.envs: List[Optional[Any]] = [None] * n_envs
        self.task_ids: List[str] = [""] * n_envs
        self.task_descriptions: List[str] = [""] * n_envs
        self.supervisors: List[Dict] = [{}] * n_envs
        self.step_counts: List[int] = [0] * n_envs
        self.dones: List[bool] = [False] * n_envs
        self.histories: List[List[Dict]] = [[] for _ in range(n_envs)]
        self.episode_logs: List[List[Dict]] = [[] for _ in range(n_envs)]

    @staticmethod
    def _normalize_kwargs(kwargs, n_envs) -> list:
        """Normalize kwargs from veRL (np.array of dicts) to list of dicts.

        veRL passes: envs.reset(kwargs=gen_batch.non_tensor_batch.pop('env_kwargs', None))
        which is an np.array of per-sample dicts like [{"task_id": "xxx"}, ...].
        """
        if kwargs is None:
            return [None] * n_envs
        if isinstance(kwargs, np.ndarray):
            kwargs = kwargs.tolist()
        if isinstance(kwargs, (list, tuple)):
            return list(kwargs)
        # Single dict fallback
        return [kwargs] * n_envs

    def reset(self, kwargs=None):
        """Reset all environments."""
        from skill_sphere.env.appworld_wrapper import AppWorldEnv

        payloads = self._normalize_kwargs(kwargs, self.n_envs)
        task_ids = []
        for i in range(self.n_envs):
            payload = payloads[i] if i < len(payloads) else None
            if isinstance(payload, dict):
                task_ids.append(payload.get("task_id", ""))
            else:
                task_ids.append("")

        text_obs = []
        infos = []

        # Close old envs and reset state
        for i in range(self.n_envs):
            task_id = task_ids[i] if i < len(task_ids) else ""
            self.task_ids[i] = task_id
            self.step_counts[i] = 0
            self.dones[i] = False
            self.histories[i] = []
            self.episode_logs[i] = []
            if self.envs[i] is not None:
                try:
                    self.envs[i].close()
                except Exception:
                    pass
                self.envs[i] = None

        text_obs = [""] * self.n_envs
        infos = [None] * self.n_envs

        def _reset_single(i, task_id):
            if not task_id:
                return (i, None, "", {"task_id": "", "won": 0.0, "is_action_valid": np.array(True)}, True, {})
            try:
                env = AppWorldEnv(
                    experiment_name=f"{self.experiment_name}_{i}",
                    max_interactions=self.max_steps,
                )
                init_result = env.reset(task_id)
                sup = init_result.supervisor
                if isinstance(sup, dict):
                    supervisor = {
                        "first_name": sup.get("first_name", ""),
                        "last_name": sup.get("last_name", ""),
                        "email": sup.get("email", ""),
                        "phone_number": sup.get("phone_number", ""),
                    }
                else:
                    supervisor = {
                        "first_name": getattr(sup, "first_name", ""),
                        "last_name": getattr(sup, "last_name", ""),
                        "email": getattr(sup, "email", ""),
                        "phone_number": getattr(sup, "phone_number", ""),
                    }
                info = {
                    "task_id": task_id,
                    "instruction": init_result.instruction,
                    "supervisor": supervisor,
                    "allowed_apps": init_result.allowed_apps,
                    "won": 0.0,
                    "is_action_valid": np.array(True),
                }
                return (i, env, init_result.instruction, info, False, supervisor)
            except Exception as e:
                print(f"[ATS VecEnv] Reset failed for {task_id}: {e}")
                return (i, None, "", {"task_id": task_id, "won": 0.0, "is_action_valid": np.array(True)}, True, {})

        # Parallel reset
        active = [(i, self.task_ids[i]) for i in range(self.n_envs)]
        with ThreadPoolExecutor(max_workers=max(1, len(active))) as pool:
            futures = [pool.submit(_reset_single, i, tid) for i, tid in active]
            for future in as_completed(futures):
                result = future.result()
                i = result[0]
                if result[4]:  # failed or empty
                    self.envs[i] = None
                    self.supervisors[i] = {}
                    self.dones[i] = True
                    text_obs[i] = ""
                    infos[i] = result[3]
                else:
                    self.envs[i] = result[1]
                    self.task_descriptions[i] = result[2]
                    self.supervisors[i] = result[5]
                    text_obs[i] = result[2]
                    infos[i] = result[3]

        return text_obs, infos

    def _step_single(self, i: int, code: str):
        """Step a single environment. Returns (i, obs, reward, done, info)."""
        if self.dones[i] or self.envs[i] is None:
            return (i, "", 0.0, True, {"is_action_valid": np.array(True), "won": 0.0})

        self.step_counts[i] += 1
        try:
            step_result = self.envs[i].step(code)
            observation = step_result.observation
            task_completed = step_result.task_completed

            self.histories[i].append({
                "step": self.step_counts[i],
                "action": code,
                "observation": observation[:2000],
            })
            self.episode_logs[i].append({
                "step": self.step_counts[i],
                "action": code,
                "observation": observation[:500],
            })

            obs = observation[:2000]

            if task_completed or self.step_counts[i] >= self.max_steps:
                self.dones[i] = True
                try:
                    eval_result = self.envs[i].evaluate()
                    won = 1.0 if eval_result.success else 0.0
                    tgc = eval_result.pass_percentage / 100.0
                except Exception:
                    won = 0.0
                    tgc = 0.0
                info = {
                    "is_action_valid": np.array(True),
                    "won": won,
                    "success": won > 0,
                    "tgc": tgc,
                    "n_steps": self.step_counts[i],
                    "task_id": self.task_ids[i],
                    "trajectory_text": self._get_trajectory_text(i),
                }
                return (i, obs, won, True, info)
            else:
                return (i, obs, 0.0, False, {"is_action_valid": np.array(True), "won": 0.0})

        except Exception as e:
            print(f"[ATS VecEnv] Step error for {self.task_ids[i]}: {e}")
            self.dones[i] = True
            return (i, f"Error: {str(e)[:200]}", 0.0, True, {
                "is_action_valid": np.array(False),
                "won": 0.0,
                "success": False,
                "tgc": 0.0,
                "n_steps": self.step_counts[i],
                "task_id": self.task_ids[i],
                "trajectory_text": self._get_trajectory_text(i),
            })

    def step(self, actions: List[str]):
        """Step all environments in parallel."""
        text_obs = [""] * self.n_envs
        rewards = np.zeros(self.n_envs, dtype=np.float32)
        dones = np.zeros(self.n_envs, dtype=bool)
        infos = [None] * self.n_envs

        # Collect active envs for parallel execution
        active = [(i, actions[i]) for i in range(self.n_envs)
                  if not self.dones[i] and self.envs[i] is not None]
        # Fill in already-done envs
        for i in range(self.n_envs):
            if self.dones[i] or self.envs[i] is None:
                dones[i] = True
                infos[i] = {"is_action_valid": np.array(True), "won": 0.0}

        if active:
            with ThreadPoolExecutor(max_workers=len(active)) as pool:
                futures = {pool.submit(self._step_single, i, code): i for i, code in active}
                for future in as_completed(futures):
                    idx, obs, reward, done, info = future.result()
                    text_obs[idx] = obs
                    rewards[idx] = reward
                    dones[idx] = done
                    infos[idx] = info

        return text_obs, rewards, dones, infos

    def _get_trajectory_text(self, env_idx: int) -> str:
        parts = [f"Task: {self.task_descriptions[env_idx]}"]
        for log in self.episode_logs[env_idx]:
            parts.append(f"\n[Step {log['step']}]")
            parts.append(f"Action: {log['action'][:300]}")
            parts.append(f"Observation: {log['observation'][:300]}")
        return "\n".join(parts)

    def close(self):
        for env in self.envs:
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass


# ── Prompt Templates ─────────────────────────────────────────────────────
# Adapted from SkillRL's APPWORLD_TEMPLATE_NO_HIS and APPWORLD_TEMPLATE.
# These are the full templates with example and 16 key instructions,
# proven to work well with 8B models on AppWorld.
# Double braces {{ }} are used for literal braces in .format() strings.

APPWORLD_ATS_TEMPLATE_NO_HIS = """I am your supervisor and you are a super intelligent AI Assistant whose job is to achieve my day-to-day tasks completely autonomously.

To do this, you will need to interact with app/s (e.g., spotify, venmo, etc) using their associated APIs on my behalf. For this you will undertake a *multi-step conversation* using a python REPL environment. That is, you will write the python code and the environment will execute it and show you the result, based on which, you will write python code for the next step and so on, until you've achieved the goal. This environment will let you interact with app/s using their associated APIs on my behalf.

Here are three key APIs that you need to know to get more information

# To get a list of apps that are available to you.
print(apis.api_docs.show_app_descriptions())

# To get the list of apis under any app listed above, e.g. supervisor
print(apis.api_docs.show_api_descriptions(app_name='supervisor'))

# To get the specification of a particular api, e.g. supervisor app's show_account_passwords
print(apis.api_docs.show_api_doc(app_name='supervisor', api_name='show_account_passwords'))

Each code execution will produce an output that you can use in subsequent calls. Using these APIs, you can now generate code, that the environment will execute, to solve the task.

-----------------------------
Here is an example:

My name is: supervisor_first_name supervisor_last_name. My personal email is supervisor_email and phone number is supervisor_phone_number.

Your task is: What is the password for my Spotify account?

Code 1:
print(apis.api_docs.show_app_descriptions())

Result 1:
[
  {{
    "name": "api_docs",
    "description": "An app to search and explore API documentation."
  }},
  {{
    "name": "supervisor",
    "description": "An app to access supervisor's personal information, account credentials, addresses, payment cards, and manage the assigned task."
  }},
  ...
  {{
    "name": "spotify",
    "description": "A music streaming app to stream songs and manage song, album and playlist libraries."
  }},
  {{
    "name": "venmo",
    "description": "A social payment app to send, receive and request money to and from others."
  }},
  ...
]

Code 2:
print(apis.api_docs.show_api_descriptions(app_name='supervisor'))

Result 2:
[
  ...
  "show_account_passwords : Show your supervisor's account passwords."
  ...
]

Code 3:
print(apis.api_docs.show_api_doc(app_name='supervisor', api_name='show_account_passwords'))

Result 3:
{{
  'app_name': 'supervisor',
  'api_name': 'show_account_passwords',
  'path': '/account_passwords',
  'method': 'GET',
  'description': "Show your supervisor's app account passwords.",
  'parameters': [],
  'response_schemas': {{
    'success': [{{'account_name': 'string', 'password': 'string'}}],
    'failure': {{'message': 'string'}}
  }}
}}

Code 4:
print(apis.supervisor.show_account_passwords())

Result 4:
[
  {{
    "account_name": "spotify",
    "password": "dummy_spotify_pass"
  }},
  {{
    "account_name": "file_system",
    "password": "dummy_fs_pass"
  }},
  ...
]

Code 5:
# So the Spotify password is an entry in the `passwords` list with the account_name=spotify.
spotify_password = [account_password["account_name"] == "spotify" for account_password in passwords][0]["password"]
print(spotify_password)

Result 5:
dummy_spotify_pass

Code 6:
# When the task is completed, I need to call apis.supervisor.complete_task(). If there is an answer, I need to pass it as an argument `answer`. I will pass the spotify_password as an answer.
apis.supervisor.complete_task(answer=spotify_password)

Result 6:
Marked the active task complete.
-----------------------------

Key Instructions and Disclaimers:
1. The email addresses, access tokens and variables (e.g. spotify_password) in the example above were only for demonstration. Obtain the correct information by calling relevant APIs yourself.
2. Only generate valid code blocks, i.e., do not put them in ```...``` or add any extra formatting. Any thoughts should be put as code comments.
3. You can use the variables from the previous code blocks in the subsequent code blocks.
4. Write small chunks of code and only one chunk of code in every step. Make sure everything is working correctly before making any irreversible change.
5. The provided Python environment has access to its standard library. But modules and functions that have a risk of affecting the underlying OS, file system or process are disabled. You will get an error if do call them.
6. Any reference to a file system in the task instructions means the file system *app*, operable via given APIs, and not the actual file system the code is running on. So do not write code making calls to os-level modules and functions.
7. To interact with apps, only use the provided APIs, and not the corresponding Python packages. E.g., do NOT use `spotipy` for Spotify. Remember, the environment only has the standard library.
8. The provided API documentation has both the input arguments and the output JSON schemas. All calls to APIs and parsing its outputs must be as per this documentation.
9. For APIs that return results in "pages", make sure to consider all pages. Use a loop:
page = 1
all_items = []
while True:
    result = apis.APP.some_list_api(access_token=token, page=page)
    all_items.extend(result)
    if len(result) == 0:
        break
    page += 1
10. To obtain current date or time, use Python functions like `datetime.now()` or obtain it from the phone app. Do not rely on your existing knowledge of what the current date or time is.
11. For all temporal requests, use proper time boundaries, e.g., if I ask for something that happened yesterday, make sure to consider the time between 00:00:00 and 23:59:59. All requests are concerning a single, default (no) time zone.
12. Any reference to my friends, family or any other person or relation refers to the people in my phone's contacts list.
13. All my personal information, and information about my app account credentials, physical addresses and owned payment cards are stored in the "supervisor" app. You can access them via the APIs provided by the supervisor app.
14. The answers, when given, should be just entity or number, not full sentences, e.g., `answer=10` for "How many songs are in the Spotify queue?". When an answer is a number, it should be in numbers, not in words, e.g., "10" and not "ten".
15. You can also pass `status="fail"` in the complete_task API if you are sure you cannot solve it and want to exit.
16. Once you believe the task is complete, you MUST call `apis.supervisor.complete_task()` to finalize it. If the task requires an answer, provide it using the answer argument — for example, `apis.supervisor.complete_task(answer=<answer>)`. For tasks that do not require an answer, either omit the argument. The task will not end automatically — it will remain open until you explicitly make this call.
{skill_section}
Using these APIs, now begin writing code cells step-by-step to solve the actual task:

My name is: {supervisor_first_name} {supervisor_last_name}. My personal email is {supervisor_email} and phone number is {supervisor_phone_number}.

Your task is: {task_description}

Now it's your turn to generate code to solve the task.
You should first reason step-by-step about which APIs to call, what arguments to use, and how to build your code block to complete the task. Put your reasoning as code comments.
Present the solution code body within <code> </code> tags."""


APPWORLD_ATS_TEMPLATE_WITH_HIS = """I am your supervisor and you are a super intelligent AI Assistant whose job is to achieve my day-to-day tasks completely autonomously.

To do this, you will need to interact with app/s (e.g., spotify, venmo, etc) using their associated APIs on my behalf. For this you will undertake a *multi-step conversation* using a python REPL environment. That is, you will write the python code and the environment will execute it and show you the result, based on which, you will write python code for the next step and so on, until you've achieved the goal. This environment will let you interact with app/s using their associated APIs on my behalf.

Here are three key APIs that you need to know to get more information

# To get a list of apps that are available to you.
print(apis.api_docs.show_app_descriptions())

# To get the list of apis under any app listed above, e.g. supervisor
print(apis.api_docs.show_api_descriptions(app_name='supervisor'))

# To get the specification of a particular api, e.g. supervisor app's show_account_passwords
print(apis.api_docs.show_api_doc(app_name='supervisor', api_name='show_account_passwords'))

Each code execution will produce an output that you can use in subsequent calls. Using these APIs, you can now generate code, that the environment will execute, to solve the task.

-----------------------------
Here is an example:

My name is: supervisor_first_name supervisor_last_name. My personal email is supervisor_email and phone number is supervisor_phone_number.

Your task is: What is the password for my Spotify account?

Code 1:
print(apis.api_docs.show_app_descriptions())

Result 1:
[
  {{
    "name": "api_docs",
    "description": "An app to search and explore API documentation."
  }},
  {{
    "name": "supervisor",
    "description": "An app to access supervisor's personal information, account credentials, addresses, payment cards, and manage the assigned task."
  }},
  ...
  {{
    "name": "spotify",
    "description": "A music streaming app to stream songs and manage song, album and playlist libraries."
  }},
  {{
    "name": "venmo",
    "description": "A social payment app to send, receive and request money to and from others."
  }},
  ...
]

Code 2:
print(apis.api_docs.show_api_descriptions(app_name='supervisor'))

Result 2:
[
  ...
  "show_account_passwords : Show your supervisor's account passwords."
  ...
]

Code 3:
print(apis.api_docs.show_api_doc(app_name='supervisor', api_name='show_account_passwords'))

Result 3:
{{
  'app_name': 'supervisor',
  'api_name': 'show_account_passwords',
  'path': '/account_passwords',
  'method': 'GET',
  'description': "Show your supervisor's app account passwords.",
  'parameters': [],
  'response_schemas': {{
    'success': [{{'account_name': 'string', 'password': 'string'}}],
    'failure': {{'message': 'string'}}
  }}
}}

Code 4:
print(apis.supervisor.show_account_passwords())

Result 4:
[
  {{
    "account_name": "spotify",
    "password": "dummy_spotify_pass"
  }},
  {{
    "account_name": "file_system",
    "password": "dummy_fs_pass"
  }},
  ...
]

Code 5:
# So the Spotify password is an entry in the `passwords` list with the account_name=spotify.
spotify_password = [account_password["account_name"] == "spotify" for account_password in passwords][0]["password"]
print(spotify_password)

Result 5:
dummy_spotify_pass

Code 6:
# When the task is completed, I need to call apis.supervisor.complete_task(). If there is an answer, I need to pass it as an argument `answer`. I will pass the spotify_password as an answer.
apis.supervisor.complete_task(answer=spotify_password)

Result 6:
Marked the active task complete.
-----------------------------

Key Instructions and Disclaimers:
1. The email addresses, access tokens and variables (e.g. spotify_password) in the example above were only for demonstration. Obtain the correct information by calling relevant APIs yourself.
2. Only generate valid code blocks, i.e., do not put them in ```...``` or add any extra formatting. Any thoughts should be put as code comments.
3. You can use the variables from the previous code blocks in the subsequent code blocks.
4. Write small chunks of code and only one chunk of code in every step. Make sure everything is working correctly before making any irreversible change.
5. The provided Python environment has access to its standard library. But modules and functions that have a risk of affecting the underlying OS, file system or process are disabled. You will get an error if do call them.
6. Any reference to a file system in the task instructions means the file system *app*, operable via given APIs, and not the actual file system the code is running on. So do not write code making calls to os-level modules and functions.
7. To interact with apps, only use the provided APIs, and not the corresponding Python packages. E.g., do NOT use `spotipy` for Spotify. Remember, the environment only has the standard library.
8. The provided API documentation has both the input arguments and the output JSON schemas. All calls to APIs and parsing its outputs must be as per this documentation.
9. For APIs that return results in "pages", make sure to consider all pages. Use a loop:
page = 1
all_items = []
while True:
    result = apis.APP.some_list_api(access_token=token, page=page)
    all_items.extend(result)
    if len(result) == 0:
        break
    page += 1
10. To obtain current date or time, use Python functions like `datetime.now()` or obtain it from the phone app. Do not rely on your existing knowledge of what the current date or time is.
11. For all temporal requests, use proper time boundaries, e.g., if I ask for something that happened yesterday, make sure to consider the time between 00:00:00 and 23:59:59. All requests are concerning a single, default (no) time zone.
12. Any reference to my friends, family or any other person or relation refers to the people in my phone's contacts list.
13. All my personal information, and information about my app account credentials, physical addresses and owned payment cards are stored in the "supervisor" app. You can access them via the APIs provided by the supervisor app.
14. The answers, when given, should be just entity or number, not full sentences, e.g., `answer=10` for "How many songs are in the Spotify queue?". When an answer is a number, it should be in numbers, not in words, e.g., "10" and not "ten".
15. You can also pass `status="fail"` in the complete_task API if you are sure you cannot solve it and want to exit.
16. Once you believe the task is complete, you MUST call `apis.supervisor.complete_task()` to finalize it. If the task requires an answer, provide it using the answer argument — for example, `apis.supervisor.complete_task(answer=<answer>)`. For tasks that do not require an answer, either omit the argument. The task will not end automatically — it will remain open until you explicitly make this call.
{skill_section}
Using these APIs, now begin writing code cells step-by-step to solve the actual task:

My name is: {supervisor_first_name} {supervisor_last_name}. My personal email is {supervisor_email} and phone number is {supervisor_phone_number}.

Your task is: {task_description}

Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} codes you generated and the corresponding environment return:
{action_history}

Now you are at step {current_step} and it's your turn to generate code for this step.
First, carefully reflect on the history of interactions and the most recent error messages. Then, reason about what should be done next, which APIs to call, what arguments to use, and how to build your code block to complete the task. Put your reasoning as code comments.
Present the solution code body within <code> </code> tags."""


# ── Environment Manager (veRL EnvironmentManagerBase compatible) ─────────

class AppWorldATSEnvironmentManager:
    """AppWorld environment manager with ATS+Sphere integration.

    Follows veRL's EnvironmentManagerBase interface.
    Mirrors SkillRL's AppWorldEnvironmentManager pattern:
      - Full prompt template with example + 16 instructions
      - Conversation history management via self.memory
      - Per-step observation formatting
    """

    # Max characters for action history to prevent context overflow
    MAX_HISTORY_CHARS = 10000
    # How many recent steps to include in the prompt
    DEFAULT_HISTORY_LENGTH = 10

    def __init__(self, envs, projection_f, config):
        self.envs = envs
        self.projection_f = projection_f
        self.config = config

        # Per-env state (like SkillRL's SimpleMemory)
        self.tasks: List[str] = []
        self.supervisors: List[Dict] = []
        self.memory: List[List[Dict]] = []  # [env_idx][step] = {action, text_obs}

        # History length from config (default 10, like SkillRL)
        try:
            self._history_length = int(config.env.get("history_length", self.DEFAULT_HISTORY_LENGTH))
        except Exception:
            self._history_length = self.DEFAULT_HISTORY_LENGTH

        # Sphere integration (lazy init)
        self._sphere = None
        self._sphere_encoder = None
        self._skills_path = None
        self._injector = None  # DynamicInjector (full pipeline)

        # Per-env sphere state (initialized in reset())
        self._intent_vectors = []      # per-env intent point on sphere
        self._intent_trackers = []     # per-env IntentTracker
        self._sgc_modules = []         # per-env SphereConfidence
        self._sgc_scores = []          # per-env last SGC score (for reward manager)
        self._recently_used = []       # per-env recently used skill indices
        self._prev_codes = []          # per-env previous codes (loop detection)
        self._category_masks = []      # per-env: valid skill indices after category filtering
        self._skills_used = []         # per-env: unique skill indices injected this episode

        # Try to load sphere from config
        skills_path = None
        try:
            skills_path = config.get("reward_model", {}).get("reward_kwargs", {}).get("skills_path")
        except Exception:
            pass
        if not skills_path:
            skills_path = os.environ.get("ATS_SKILLS_PATH")
        if skills_path and os.path.exists(str(skills_path)):
            self._skills_path = str(skills_path)
            self._init_sphere()

    def _init_sphere(self):
        """Initialize SkillSphere + full injection pipeline (DynamicInjector + IntentTracker + SGC)."""
        if not self._skills_path:
            return
        try:
            from skill_sphere.skill_bank.encoder import SkillEncoder
            from skill_sphere.skill_bank.skill_sphere import SkillSphere

            self._sphere_encoder = SkillEncoder(device="cpu")
            self._sphere = SkillSphere.from_skillrl_json(
                self._skills_path, encoder=self._sphere_encoder, device="cpu"
            )
            print(f"[ATS EnvMgr] Sphere loaded: {len(self._sphere.skills)} skills")

            # Initialize full injection pipeline (mirrors appworld_agent.py)
            if _SPHERE_MODULES_AVAILABLE:
                self._injector = DynamicInjector(
                    sigma=1.5,
                    min_weight=0.05,
                    min_inject_strength=0.15,
                    max_skills=5,
                    relevance_k=10,
                    redundancy_threshold=0.85,
                )
                self._injector.calibrate(self._sphere.vectors)
                print(f"[ATS EnvMgr] Full Sphere pipeline initialized (DynamicInjector + IntentTracker + SGC)")
            else:
                print(f"[ATS EnvMgr] WARNING: Sphere modules not available, falling back to cosine top-k")
        except Exception as e:
            print(f"[ATS EnvMgr] Failed to load sphere: {e}")
            self._sphere = None

    def _build_category_mask(self, allowed_apps: List[str]) -> List[int]:
        """Build mask of valid skill indices based on allowed apps for this task.

        Design doc: "Category过滤：当前任务涉及的app → 去掉不相关category的skill"
        - general / common_mistakes / multi_app always pass (universal skills)
        - App-specific categories (spotify, venmo, file_system) pass only if in allowed_apps

        Note: Sphere's from_skillrl_json strips 'appworld/' prefix, so categories
        are 'general', 'spotify', 'venmo', etc. (not 'appworld/spotify').
        """
        if self._sphere is None or not allowed_apps:
            return list(range(len(self._sphere.skills))) if self._sphere else []

        # Always-pass categories
        universal = {"general", "common_mistakes", "multi_app"}
        allowed_set = {a.lower() for a in allowed_apps}

        valid_indices = []
        for i, skill in enumerate(self._sphere.skills):
            cat = skill.category.lower()
            # Strip 'appworld/' prefix if present (handles both formats)
            if "/" in cat:
                cat = cat.split("/", 1)[1]
            if cat in universal or cat in allowed_set:
                valid_indices.append(i)

        return valid_indices

    def _retrieve_skills_simple(self, query: str, max_skills: int = 5) -> str:
        """Fallback: simple cosine top-k retrieval (no Sphere modules)."""
        if self._sphere is None or self._sphere_encoder is None:
            return ""

        query_vec = self._sphere_encoder.encode_query(query)
        sims = torch.mv(self._sphere.vectors, query_vec)
        topk = min(max_skills, len(self._sphere.skills))
        _, top_indices = torch.topk(sims, topk)

        lines = []
        for idx in top_indices.tolist():
            skill = self._sphere.skills[idx]
            lines.append(f"- {skill.name}: {skill.principle}")
        return "\n".join(lines)

    def _sphere_inject(self, env_idx: int, query: str, step_num: int, action: str = "") -> str:
        """Full Sphere injection pipeline for one environment.

        Design doc 4-step retrieval:
        1. Category过滤: filter skills by task's allowed_apps
        2. Encode query → intent point update via slerp
        3. SGC confidence + loop detection
        4. DynamicInjector.decide() → complementary retrieval + injection weights
        5. Format skills with Fréchet mean re-ranking
        """
        if self._sphere is None or self._sphere_encoder is None:
            return ""

        # --- Full pipeline (DynamicInjector + IntentTracker + SGC) ---
        if self._injector is not None and env_idx < len(self._intent_vectors):
            t = self._intent_vectors[env_idx]
            tracker = self._intent_trackers[env_idx]
            sgc = self._sgc_modules[env_idx]
            prev_codes = self._prev_codes[env_idx]
            recently_used = self._recently_used[env_idx]

            # Step 1: Category filtering — only pass skills relevant to this task's apps
            cat_mask = self._category_masks[env_idx] if env_idx < len(self._category_masks) else None
            if cat_mask and len(cat_mask) < len(self._sphere.skills):
                mask_tensor = torch.tensor(cat_mask, dtype=torch.long)
                filtered_vectors = self._sphere.vectors[mask_tensor]
                filtered_skills = [self._sphere.skills[i] for i in cat_mask]
                # Map recently_used indices to filtered space
                mask_set = {orig: filt for filt, orig in enumerate(cat_mask)}
                filtered_recently = [mask_set[r] for r in recently_used if r in mask_set]
            else:
                filtered_vectors = self._sphere.vectors
                filtered_skills = self._sphere.skills
                filtered_recently = recently_used
                cat_mask = list(range(len(self._sphere.skills)))

            # Step 2: Encode context and update intent via slerp
            context_vec = self._sphere_encoder.encode_query(query)
            drift_info = tracker.update(context_vec)
            t = slerp(
                t.unsqueeze(0), context_vec.unsqueeze(0), drift_info.alpha
            ).squeeze(0)
            self._intent_vectors[env_idx] = t

            # Loop detection
            in_loop = (
                len(prev_codes) >= 2
                and prev_codes[-1] == prev_codes[-2]
                and len(prev_codes[-1]) > 0
            )
            force = in_loop or step_num <= 1

            # Step 3: SGC confidence
            isolation = self._injector.compute_isolation(t, filtered_vectors)
            sgc_signals = sgc.compute(
                coherence=drift_info.coherence,
                stability=drift_info.stability,
                isolation_score=isolation,
            )
            effective_confidence = 0.1 if in_loop else sgc_signals.sgc
            self._sgc_scores[env_idx] = sgc_signals.sgc

            # Skill rotation when stuck or low confidence
            rotation_active = in_loop or effective_confidence < 0.4
            rotation_list = filtered_recently if rotation_active else None

            # Step 4: DynamicInjector decision (on category-filtered skills)
            result = self._injector.decide(
                intent_point=t,
                confidence=effective_confidence,
                skill_vectors=filtered_vectors,
                force_inject=force,
                drift_rate=drift_info.drift_rate,
                recently_used=rotation_list,
            )

            # Map selected indices back to original space and track
            if result.selected_indices:
                original_indices = [cat_mask[i] for i in result.selected_indices]
                recently_used.extend(original_indices)
                self._recently_used[env_idx] = recently_used[-15:]
                # Track unique skills used this episode (for verifier)
                self._skills_used[env_idx].update(original_indices)

            # Track previous code for loop detection
            if action:
                prev_codes.append(action)
                self._prev_codes[env_idx] = prev_codes[-5:]

            if not result.should_inject:
                return ""

            # Step 5: Format with Fréchet mean re-ranking
            skill_text = self._injector.format_injected_skills(
                result, filtered_skills, skill_vectors=filtered_vectors,
            )

            if in_loop and skill_text:
                skill_text += "\n\nIMPORTANT: You seem to be repeating the same code. Try a DIFFERENT approach."

            return skill_text

        # --- Fallback: simple cosine top-k ---
        return self._retrieve_skills_simple(query)

    def _format_skill_section(self, query: str, env_idx: int = 0, step_num: int = 0, action: str = "") -> str:
        """Format skill section for template injection."""
        if self._injector is not None:
            skill_text = self._sphere_inject(env_idx, query, step_num, action)
        else:
            skill_text = self._retrieve_skills_simple(query)
        if not skill_text:
            return ""
        return f"\n\nRelevant Skill Guidance:\n{skill_text}\n"

    def reset(self, kwargs) -> Tuple[Dict[str, Any], List[Dict]]:
        """Reset all environments and return initial observations."""
        n_envs = len(kwargs) if kwargs is not None else "?"
        print(f"[ATS Env] reset: {n_envs} envs", flush=True)
        text_obs, infos = self.envs.reset(kwargs=kwargs)

        # Initialize per-env state
        batch_size = len(text_obs)
        self.tasks = list(text_obs)
        self.supervisors = [info.get("supervisor", {}) for info in infos]
        self.memory = [[] for _ in range(batch_size)]

        # Initialize per-env Sphere state (intent vectors, trackers, SGC)
        self._intent_vectors = []
        self._intent_trackers = []
        self._sgc_modules = []
        self._sgc_scores = [0.0] * batch_size
        self._recently_used = [[] for _ in range(batch_size)]
        self._prev_codes = [[] for _ in range(batch_size)]
        self._skills_used = [set() for _ in range(batch_size)]

        # Build per-env category masks from allowed_apps
        self._category_masks = []
        if self._sphere is not None:
            for info in infos:
                allowed_apps = info.get("allowed_apps", [])
                mask = self._build_category_mask(allowed_apps)
                self._category_masks.append(mask)
        else:
            self._category_masks = [[] for _ in range(batch_size)]

        if self._sphere is not None and self._sphere_encoder is not None and self._injector is not None:
            for i in range(batch_size):
                # Encode task description as initial intent point
                t = self._sphere_encoder.encode_query(text_obs[i]) if text_obs[i] else torch.zeros(self._sphere.vectors.shape[1])
                self._intent_vectors.append(t)
                # Fresh IntentTracker per env, calibrated from sphere
                tracker = IntentTracker()
                tracker.calibrate(self._injector.retriever.d_typical)
                tracker.reset(t)
                self._intent_trackers.append(tracker)
                # Fresh SGC per env
                sgc = SphereConfidence()
                sgc.reset()
                self._sgc_modules.append(sgc)

        # Build full text observations
        full_text_obs = self._build_text_obs(text_obs, infos, init=True)

        return {
            "text": full_text_obs,
            "image": None,
            "anchor": text_obs,
        }, infos

    def step(self, text_actions: List[str]):
        """Execute text actions and return next state."""
        if not hasattr(self, '_step_counter'):
            self._step_counter = 0
        self._step_counter += 1
        n_active = sum(1 for a in text_actions if a and a.strip())
        if self._step_counter % 5 == 1 or n_active <= 2:
            print(f"[ATS Env] step {self._step_counter}: {n_active} active envs", flush=True)
        actions, valids = self.projection_f(text_actions)
        text_obs, rewards, dones, infos = self.envs.step(actions)

        # Store step in memory (like SkillRL's self.memory.store)
        # text_obs[i] is the RESULT of executing actions[i]
        for i in range(len(text_obs)):
            self.memory[i].append({
                "text_obs": text_obs[i],
                "action": actions[i],
            })

        for i, info in enumerate(infos):
            info["is_action_valid"] = to_numpy(valids[i])
            # Sphere SGC (agent's position stability on skill sphere)
            # Always set from Sphere — VecEnv uses separate 'tgc' key for AppWorld eval
            if self._injector is not None and i < len(self._sgc_scores):
                info["sgc"] = self._sgc_scores[i]
            # Propagate skills_used for verifier (only score injected skills)
            if (i < len(self._skills_used) and self._skills_used[i]
                    and "skills_used" not in info and self._sphere is not None):
                info["skills_used"] = [
                    self._sphere.skills[idx].name
                    for idx in self._skills_used[i]
                    if idx < len(self._sphere.skills)
                ]

        # Build full text observations with history
        full_text_obs = self._build_text_obs(text_obs, infos, init=False)

        next_observations = {
            "text": full_text_obs,
            "image": None,
            "anchor": text_obs,
        }

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def _build_text_obs(
        self, text_obs: List[str], infos: List[Dict], init: bool = False
    ) -> List[str]:
        """Build full text observations with template + optional history.

        init=True: Uses APPWORLD_ATS_TEMPLATE_NO_HIS (first step, no history)
        init=False: Uses APPWORLD_ATS_TEMPLATE_WITH_HIS (includes action history)
        """
        prompts = []
        for i in range(len(text_obs)):
            if not text_obs[i] and init:
                prompts.append("")
                continue
            if not init and not text_obs[i]:
                prompts.append("")
                continue

            supervisor = self.supervisors[i] if i < len(self.supervisors) else {}
            task_desc = self.tasks[i] if i < len(self.tasks) else text_obs[i]

            # Sphere skill injection (full pipeline: IntentTracker + SGC + DynamicInjector)
            if init:
                skill_section = self._format_skill_section(task_desc, env_idx=i, step_num=0)
            else:
                # Per-step: build context from task + recent action + current obs
                step_num = self.envs.step_counts[i]
                recent = self.memory[i][-1] if self.memory[i] else {}
                recent_action = recent.get('action', '')
                context = f"{task_desc}. Step {step_num}: {recent_action[:100]}. Result: {text_obs[i][:200]}"
                skill_section = self._format_skill_section(context, env_idx=i, step_num=step_num, action=recent_action)

            if init:
                prompt = APPWORLD_ATS_TEMPLATE_NO_HIS.format(
                    supervisor_first_name=supervisor.get("first_name", ""),
                    supervisor_last_name=supervisor.get("last_name", ""),
                    supervisor_email=supervisor.get("email", ""),
                    supervisor_phone_number=supervisor.get("phone_number", ""),
                    task_description=task_desc,
                    skill_section=skill_section,
                )
            else:
                # Build action history from memory (like SkillRL)
                history = self.memory[i][-self._history_length:]
                history_parts = []
                for j, step_data in enumerate(history):
                    step_num = len(self.memory[i]) - len(history) + j + 1
                    history_parts.append(
                        f"Code {step_num}:\n{step_data['action']}\n\n"
                        f"Result {step_num}:\n{step_data['text_obs']}"
                    )
                action_history = "\n\n".join(history_parts)

                # Truncate history if too long
                if len(action_history) > self.MAX_HISTORY_CHARS:
                    action_history = action_history[-self.MAX_HISTORY_CHARS:]

                # current_step = next step the model will generate
                current_step = len(self.memory[i]) + 1

                prompt = APPWORLD_ATS_TEMPLATE_WITH_HIS.format(
                    supervisor_first_name=supervisor.get("first_name", ""),
                    supervisor_last_name=supervisor.get("last_name", ""),
                    supervisor_email=supervisor.get("email", ""),
                    supervisor_phone_number=supervisor.get("phone_number", ""),
                    task_description=task_desc,
                    step_count=len(self.memory[i]),
                    history_length=min(len(self.memory[i]), self._history_length),
                    action_history=action_history,
                    current_step=current_step,
                    skill_section=skill_section,
                )

            prompts.append(prompt)

        return prompts

    def close(self):
        """Close all environments."""
        self.envs.close()

    def success_evaluator(self, *args, **kwargs) -> Dict[str, np.ndarray]:
        """Evaluate episode success for veRL logging."""
        total_infos = kwargs["total_infos"]
        total_batch_list = kwargs["total_batch_list"]
        batch_size = len(total_batch_list)

        success = defaultdict(list)

        for bs in range(batch_size):
            self._process_batch(bs, total_batch_list, total_infos, success)

        assert len(success["success_rate"]) == batch_size
        return {key: np.array(value) for key, value in success.items()}

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        """Process one batch item for success evaluation."""
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item["active_masks"]:
                info = total_infos[batch_idx][i]
                won_value = float(info["won"])
                success["success_rate"].append(won_value)
                return
        # No active step found (e.g., env reset failed) — count as failure
        success["success_rate"].append(0.0)
