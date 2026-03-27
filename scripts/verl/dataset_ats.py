"""
ATS Dataset for veRL GRPO training.

Loads AppWorld tasks with training skill's Data Selection policy.
Each item provides the task prompt + Sphere context for multi-turn rollout.

Training skill Data Selection examples:
  - "Early Stage Balanced": uniform sampling, exclude >80% success types
  - "Weakness Focused": 70% lowest success types, 30% random
  - "Efficiency Push": tasks with >30% success but avg steps > 20
"""

import copy
import json
import os
import random
from typing import Any, Dict, List, Optional, Union

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

try:
    from omegaconf import DictConfig, ListConfig
    import verl.utils.torch_functional as verl_F
    from verl.utils.model import compute_position_id_with_mask
except ImportError:
    DictConfig = dict
    ListConfig = list

# AppWorld task loading
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

APPWORLD_SYSTEM_PROMPT = """You are an AI Assistant whose job is to complete day-to-day tasks fully autonomously on behalf of a supervisor.

To do this, you interact with apps (e.g., spotify, venmo, gmail, etc.) using their APIs. You undertake a multi-step conversation using a Python REPL environment. You write Python code, the environment executes it and shows you the result, based on which you write the next step, and so on until the goal is achieved.

## How to discover and use APIs

There are 3 key APIs for discovering available functionality:
1. `apis.api_docs.show_app_descriptions()` — lists all available apps with descriptions
2. `apis.api_docs.show_api_descriptions(app_name="APP")` — lists all APIs for a given app
3. `apis.api_docs.show_api_doc(app_name="APP", api_name="API")` — shows full spec for an API

Always check the API spec before calling an unfamiliar API.

## How to log in to apps

Most APIs require an access_token. To get one:
1. Call `apis.supervisor.show_account_passwords()` to get all passwords
2. Call `apis.APP_NAME.login(username=EMAIL, password=PW)` to get access_token
3. For phone app, use phone_number as username instead of email

## Output format

Write Python code in ```python ... ``` blocks. One code block per step."""


class ATSDataset(Dataset):
    """Dataset that loads AppWorld tasks with training skill data selection.

    Supports two modes:
    1. 'appworld://split_name' — load task IDs from AppWorld split
    2. Path to task file — load specific task IDs from file

    Data selection is controlled by the active training skill's policy.
    """

    def __init__(
        self,
        data_files: Union[str, List[str]],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
    ):
        if not isinstance(data_files, (list, ListConfig)):
            data_files = [data_files]

        self.tokenizer = tokenizer
        self.config = config
        self.processor = processor
        self.max_prompt_length = config.get("max_prompt_length", 4096)
        self.truncation = config.get("truncation", "error")
        self.return_raw_chat = config.get("return_raw_chat", True)

        # ATS config
        ats_config = config.get("ats", {})
        self.skills_path = ats_config.get("skills_path", "data/skills/appworld_skills_ats.json")
        self.data_selection_policy = ats_config.get("data_selection", "balanced")
        self.task_stats_path = ats_config.get("task_stats_path", None)

        # Load task IDs
        self.task_ids = self._load_task_ids(data_files[0], ats_config)
        self.split_name = self._resolve_split(data_files[0])

        # Load task stats for data selection (if available)
        self.task_stats = {}
        if self.task_stats_path and os.path.exists(self.task_stats_path):
            with open(self.task_stats_path) as f:
                self.task_stats = json.load(f)

        # Apply data selection policy
        self.selected_task_ids = self._apply_data_selection(self.task_ids)

        print(f"[ATS Dataset] Split={self.split_name}, "
              f"tasks={len(self.task_ids)}, selected={len(self.selected_task_ids)}, "
              f"policy={self.data_selection_policy}")

    def _resolve_split(self, data_file: str) -> str:
        if isinstance(data_file, str) and data_file.startswith("appworld://"):
            return data_file.split("://", 1)[1]  # e.g. "train+dev+test_challenge"
        return "custom"

    def _load_task_ids(self, data_file: str, ats_config: dict) -> List[str]:
        """Load task IDs from AppWorld split(s) or file.

        Supports multi-split: 'appworld://train+dev+test_challenge'
        merges all splits into one training pool.
        """
        if isinstance(data_file, str) and data_file.startswith("appworld://"):
            spec = data_file.split("://", 1)[1]
            splits = [s.strip() for s in spec.split("+")]
            ids = []
            for split in splits:
                appworld_root = os.environ.get("APPWORLD_ROOT")
                if not appworld_root:
                    raise EnvironmentError(
                        "APPWORLD_ROOT environment variable must be set "
                        "(e.g. export APPWORLD_ROOT=/path/to/srpo)"
                    )
                task_file = f"{appworld_root}/data/datasets/{split}.txt"
                if os.path.exists(task_file):
                    with open(task_file) as f:
                        ids.extend([line.strip() for line in f if line.strip()])
                else:
                    try:
                        if not os.environ.get("APPWORLD_ROOT", ""):
                            raise EnvironmentError(
                                "APPWORLD_ROOT environment variable must be set "
                                "to load AppWorld splits"
                            )
                        from appworld import load_task_ids
                        ids.extend(list(load_task_ids(split)))
                    except ImportError:
                        print(f"[ATS Dataset] WARNING: Cannot load split '{split}', skipping")
            if not ids:
                raise FileNotFoundError(f"No tasks loaded from: {data_file}")
            # Deduplicate (in case of overlap) while preserving order
            seen = set()
            unique_ids = []
            for tid in ids:
                if tid not in seen:
                    seen.add(tid)
                    unique_ids.append(tid)
            ids = unique_ids
            limit = ats_config.get("max_samples")
            if limit:
                ids = ids[:int(limit)]
            print(f"[ATS Dataset] Loaded {len(ids)} tasks from splits: {splits}")
            return ids
        elif os.path.exists(str(data_file)):
            with open(data_file) as f:
                return [line.strip() for line in f if line.strip()]
        else:
            raise FileNotFoundError(f"Unknown data source: {data_file}")

    def _apply_data_selection(self, task_ids: List[str]) -> List[str]:
        """Apply training skill's Data Selection policy.

        Policies (per-task granularity, finer than per-type):
        - "balanced": uniform sampling, exclude tasks with >80% success rate
        - "weakness_focused": 70% lowest success tasks, 30% random from rest
        - "efficiency_push": tasks with >30% success but avg steps > 20
        """
        if not self.task_stats:
            return task_ids  # No stats yet, use all tasks

        policy = self.data_selection_policy

        if policy == "balanced":
            # Exclude tasks with >80% success rate
            selected = []
            for tid in task_ids:
                stats = self.task_stats.get(tid, {})
                success_rate = stats.get("success_rate", 0.0)
                if success_rate <= 0.8:
                    selected.append(tid)
            if not selected:
                print(f"[ATS Dataset] WARNING: balanced selection empty (all tasks >80% sr), using all tasks")
                return task_ids
            return selected

        elif policy == "weakness_focused":
            # Sort by success rate, take 70% from bottom + 30% random from rest
            scored = []
            for tid in task_ids:
                stats = self.task_stats.get(tid, {})
                sr = stats.get("success_rate", 0.0)
                scored.append((sr, tid))
            scored.sort()
            n_weak = int(len(scored) * 0.7)
            n_random = len(scored) - n_weak
            weak = [tid for _, tid in scored[:n_weak]]
            rest = [tid for _, tid in scored[n_weak:]]
            # Seeded shuffle for reproducibility across GRPO workers
            rng = random.Random(42)
            rng.shuffle(rest)
            return weak + rest[:n_random]

        elif policy == "efficiency_push":
            # Tasks that succeed but are slow (can do it, but do it slowly)
            selected = []
            for tid in task_ids:
                stats = self.task_stats.get(tid, {})
                sr = stats.get("success_rate", 0.0)
                avg_steps = stats.get("avg_steps", 0)
                if sr > 0.3 and avg_steps > 20:
                    selected.append(tid)
            if not selected:
                print(f"[ATS Dataset] WARNING: efficiency_push selection empty, using all tasks")
                return task_ids
            return selected

        return task_ids

    def __len__(self):
        return len(self.selected_task_ids)

    def __getitem__(self, item):
        task_id = self.selected_task_ids[item]

        # Build initial prompt
        # Note: The actual task description is injected by the env manager at reset time.
        # Here we provide a minimal prompt that includes the system instructions.
        # The env manager will replace this with the full prompt including task + supervisor info.
        messages = [
            {"role": "system", "content": APPWORLD_SYSTEM_PROMPT},
            {"role": "user", "content": f"Begin the task. [task_id={task_id}]"},
        ]

        raw_prompt = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
            enable_thinking=False,
        )
        model_inputs = self.tokenizer(
            raw_prompt, return_tensors="pt", add_special_tokens=False
        )
        input_ids = model_inputs.pop("input_ids")
        attention_mask = model_inputs.pop("attention_mask")

        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        position_ids = compute_position_id_with_mask(attention_mask)

        env_payload = {
            "task_id": task_id,
        }

        row_dict = {
            "input_ids": input_ids[0],
            "attention_mask": attention_mask[0],
            "position_ids": position_ids[0],
            "raw_prompt_ids": self.tokenizer.encode(raw_prompt, add_special_tokens=False),
            "raw_prompt": messages,
            "data_source": "appworld_ats",
            "index": item,             # veRL expects top-level index
            "tools_kwargs": {},         # veRL expects top-level tools_kwargs
            "env_kwargs": env_payload,
            "extra_info": {
                "task_id": task_id,
                "index": item,
                "need_tools_kwargs": False,
            },
        }

        if not self.return_raw_chat:
            row_dict.pop("raw_prompt")

        return row_dict

    def update_data_selection(self, policy: str, task_stats: Dict = None):
        """Called by outer loop to update data selection policy.

        This is triggered when training skill changes.
        """
        self.data_selection_policy = policy
        if task_stats:
            self.task_stats = task_stats
        self.selected_task_ids = self._apply_data_selection(self.task_ids)
        print(f"[ATS Dataset] Updated data selection: policy={policy}, "
              f"selected={len(self.selected_task_ids)}/{len(self.task_ids)}")
