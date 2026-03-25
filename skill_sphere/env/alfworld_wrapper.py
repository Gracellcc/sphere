"""ALFWorld environment wrapper for the Skill Sphere framework.

Wraps the ALFWorld TextWorld environment into a clean interface
for agent interaction, with task metadata extraction.
"""

from __future__ import annotations

import os
import yaml
from dataclasses import dataclass
from pathlib import Path


@dataclass
class StepResult:
    """Result of a single environment step."""

    observation: str
    reward: float
    done: bool
    admissible_commands: list[str]
    task_type: str
    info: dict


# ALFWorld task types derived from the game file path
TASK_TYPES = {
    "pick_and_place": "pick",
    "look_at_obj": "examine",
    "pick_clean_then_place": "clean",
    "pick_heat_then_place": "heat",
    "pick_cool_then_place": "cool",
    "pick_two_obj": "pick_two",
}


def _detect_task_type(game_file: str) -> str:
    """Detect ALFWorld task type from game file path."""
    for key, task_type in TASK_TYPES.items():
        if key in game_file:
            return task_type
    return "unknown"


def _load_alfworld_config(config_path: str | None = None) -> dict:
    """Load ALFWorld config from a YAML file.

    If no path is given, uses the SkillRL config bundled with this project.
    """
    if config_path is None:
        # Default: use SkillRL's config
        project_root = Path(__file__).resolve().parents[2]
        config_path = str(
            project_root
            / "external"
            / "SkillRL"
            / "agent_system"
            / "environments"
            / "env_package"
            / "alfworld"
            / "configs"
            / "config_tw.yaml"
        )

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"ALFWorld config not found: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    return config


# Map our split names to AlfredTWEnv split names
_SPLIT_MAP = {
    "eval_out_of_distribution": "eval_out_of_distribution",
    "eval_in_distribution": "eval_in_distribution",
    "valid_unseen": "eval_out_of_distribution",
    "valid_seen": "eval_in_distribution",
    "train": "train",
}


class ALFWorldEnv:
    """Wrapper for ALFWorld TextWorld environment.

    Loads the environment using SkillRL's AlfredTWEnv under the hood,
    providing a simple reset()/step() interface.
    """

    def __init__(
        self,
        split: str = "eval_in_distribution",
        max_steps: int = 50,
        config_path: str | None = None,
    ):
        """
        Args:
            split: Which split to use. Options:
                   "train", "eval_in_distribution", "eval_out_of_distribution",
                   "valid_seen", "valid_unseen"
            max_steps: Maximum steps per episode before forced termination.
            config_path: Path to alfworld config YAML. If None, uses SkillRL's.
        """
        self.split = _SPLIT_MAP.get(split, split)
        self.max_steps = max_steps
        self.config_path = config_path
        self._env = None
        self._current_step = 0
        self._task_type = "unknown"
        self._game_file = ""
        self._num_games = 0
        self._episode_count = 0

    def _ensure_env(self):
        """Lazy initialization of ALFWorld environment."""
        if self._env is not None:
            return

        config = _load_alfworld_config(self.config_path)

        # Import AlfredTWEnv from the alfworld package
        from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv

        tw_env = AlfredTWEnv(config, train_eval=self.split)
        self._num_games = tw_env.num_games
        self._env = tw_env.init_env(batch_size=1)

    @property
    def num_games(self) -> int:
        """Number of games available in the current split."""
        self._ensure_env()
        return self._num_games

    def reset(self) -> StepResult:
        """Reset to a new episode.

        Returns:
            Initial StepResult with the starting observation.

        Raises:
            StopIteration: When all episodes in the split have been exhausted.
        """
        self._ensure_env()

        obs, info = self._env.reset()
        self._current_step = 0
        self._episode_count += 1

        # Extract task metadata
        if "extra.gamefile" in info:
            self._game_file = info["extra.gamefile"][0]
            self._task_type = _detect_task_type(self._game_file)

        observation = obs[0] if isinstance(obs, (list, tuple)) else obs
        admissible = info.get("admissible_commands", [[]])[0]

        return StepResult(
            observation=observation,
            reward=0.0,
            done=False,
            admissible_commands=admissible,
            task_type=self._task_type,
            info=info,
        )

    def step(self, action: str) -> StepResult:
        """Take an action in the environment.

        Args:
            action: Text action string (e.g., "go to countertop 1").

        Returns:
            StepResult with new observation, reward, done flag, etc.
        """
        self._current_step += 1

        obs, scores, dones, infos = self._env.step([action])

        observation = obs[0] if isinstance(obs, (list, tuple)) else obs
        reward = scores[0] if isinstance(scores, (list, tuple)) else scores
        done = dones[0] if isinstance(dones, (list, tuple)) else dones
        admissible = infos.get("admissible_commands", [[]])[0]

        # Force termination if max steps reached
        if self._current_step >= self.max_steps:
            done = True

        return StepResult(
            observation=observation,
            reward=float(reward),
            done=bool(done),
            admissible_commands=admissible,
            task_type=self._task_type,
            info=infos,
        )

    @property
    def current_step(self) -> int:
        return self._current_step

    @property
    def task_type(self) -> str:
        return self._task_type

    @property
    def game_file(self) -> str:
        return self._game_file
