"""
Patch veRL's hardcoded dispatch tables to register ATS reward manager and environment.

veRL (SelfSkill fork) uses hardcoded if/elif chains in:
  - verl/trainer/main_ppo.py (reward_manager_name dispatch)
  - agent_system/environments/env_manager.py (env_name dispatch)

This script patches both files to add 'ats' cases. Idempotent — safe to run multiple times.
"""

import os
import sys

SELFSKILL_ROOT = os.environ.get("SELFSKILL_ROOT", "")
if not SELFSKILL_ROOT:
    print("ERROR: SELFSKILL_ROOT environment variable must be set "
          "(e.g. export SELFSKILL_ROOT=/path/to/SelfSkill)")
    sys.exit(1)
SPHERE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ATS_MARKER = "# === ATS+Sphere patch ==="


def patch_main_ppo():
    """Add 'ats' case to reward manager dispatch in main_ppo.py."""
    path = os.path.join(SELFSKILL_ROOT, "verl/trainer/main_ppo.py")
    if not os.path.exists(path):
        print(f"[patch] WARNING: {path} not found, skipping reward manager patch")
        return False

    with open(path) as f:
        content = f.read()

    if ATS_MARKER in content:
        print("[patch] main_ppo.py already patched, skipping")
        return True

    # Find the pattern:
    #   elif reward_manager_name == 'appworld_skill':
    #       from agent_system.reward_manager import AppWorldSkillRewardManager
    #       reward_manager_cls = AppWorldSkillRewardManager
    #   else:
    #       raise NotImplementedError

    old = """        elif reward_manager_name == 'appworld_skill':
            from agent_system.reward_manager import AppWorldSkillRewardManager
            reward_manager_cls = AppWorldSkillRewardManager
        else:
            raise NotImplementedError"""

    new = """        elif reward_manager_name == 'appworld_skill':
            from agent_system.reward_manager import AppWorldSkillRewardManager
            reward_manager_cls = AppWorldSkillRewardManager
        elif reward_manager_name == 'ats':  {marker}
            from scripts.verl.reward_manager_ats import ATSRewardManager
            reward_manager_cls = ATSRewardManager
        else:
            raise NotImplementedError""".format(marker=ATS_MARKER)

    if old not in content:
        print("[patch] WARNING: Cannot find reward manager dispatch pattern in main_ppo.py")
        print("[patch] You may need to manually add the 'ats' case")
        return False

    content = content.replace(old, new)
    with open(path, "w") as f:
        f.write(content)

    print("[patch] main_ppo.py patched: added 'ats' reward manager")
    return True


def patch_env_manager():
    """Add 'appworld_ats' case to env dispatch in env_manager.py."""
    path = os.path.join(SELFSKILL_ROOT, "agent_system/environments/env_manager.py")
    if not os.path.exists(path):
        print(f"[patch] WARNING: {path} not found, skipping env manager patch")
        return False

    with open(path) as f:
        content = f.read()

    if ATS_MARKER in content:
        print("[patch] env_manager.py already patched, skipping")
        return True

    # Insert before the 'appworld_skill' case
    old = '    elif "appworld_skill" in config.env.env_name.lower():'

    new = """    elif "appworld_ats" in config.env.env_name.lower():  {marker}
        from scripts.verl.env_appworld_ats import (
            build_appworld_ats_envs,
            appworld_ats_projection,
            AppWorldATSEnvironmentManager,
        )

        env_config = OmegaConf.to_container(config.env.get("appworld_ats", {{}}), resolve=True)
        _envs = build_appworld_ats_envs(
            dataset_name="train",
            seed=config.env.seed,
            env_num=config.data.train_batch_size,
            group_n=group_n,
            env_config=env_config,
        )
        _val_envs = build_appworld_ats_envs(
            dataset_name="test_normal",
            seed=config.env.seed + 1000,
            env_num=config.data.val_batch_size,
            group_n=1,
            env_config=env_config,
        )
        projection_f = partial(appworld_ats_projection)
        envs = AppWorldATSEnvironmentManager(_envs, projection_f, config)
        val_envs = AppWorldATSEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "appworld_skill" in config.env.env_name.lower():""".format(marker=ATS_MARKER)

    if old not in content:
        print("[patch] WARNING: Cannot find 'appworld_skill' env dispatch pattern in env_manager.py")
        return False

    content = content.replace(old, new)
    with open(path, "w") as f:
        f.write(content)

    print("[patch] env_manager.py patched: added 'appworld_ats' environment")
    return True


def patch_rollout_loop():
    """Patch rollout_loop.py to propagate trajectory_text and sgc_score from env infos.

    veRL's gather_rollout_data only propagates episode_rewards, episode_lengths,
    and tool_callings. ATS needs trajectory_text (for verifier) and sgc_score
    (for SGC gate) from the env's final step info.
    """
    path = os.path.join(SELFSKILL_ROOT, "agent_system/multi_turn_rollout/rollout_loop.py")
    if not os.path.exists(path):
        print(f"[patch] WARNING: {path} not found, skipping rollout loop patch")
        return False

    with open(path) as f:
        content = f.read()

    if ATS_MARKER in content:
        print("[patch] rollout_loop.py already patched, skipping")
        return True

    # Patch: after episode_lengths/tool_callings assignment, add trajectory_text + sgc_score
    old = """                    # tool_callings
                    data['tool_callings'] = tool_callings[bs]
                    # success_rate
                    for key, value in success_rate.items():
                        data[key] = value"""

    new = """                    # tool_callings
                    data['tool_callings'] = tool_callings[bs]
                    # ATS: propagate trajectory_text + sgc/tgc + skills_used from env infos  {marker}
                    _last_info = total_infos[bs][-1] if total_infos[bs] else {{}}
                    if 'trajectory_text' in _last_info:
                        data['trajectory_text'] = _last_info['trajectory_text']
                    if 'sgc' in _last_info:
                        data['sgc_score'] = float(_last_info['sgc'])
                    if 'tgc' in _last_info:
                        data['tgc_score'] = float(_last_info['tgc'])
                    if 'skills_used' in _last_info:
                        data['skills_used'] = _last_info['skills_used']
                    # success_rate
                    for key, value in success_rate.items():
                        data[key] = value""".format(marker=ATS_MARKER)

    if old not in content:
        print("[patch] WARNING: Cannot find rollout loop gather pattern in rollout_loop.py")
        return False

    content = content.replace(old, new)
    with open(path, "w") as f:
        f.write(content)

    print("[patch] rollout_loop.py patched: added trajectory_text + sgc_score propagation")
    return True


def main():
    print(f"[patch] SELFSKILL_ROOT: {SELFSKILL_ROOT}")
    print(f"[patch] SPHERE_ROOT: {SPHERE_ROOT}")

    ok1 = patch_main_ppo()
    ok2 = patch_env_manager()
    ok3 = patch_rollout_loop()

    if ok1 and ok2 and ok3:
        print("[patch] All patches applied successfully")
    else:
        print("[patch] Some patches failed — check warnings above")
        sys.exit(1)


if __name__ == "__main__":
    main()
