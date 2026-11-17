"""Train the delay-randomized H1 rough task (h1_delay_cfg.py) with Isaac Lab's own rsl_rl train.py: registers the task
id, then runs train.py unchanged. Same PPO settings as Isaac-Velocity-Rough-H1-v0 (3000 iterations, seed 42).

Usage: python train_delay.py --headless [--max_iterations N --num_envs N]   -> logs/rsl_rl/h1_rough_delay/<run>
"""
import os
import runpy
import sys

import gymnasium as gym

LAB = os.path.join(os.environ.get("LAB", r"C:\Storage\IsaacSim\IsaacLab"), "scripts", "reinforcement_learning", "rsl_rl")
TASK = "Isaac-Velocity-Rough-H1-Delay-v0"
gym.register(id=TASK, entry_point="isaaclab.envs:ManagerBasedRLEnv", disable_env_checker=True, kwargs={
    "env_cfg_entry_point": "h1_delay_cfg:H1RoughDelayEnvCfg",
    "rsl_rl_cfg_entry_point":
        "isaaclab_tasks.manager_based.locomotion.velocity.config.h1.agents.rsl_rl_ppo_cfg:H1RoughPPORunnerCfg",
})
sys.path.insert(0, LAB)  # train.py imports cli_args from its own folder
# --experiment_name is ignored by this Isaac Lab version's cli_args, so set it as a hydra override
sys.argv = [os.path.join(LAB, "train.py"), "--task", TASK] + sys.argv[1:] + ["agent.experiment_name=h1_rough_delay"]
runpy.run_path(sys.argv[0], run_name="__main__")
