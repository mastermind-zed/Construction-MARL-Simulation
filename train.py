import argparse
import yaml
from ray import tune
from ray.tune import RunConfig
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from ray.train import CheckpointConfig
from agents.marl_wrapper import ConstructionParallelEnv
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv

def env_creator(config):
    env = ConstructionParallelEnv(
        render=config.get("env_config", {}).get("render", False), 
        num_robots=config.get("env_config", {}).get("num_robots", 4),
        comm_mode=config.get("env_config", {}).get("comm_mode", "hybrid"),
        config={**config.get("env_config", {}), **config.get("scenario_details", {})}
    )
    return ParallelPettingZooEnv(env)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/experiment_config.yaml")
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config_data = yaml.safe_load(f)

    register_env("construction_v0", lambda cfg: env_creator(config_data))

    # Configure PPO (MAPPO) with Multi-Agent Policy
    ppo_config = (
        PPOConfig()
        .environment("construction_v0")
        .framework(config_data["training_config"]["framework"])
        .resources(num_cpus_for_main_process=1)
        .env_runners(num_env_runners=config_data["training_config"]["num_workers"])
        .training(
            train_batch_size=config_data["training_config"]["train_batch_size"],
            lr=config_data["training_config"]["lr"],
            gamma=config_data["training_config"]["gamma"],
        )
        .multi_agent(
            policies={f"robot_{i}": (None, None, None, {}) for i in range(config_data["env_config"]["num_robots"])},
            policy_mapping_fn=lambda agent_id, *args, **kwargs: agent_id,
        )
    )

    tune.Tuner(
        "PPO",
        run_config=RunConfig(
            stop={"training_iteration": config_data["training_config"].get("stop_iterations", 100)},
            checkpoint_config=CheckpointConfig(num_to_keep=5, checkpoint_at_end=True)
        ),
        param_space=ppo_config.to_dict(),
    ).fit()
