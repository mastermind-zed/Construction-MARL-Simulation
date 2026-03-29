from pettingzoo import ParallelEnv
from env.construction_env import ConstructionEnv
import gymnasium as gym
import numpy as np

class ConstructionParallelEnv(ParallelEnv):
    """
    PettingZoo ParallelEnv wrapper for the Construction Scenario.
    """
    metadata = {'render.modes': ['human'], "name": "construction_v0"}

    def __init__(self, render=False, num_robots=4, comm_mode="hybrid", config=None):
        self.underlying_env = ConstructionEnv(render=render, num_robots=num_robots, comm_mode=comm_mode, config=config)
        self.agents = [f"robot_{i}" for i in range(num_robots)]
        self.possible_agents = self.agents[:]
        
        self._action_spaces = {agent: self.underlying_env.action_space[agent] for agent in self.agents}
        self._observation_spaces = {agent: self.underlying_env.observation_space[agent] for agent in self.agents}

    def observation_space(self, agent):
        return self._observation_spaces[agent]

    def action_space(self, agent):
        return self._action_spaces[agent]

    def reset(self, seed=None, options=None):
        obs_dict = self.underlying_env.reset()
        self.agents = self.possible_agents[:]
        return obs_dict, {}

    def step(self, actions):
        obs_dict, rewards, terminations, truncations, infos = self.underlying_env.step(actions)
        return obs_dict, rewards, terminations, truncations, infos

    def render(self):
        pass

    def close(self):
        self.underlying_env.close()

if __name__ == "__main__":
    env = ConstructionParallelEnv(render=True)
    obs = env.reset()
    for _ in range(50):
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        obs, rewards, terms, truncs, infos = env.step(actions)
    env.close()
