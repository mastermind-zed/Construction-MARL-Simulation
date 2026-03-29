import pygame
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import time
import sys

class ConstructionEnv(gym.Env):
    """
    2D PyGame Environment for Multi-Agent Construction Scenario.
    """
    def __init__(self, render=False, num_robots=4, comm_mode="hybrid", config=None):
        super(ConstructionEnv, self).__init__()
        self.render_mode = render
        self.num_robots = num_robots
        self.comm_mode = comm_mode 
        self.config = config or {}
        
        # Environmental disturbances
        self.failure_prob = self.config.get("failure_rate", 0.005)
        self.obstacle_count = self.config.get("obstacle_count", 5)
        self.obstacles = []
        for _ in range(self.obstacle_count):
            self.obstacles.append({
                "pos": np.array([np.random.uniform(150, 650), np.random.uniform(150, 450)]),
                "size": np.array([40, 40])
            })
        
        # Grid/World settings
        self.main_width, self.main_height = 800, 600
        self.hud_width = 250
        self.width = self.main_width + self.hud_width
        self.height = self.main_height
        
        # Action space: 0: Idle, 1: North, 2: South, 3: East, 4: West
        self.action_space = spaces.Dict({
            f"robot_{i}": spaces.Discrete(5) for i in range(num_robots)
        })
        
        # Observation space: 
        # decentralized: [x, y, vx, vy, battery, has_material] (6)
        # hybrid: Core (6) + 2 neighbors * [rel_x, rel_y, bat, mat] (8) = 14
        obs_size = 6
        if self.comm_mode == "hybrid":
            obs_size = 14
        elif self.comm_mode == "centralized":
            obs_size = 6 + (num_robots - 1) * 6 # Core + all others
            
        self.observation_space = spaces.Dict({
            f"robot_{i}": spaces.Box(low=-1, high=1, shape=(obs_size,), dtype=np.float32) 
            for i in range(num_robots)
        })

        self.robot_tex = None
        self.ground_tex = None
        if self.render_mode:
            pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("MARL Construction Site - Premium Simulation")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("Arial", 18)
            self.bold_font = pygame.font.SysFont("Arial", 20, bold=True)
            
            # Load Assets
            try:
                self.ground_tex = pygame.image.load("assets/ground.png")
                self.ground_tex = pygame.transform.scale(self.ground_tex, (self.main_width, self.main_height))
                self.robot_tex = pygame.image.load("assets/robot.png")
                self.robot_tex = pygame.transform.scale(self.robot_tex, (40, 40))
            except:
                self.ground_tex = None
                self.robot_tex = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize Robot States
        self.robot_states = []
        for i in range(self.num_robots):
            self.robot_states.append({
                "pos": np.array([np.random.uniform(100, self.main_width-100), 
                                 np.random.uniform(100, self.main_height-100)], dtype=np.float32),
                "battery": 100.0,
                "has_material": 0.0,
                "distance": 0.0,
                "is_failed": False,
                "failure_timer": 0
            })
            
        # Global metrics
        self.collision_count = 0
        self.step_count = 0
        self.total_progress = 0
            
        # Static Areas with Priority & Deadline
        self.storage_pos = np.array([100, 100])
        zones_cfg = self.config.get("zones", [
            {"name": "Zone A", "priority": 5, "deadline": 500},
            {"name": "Zone B", "priority": 3, "deadline": 800},
            {"name": "Zone C", "priority": 1, "deadline": 1000}
        ])
        
        self.construction_zones = []
        for i, z in enumerate(zones_cfg):
            self.construction_zones.append({
                "pos": np.array([700, 100 + i * 200]),
                "label": z["name"],
                "priority": z["priority"],
                "deadline": z["deadline"],
                "progress": 0
            })
        
        return self._get_obs()

    def step(self, actions):
        speed = 4.0
        for i in range(self.num_robots):
            old_pos = self.robot_states[i]["pos"].copy()
            # Environmental Disturbance: Stochastic Failure
            if self.robot_states[i]["is_failed"]:
                self.robot_states[i]["failure_timer"] -= 1
                if self.robot_states[i]["failure_timer"] <= 0:
                    self.robot_states[i]["is_failed"] = False
                continue # Skip actions if failed
            
            if np.random.random() < self.failure_prob:
                self.robot_states[i]["is_failed"] = True
                self.robot_states[i]["failure_timer"] = np.random.randint(20, 50)
                continue

            # Movement
            action = actions.get(f"robot_{i}", 0)
            new_pos = self.robot_states[i]["pos"].copy()
            if action == 1: # North
                new_pos[1] -= speed
            elif action == 2: # South
                new_pos[1] += speed
            elif action == 3: # East
                new_pos[0] += speed
            elif action == 4: # West
                new_pos[0] -= speed
            
            # Obstacle checks
            collision = False
            for obs in self.obstacles:
                if (new_pos[0] > obs["pos"][0] - 25 and new_pos[0] < obs["pos"][0] + 25 and
                    new_pos[1] > obs["pos"][1] - 25 and new_pos[1] < obs["pos"][1] + 25):
                    collision = True
                    break
            
            if not collision:
                self.robot_states[i]["pos"] = new_pos
            
            # Boundary checks (within main site)
            self.robot_states[i]["pos"] = np.clip(self.robot_states[i]["pos"], [20, 20], [self.main_width-20, self.main_height-20])
            
            # Distance tracking
            self.robot_states[i]["distance"] += np.linalg.norm(self.robot_states[i]["pos"] - old_pos) / 10.0
            
            # Interaction with Storage
            if np.linalg.norm(self.robot_states[i]["pos"] - self.storage_pos) < 40:
                if self.robot_states[i]["has_material"] < 1.0:
                    self.robot_states[i]["has_material"] = 1.0
                    self.robot_states[i]["battery"] = min(100, self.robot_states[i]["battery"] + 10) # Recharge slightly at storage
            
            # Interaction with Zones
            for zone in self.construction_zones:
                if np.linalg.norm(self.robot_states[i]["pos"] - zone["pos"]) < 40:
                    if self.robot_states[i]["has_material"] > 0:
                        self.robot_states[i]["has_material"] = 0
                        zone["progress"] += 1
            
            # Battery drain
            if action != 0:
                self.robot_states[i]["battery"] -= 0.08
            else:
                self.robot_states[i]["battery"] -= 0.01 # Idle drain
                
        # 2. Collision Detection
        for i in range(self.num_robots):
            for j in range(i + 1, self.num_robots):
                dist = np.linalg.norm(self.robot_states[i]["pos"] - self.robot_states[j]["pos"])
                if dist < 30: # Collision threshold
                    self.collision_count += 1
        
        self.step_count += 1
        self.total_progress = sum(z["progress"] for z in self.construction_zones)
        
        if self.render_mode:
            self.render()
            
        obs = self._get_obs()
        rewards = {f"robot_{i}": 0.0 for i in range(self.num_robots)}
        
        # Priority and Deadline based rewards
        global_reward = 0
        for zone in self.construction_zones:
            if self.step_count > zone["deadline"] and zone["progress"] < 50:
                global_reward -= 0.1 * zone["priority"] # Deadline penalty
                
        for i in range(self.num_robots):
            # Individual progress reward
            if actions.get(f"robot_{i}") != 0:
                rewards[f"robot_{i}"] -= 0.01 # Action penalty
            
            # Sharing some global success
            rewards[f"robot_{i}"] += global_reward
            
        terminations = {f"robot_{i}": self.robot_states[i]["battery"] <= 0 for i in range(self.num_robots)}
        truncations = {f"robot_{i}": self.step_count > 1000 for i in range(self.num_robots)}
        
        info = {
            "collisions": self.collision_count,
            "progress": self.total_progress,
            "steps": self.step_count
        }
        
        return obs, rewards, terminations, truncations, {f"robot_{i}": info for i in range(self.num_robots)}

    def _get_obs(self):
        obs = {}
        for i in range(self.num_robots):
            s = self.robot_states[i]
            # Core observation
            core = [
                s["pos"][0] / self.main_width,
                s["pos"][1] / self.main_height,
                0.0, 0.0, # Placeholder for velocity
                s["battery"] / 100.0,
                s["has_material"]
            ]
            
            if self.comm_mode == "hybrid":
                # Find 2 nearest neighbors
                neighbors = []
                for j in range(self.num_robots):
                    if i == j: continue
                    dist = np.linalg.norm(s["pos"] - self.robot_states[j]["pos"])
                    neighbors.append((dist, j))
                neighbors.sort()
                
                # Take top 2
                comm_data = []
                for k in range(min(2, len(neighbors))):
                    idx = neighbors[k][1]
                    ns = self.robot_states[idx]
                    rel_pos = (ns["pos"] - s["pos"]) / 100.0 # Normalized relative pos
                    comm_data.extend([rel_pos[0], rel_pos[1], ns["battery"]/100.0, ns["has_material"]])
                
                # Pad if fewer than 2 neighbors (not really possible here but for safety)
                while len(comm_data) < 8:
                    comm_data.extend([0.0, 0.0, 0.0, 0.0])
                
                obs[f"robot_{i}"] = np.array(core + comm_data, dtype=np.float32)
            
            elif self.comm_mode == "centralized":
                others = []
                for j in range(self.num_robots):
                    if i == j: continue
                    os = self.robot_states[j]
                    others.extend([
                        os["pos"][0] / self.main_width,
                        os["pos"][1] / self.main_height,
                        0.0, 0.0,
                        os["battery"] / 100.0,
                        os["has_material"]
                    ])
                obs[f"robot_{i}"] = np.array(core + others, dtype=np.float32)
                
            else: # decentralized
                obs[f"robot_{i}"] = np.array(core, dtype=np.float32)
                
        return obs

    def render(self):
        # 1. Main Construction Area
        if self.ground_tex:
            self.screen.blit(self.ground_tex, (0, 0))
        else:
            self.screen.fill((180, 160, 140), rect=(0, 0, self.main_width, self.main_height)) # Dirt color
        
        # Draw some "grid lines" or dust
        for x in range(0, self.main_width, 100):
            pygame.draw.line(self.screen, (160, 140, 120), (x, 0), (x, self.main_height), 1)
        for y in range(0, self.main_height, 100):
            pygame.draw.line(self.screen, (160, 140, 120), (0, y), (self.main_width, y), 1)

        # Draw Storage (Industrial blue)
        pygame.draw.circle(self.screen, (30, 80, 150), self.storage_pos, 45, 3)
        pygame.draw.circle(self.screen, (50, 100, 200), self.storage_pos, 35)
        lbl = self.font.render("MATERIAL STORAGE", True, (255, 255, 255))
        self.screen.blit(lbl, (self.storage_pos[0]-70, self.storage_pos[1]+45))
        
        # Draw construction zones (Construction orange)
        for zone in self.construction_zones:
            # Outer border
            z_rect = (zone["pos"][0]-35, zone["pos"][1]-35, 70, 70)
            color = (255, 140, 0) if self.step_count < zone["deadline"] else (200, 50, 50)
            pygame.draw.rect(self.screen, color, z_rect, 4)
            # Inner fill based on progress
            prog_height = int(70 * min(1.0, zone["progress"]/50.0))
            if prog_height > 0:
                pygame.draw.rect(self.screen, (200, 100, 0), (zone["pos"][0]-35, zone["pos"][1]+35-prog_height, 70, prog_height))
            
            lbl = self.font.render(f"{zone['label']} (P{zone['priority']})", True, (50, 50, 50))
            self.screen.blit(lbl, (zone["pos"][0]-45, zone["pos"][1]-55))
            
        # Draw Obstacles (Gray concrete)
        for obs in self.obstacles:
            pygame.draw.rect(self.screen, (100, 100, 100), (int(obs["pos"][0]-20), int(obs["pos"][1]-20), 40, 40))
            pygame.draw.rect(self.screen, (120, 120, 120), (int(obs["pos"][0]-15), int(obs["pos"][1]-15), 30, 30))

        # Draw robots
        for i in range(self.num_robots):
            pos = self.robot_states[i]["pos"]
            if self.robot_tex:
                # Rotate robot conceptually? For now just blit
                self.screen.blit(self.robot_tex, (int(pos[0]-20), int(pos[1]-20)))
            else:
                color = (255, 0, 0) if i < self.num_robots//2 else (0, 200, 0)
                if self.robot_states[i]["is_failed"]:
                    color = (150, 150, 150)
                pygame.draw.circle(self.screen, color, pos.astype(int), 18)
            
            # Carry indicator
            if self.robot_states[i]["has_material"] > 0:
                pygame.draw.rect(self.screen, (100, 50, 20), (pos[0]-5, pos[1]-5, 10, 10))

        # 2. HUD Area
        hud_rect = (self.main_width, 0, self.hud_width, self.main_height)
        pygame.draw.rect(self.screen, (40, 44, 52), hud_rect)
        pygame.draw.line(self.screen, (100, 100, 100), (self.main_width, 0), (self.main_width, self.main_height), 2)
        
        title = self.bold_font.render("CONSTRUCTION FLEET", True, (255, 200, 0))
        self.screen.blit(title, (self.main_width + 20, 20))
        
        for i in range(self.num_robots):
            y_off = 70 + i * 85
            s = self.robot_states[i]
            
            # Robot ID
            r_id = self.font.render(f"Robot {i}: Status", True, (200, 200, 200))
            self.screen.blit(r_id, (self.main_width + 20, y_off))
            
            # Battery Bar
            bat_color = (0, 255, 0) if s["battery"] > 30 else (255, 0, 0)
            pygame.draw.rect(self.screen, (60, 60, 60), (self.main_width + 20, y_off + 25, 200, 12))
            pygame.draw.rect(self.screen, bat_color, (self.main_width + 20, y_off + 25, int(s["battery"] * 2), 12))
            
            # Stats Text
            txt = f"Load: {'[X]' if s['has_material'] > 0 else '[ ]'} Dist: {s['distance']:.1f}m"
            if s["is_failed"]:
                txt = f"SYSTEM FAILURE - RECOVERING ({s['failure_timer']})"
            stats = self.font.render(txt, True, (150, 150, 150) if not s["is_failed"] else (255, 50, 50))
            self.screen.blit(stats, (self.main_width + 20, y_off + 45))

        pygame.display.flip()
        self.clock.tick(30)

    def close(self):
        if self.render_mode:
            pygame.quit()

if __name__ == "__main__":
    import yaml
    with open("config/experiment_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    print("Starting Methodology Step 2 Visualization... Close window to exit.")
    env = ConstructionEnv(render=True, 
                          num_robots=config["env_config"]["num_robots"], 
                          comm_mode="hybrid", 
                          config={**config["env_config"], **config["scenario_details"]})
    env.reset()
    running = True
    while running:
        # User input / Close check
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Simple random-walk logic for visualization
        actions = {f"robot_{i}": np.random.randint(0, 5) for i in range(env.num_robots)}
        obs, rewards, terminations, truncations, infos = env.step(actions)
        
        # Reset any robot that dies
        if any(terminations.values()):
            env.reset()
            
    env.close()
