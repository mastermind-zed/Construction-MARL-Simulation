# Multi-Agent RL Construction Automation

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Ray Rllib](https://img.shields.io/badge/Ray-Rllib-orange.svg)](https://docs.ray.io/en/latest/rllib/index.html)
[![PyGame](https://img.shields.io/badge/PyGame-2D-green.svg)](https://www.pygame.org/)

A Multi-Agent Reinforcement Learning (MARL) framework designed to simulate and optimize autonomous robot coordination in construction environments. This project uses the **MAPPO** (Multi-Agent Proximal Policy Optimization) algorithm to manage fleet logistics, battery constraints, and task allocation.

## 🏗️ Project Overview

This simulation provides a high-fidelity 2D environment where a fleet of robots must coordinate to haul materials from a central storage area to various construction zones.

### Key Features
- **MAPPO Integration**: Uses Ray Rllib for state-of-the-art multi-agent training.
- **Dynamic Logistics**: Robots must manage battery levels and delivery progress.
- **Premium Visualization**: Built with PyGame, featuring ground textures, robot sprites, and a real-time HUD.
- **Scalable**: Easily adjust the number of robots and environment parameters via YAML configuration.

## 🚀 Getting Started

### Prerequisites
- Windows OS (Optimized for PowerShell)
- Python 3.11 (Recommended for Ray compatibility)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/mastermind-zed/Multi-Agent-RL-Construction-Automation.git
   cd Multi-Agent-RL-Construction-Automation
   ```
2. Set up the environment and install dependencies:
   ```powershell
   python -m venv venv
   .\venv\Scripts\activate
   pip install "ray[rllib]" pygame torch pyyaml pettingzoo python-docx
   ```

## 🛠️ Usage

### Run Visualization
Watch the robots in action with random policies:
```powershell
python env/construction_env.py
```

### Start Training
Train the fleet using MAPPO:
```powershell
python train.py --config config/experiment_config.yaml
```

## 📁 Repository Structure
- `env/`: Core PyGame environment logic.
- `agents/`: PettingZoo MARL wrappers.
- `config/`: YAML configuration files.
- `assets/`: Textures and visual assets.
- `results/`: Training logs and saved models.

## 📄 License
This project is for research and educational purposes.
