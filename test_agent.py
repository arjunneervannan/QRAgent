#!/usr/bin/env python3
"""
Simple test script to verify the Qwen agent works with QRAgent_Env.
"""

import sys
import os

# Add QRAgent_Env to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'QRAgent_Env'))

from qwen_agent import QwenFactorAgent, run_evaluation_episode
from envs.factor_env import FactorImproveEnv


def test_agent_basic():
    """Test basic agent functionality."""
    print("Testing Qwen Factor Agent...")
    
    # Initialize environment
    print("1. Initializing environment...")
    env = FactorImproveEnv(
        data_path="QRAgent_Env/data/ff25_value_weighted.csv",
        test_train_split=0.8,
        timesteps=5,  # Shorter test
        baseline_path="QRAgent_Env/factors/baseline.json",
        plot_path="test_plots"
    )
    
    # Initialize agent
    print("2. Initializing agent...")
    agent = QwenFactorAgent()
    
    # Test single action
    print("3. Testing single action...")
    obs, info = env.reset()
    action = agent.get_action(obs)
    print(f"Generated action: {action}")
    
    # Validate action
    is_valid = agent.validate_action(action)
    print(f"Action is valid: {is_valid}")
    
    # Test short episode
    print("4. Running short test episode...")
    episode_data = run_evaluation_episode(env, agent, max_steps=3)
    
    print(f"Test completed successfully!")
    print(f"Total reward: {episode_data['total_reward']:.3f}")
    print(f"Steps taken: {episode_data['steps_taken']}")


if __name__ == "__main__":
    test_agent_basic()
