#!/usr/bin/env python3
"""
Simple evaluation loop for the Qwen Factor Agent.
"""

import sys
import os
import json

# Add QRAgent_Env to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'QRAgent_Env'))

from envs.factor_env import FactorImproveEnv
from agent import QwenAgent
from prompt.prompt import PromptBuilder


def main():
    """Simple evaluation loop."""
    # Initialize components
    env = FactorImproveEnv(
        data_path="QRAgent_Env/data/ff25_value_weighted.csv",
        test_train_split=0.8,
        timesteps=10,
        baseline_path="QRAgent_Env/factors/baseline.json",
        plot_path="evaluation_plots"
    )
    
    agent = QwenAgent()
    prompt_builder = PromptBuilder()
    obs, _ = env.reset()
    
    # Simple while loop
    while True:
        # Create prompt and query agent
        prompt = prompt_builder.build_basic_prompt("Factor development task", obs)
        response = agent.query(prompt)
        
        # Parse action from response
        try:
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            action = json.loads(response[start_idx:end_idx]) if start_idx != -1 else {"type": "STOP"}
        except:
            action = {"type": "STOP"}
        
        # Take step in environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"Action: {action['type']}, Reward: {reward:.3f}, Budget: {obs['budget_left']}")
        
        # Check if done
        if terminated or truncated:
            break
    
    print(f"Episode complete! Final reward: {sum(env.episode_rewards):.3f}")


if __name__ == "__main__":
    main()