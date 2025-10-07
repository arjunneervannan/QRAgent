#!/usr/bin/env python3
"""
Basic evaluation loop for the Qwen Factor Agent.
This file provides a clean interface to run factor strategy evaluation without modifying QRAgent_Env.
"""

import sys
import os
import json
from typing import Dict, Any, List, Optional
from pathlib import Path

# Add QRAgent_Env to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'QRAgent_Env'))

from envs.factor_env import FactorImproveEnv
from agent_wrapper import QwenFactorAgentWrapper


class FactorEvaluationLoop:
    """
    Basic evaluation loop for factor strategy development.
    Provides a clean interface to run episodes and evaluate agent performance.
    """
    
    def __init__(self, 
                 data_path: str = "QRAgent_Env/data/ff25_value_weighted.csv",
                 test_train_split: float = 0.8,
                 timesteps: int = 10,
                 baseline_path: str = "QRAgent_Env/factors/baseline.json",
                 plot_path: str = "evaluation_plots"):
        """Initialize the evaluation loop with environment parameters."""
        self.data_path = data_path
        self.test_train_split = test_train_split
        self.timesteps = timesteps
        self.baseline_path = baseline_path
        self.plot_path = plot_path
        
        # Initialize environment
        self.env = FactorImproveEnv(
            data_path=data_path,
            test_train_split=test_train_split,
            timesteps=timesteps,
            baseline_path=baseline_path,
            plot_path=plot_path
        )
        
        # Create plot directory
        Path(plot_path).mkdir(parents=True, exist_ok=True)
    
    def run_episode(self, agent: QwenFactorAgentWrapper, max_steps: Optional[int] = None) -> Dict[str, Any]:
        """
        Run a single evaluation episode with the given agent.
        
        Args:
            agent: The QwenFactorAgentWrapper instance
            max_steps: Maximum number of steps (None for environment default)
            
        Returns:
            Dictionary containing episode data and results
        """
        print("Starting evaluation episode...")
        
        # Reset environment
        obs, info = self.env.reset()
        print(f"Initial observation: Budget={obs['budget_left']}")
        
        episode_data = {
            "actions": [],
            "rewards": [],
            "observations": [obs],
            "total_reward": 0.0,
            "steps_taken": 0,
            "final_performance": None
        }
        
        max_steps = max_steps or self.timesteps
        
        for step in range(max_steps):
            print(f"\n--- Step {step + 1} ---")
            
            # Get action from agent
            action = agent.get_action(obs)
            print(f"Agent action: {action}")
            
            # Validate action
            if not agent.validate_action(action):
                print("Warning: Invalid action generated, using fallback")
                action = {"type": "OBSERVE", "tool": "describe_data"}
            
            # Take step in environment
            obs, reward, terminated, truncated, info = self.env.step(action)
            
            print(f"Reward: {reward:.3f}, Budget: {obs['budget_left']}")
            print(f"Terminated: {terminated}, Truncated: {truncated}")
            
            # Store episode data
            episode_data["actions"].append(action)
            episode_data["rewards"].append(reward)
            episode_data["observations"].append(obs)
            episode_data["total_reward"] += reward
            episode_data["steps_taken"] += 1
            
            # Check if episode is done
            if terminated or truncated:
                print(f"Episode ended. Total reward: {episode_data['total_reward']:.3f}")
                break
        
        # Store final performance if available
        final_obs = episode_data["observations"][-1]
        if "investment_performance" in final_obs:
            episode_data["final_performance"] = final_obs["investment_performance"]
        
        return episode_data
    
    def run_multiple_episodes(self, 
                             agent: QwenFactorAgentWrapper, 
                             num_episodes: int = 3,
                             max_steps_per_episode: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Run multiple evaluation episodes.
        
        Args:
            agent: The QwenFactorAgentWrapper instance
            num_episodes: Number of episodes to run
            max_steps_per_episode: Maximum steps per episode
            
        Returns:
            List of episode data dictionaries
        """
        print(f"Running {num_episodes} evaluation episodes...")
        
        all_episodes = []
        
        for episode in range(num_episodes):
            print(f"\n{'='*50}")
            print(f"EPISODE {episode + 1}/{num_episodes}")
            print(f"{'='*50}")
            
            episode_data = self.run_episode(agent, max_steps_per_episode)
            all_episodes.append(episode_data)
            
            # Print episode summary
            self._print_episode_summary(episode + 1, episode_data)
        
        return all_episodes
    
    def _print_episode_summary(self, episode_num: int, episode_data: Dict[str, Any]):
        """Print a summary of the episode results."""
        print(f"\nEpisode {episode_num} Summary:")
        print(f"  Total reward: {episode_data['total_reward']:.3f}")
        print(f"  Steps taken: {episode_data['steps_taken']}")
        print(f"  Average reward per step: {episode_data['total_reward'] / episode_data['steps_taken']:.3f}")
        
        if episode_data['final_performance']:
            perf = episode_data['final_performance']
            print(f"  Final Performance:")
            print(f"    Strategy Sharpe (net): {perf.get('strategy_sharpe_net', 0):.3f}")
            print(f"    Strategy Sharpe (gross): {perf.get('strategy_sharpe_gross', 0):.3f}")
            print(f"    Baseline Sharpe: {perf.get('baseline_sharpe', 0):.3f}")
            if 'improvement' in perf:
                print(f"    Improvement: {perf['improvement']:.3f}")
    
    def print_overall_summary(self, all_episodes: List[Dict[str, Any]]):
        """Print overall summary across all episodes."""
        print(f"\n{'='*60}")
        print("OVERALL EVALUATION SUMMARY")
        print(f"{'='*60}")
        
        total_rewards = [ep['total_reward'] for ep in all_episodes]
        steps_taken = [ep['steps_taken'] for ep in all_episodes]
        
        print(f"Episodes run: {len(all_episodes)}")
        print(f"Average total reward: {sum(total_rewards) / len(total_rewards):.3f}")
        print(f"Average steps per episode: {sum(steps_taken) / len(steps_taken):.1f}")
        print(f"Best episode reward: {max(total_rewards):.3f}")
        print(f"Worst episode reward: {min(total_rewards):.3f}")
        
        # Performance summary
        final_performances = [ep['final_performance'] for ep in all_episodes if ep['final_performance']]
        if final_performances:
            sharpe_ratios = [perf.get('strategy_sharpe_net', 0) for perf in final_performances]
            improvements = [perf.get('improvement', 0) for perf in final_performances if 'improvement' in perf]
            
            print(f"\nPerformance Metrics:")
            print(f"  Average Sharpe ratio: {sum(sharpe_ratios) / len(sharpe_ratios):.3f}")
            print(f"  Best Sharpe ratio: {max(sharpe_ratios):.3f}")
            if improvements:
                print(f"  Average improvement: {sum(improvements) / len(improvements):.3f}")
    
    def save_results(self, all_episodes: List[Dict[str, Any]], filename: str = "evaluation_results.json"):
        """Save evaluation results to a JSON file."""
        # Convert any non-serializable objects to strings
        serializable_episodes = []
        for episode in all_episodes:
            serializable_episode = episode.copy()
            # Convert observations to serializable format
            serializable_episode['observations'] = [
                {k: v for k, v in obs.items() if isinstance(v, (str, int, float, bool, list, dict))}
                for obs in episode['observations']
            ]
            serializable_episodes.append(serializable_episode)
        
        with open(filename, 'w') as f:
            json.dump(serializable_episodes, f, indent=2, default=str)
        
        print(f"Results saved to: {filename}")


def main():
    """Main evaluation function demonstrating the evaluation loop."""
    print("Qwen Factor Agent - Basic Evaluation Loop")
    print("=" * 50)
    
    # Initialize evaluation loop
    print("Initializing evaluation loop...")
    eval_loop = FactorEvaluationLoop(
        data_path="QRAgent_Env/data/ff25_value_weighted.csv",
        test_train_split=0.8,
        timesteps=10,
        baseline_path="QRAgent_Env/factors/baseline.json",
        plot_path="evaluation_plots"
    )
    
    # Initialize agent
    print("Initializing Qwen agent...")
    agent = QwenFactorAgentWrapper()
    
    # Run evaluation
    print("Running evaluation...")
    all_episodes = eval_loop.run_multiple_episodes(agent, num_episodes=3)
    
    # Print summary and save results
    eval_loop.print_overall_summary(all_episodes)
    eval_loop.save_results(all_episodes)
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
