#!/usr/bin/env python3
"""
Hugging Face Jobs configuration for Qwen2.5-VL-7B-Instruct factor evaluation.
This script can be submitted as a Hugging Face job.
"""

import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, Any, List, Optional
import sys

# Add QRAgent_Env to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'QRAgent_Env'))

from envs.factor_env import FactorImproveEnv
from factors.validate import validate_program


class QwenFactorAgent:
    """Basic Qwen2.5-VL-7B-Instruct agent for factor evaluation."""
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct", device: str = "auto"):
        """Initialize the agent with the Qwen model."""
        self.model_name = model_name
        self.device = device
        
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map=device,
            trust_remote_code=True
        )
        
        # System prompt for the agent
        self.system_prompt = """You are an expert quantitative researcher specializing in price-based factor momentum strategies.

Your role is to build a factor model that can predict market timing across size and value factors.
There are 25 assets, which are split into 5 size buckets and 5 value buckets. You have access to price data for the assets and nothing more.
Your job is to generate price-based factor signals that can be used to predict the market timing of the size and value factors.

CORE CAPABILITIES:
- Analyze portfolio return data using statistical tools
- Design factor models using a JSON-based Domain Specific Language (DSL)
- Evaluate and improve factor performance through backtesting and edit factors

Factor model specifications:
- You have access to the following operations:
  - rolling_return: Calculate rolling returns over n periods
  - ema: Calculate exponential moving average
  - zscore_xs: Calculate cross-sectional z-scores
  - demean_xs: Demean cross-sectionally
  - winsor_quantile: Winsorize data at given quantiles
  - clip: Clip data between lo and hi values
  - delay: Delay data by d periods
  - add, sub, mul: Arithmetic operations between two signals
  - combine: Combine multiple signals with optional weights

PERFORMANCE OBJECTIVES:
- Maximize out-of-sample Sharpe ratio
- Control turnover and transaction costs
- Avoid data leakage and overfitting
- Ensure factor signals are economically meaningful

You have access to observation tools for data analysis and can propose complete factor programs. Your goal is to develop robust, profitable factor strategies through systematic analysis and iteration.

Valid actions (emit exactly ONE JSON object per step):
- OBSERVE:     {"type":"OBSERVE","tool":"<tool_name>","<params>":<values>}
  Available tools:
  - "describe_data": {"type":"OBSERVE","tool":"describe_data"}
  - "plot_returns": {"type":"OBSERVE","tool":"plot_returns"}
  - "analyze_factor_performance": {"type":"OBSERVE","tool":"analyze_factor_performance","factor_program":{...DSL JSON...}}
- FACTOR_IMPROVE: {"type":"FACTOR_IMPROVE","new_program":{...DSL JSON...}}
- STOP:        {"type":"STOP"}

Always respond with valid JSON only, no additional text."""

    def _format_observation(self, obs: Dict[str, Any]) -> str:
        """Format the observation for the model."""
        formatted = f"CURRENT STATE:\n"
        formatted += f"- Budget remaining: {obs.get('budget_left', 0)}\n"
        
        if 'current_program' in obs and obs['current_program']:
            formatted += f"- Current program: {json.dumps(obs['current_program'], indent=2)}\n"
        else:
            formatted += f"- Current program: None (baseline)\n"
            
        if 'last_eval' in obs and obs['last_eval']:
            formatted += f"- Last evaluation: {json.dumps(obs['last_eval'], indent=2)}\n"
            
        if 'equal_weight_baseline' in obs and obs['equal_weight_baseline']:
            formatted += f"- Equal weight baseline: {json.dumps(obs['equal_weight_baseline'], indent=2)}\n"
            
        if 'current_performance' in obs and obs['current_performance']:
            formatted += f"- Current performance: {json.dumps(obs['current_performance'], indent=2)}\n"
            
        if 'observation_result' in obs:
            formatted += f"- Observation result: {json.dumps(obs['observation_result'], indent=2)}\n"
            
        if 'investment_performance' in obs:
            formatted += f"- Investment performance: {json.dumps(obs['investment_performance'], indent=2)}\n"
            
        if 'validation_errors' in obs:
            formatted += f"- Validation errors: {obs['validation_errors']}\n"
            
        formatted += f"\nBased on this state, what action would you like to take?\n"
        formatted += f"Output ONE JSON action with no extra text.\n"
        formatted += f"Action JSON:"
        
        return formatted

    def _parse_model_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse the model's response to extract JSON action."""
        try:
            # Try to find JSON in the response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                action = json.loads(json_str)
                return action
            else:
                print(f"Warning: No JSON found in response: {response}")
                return None
                
        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse JSON from response: {e}")
            print(f"Response: {response}")
            return None

    def get_action(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Get action from the model based on current observation."""
        # Format the observation
        prompt = self.system_prompt + "\n\n" + self._format_observation(obs)
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract the new part (after the prompt)
        if prompt in response:
            new_response = response[len(prompt):].strip()
        else:
            new_response = response.strip()
        
        # Parse JSON action
        action = self._parse_model_response(new_response)
        
        if action is None:
            # Fallback: return a simple observe action
            print("Warning: Using fallback action")
            action = {"type": "OBSERVE", "tool": "describe_data"}
        
        return action

    def validate_action(self, action: Dict[str, Any]) -> bool:
        """Validate that the action is properly formatted."""
        if not isinstance(action, dict):
            return False
            
        if "type" not in action:
            return False
            
        action_type = action["type"]
        
        if action_type == "OBSERVE":
            return "tool" in action and action["tool"] in ["describe_data", "plot_returns", "analyze_factor_performance"]
        elif action_type == "FACTOR_IMPROVE":
            return "new_program" in action and isinstance(action["new_program"], dict)
        elif action_type == "STOP":
            return True
        else:
            return False


def run_evaluation_episode(env: FactorImproveEnv, agent: QwenFactorAgent, max_steps: int = 10) -> Dict[str, Any]:
    """Run a single evaluation episode."""
    print("Starting evaluation episode...")
    
    # Reset environment
    obs, info = env.reset()
    print(f"Initial observation: Budget={obs['budget_left']}")
    
    episode_data = {
        "actions": [],
        "rewards": [],
        "observations": [obs],
        "total_reward": 0.0,
        "steps_taken": 0
    }
    
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
        obs, reward, terminated, truncated, info = env.step(action)
        
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
    
    return episode_data


def main():
    """Main evaluation function for Hugging Face Jobs."""
    print("Qwen2.5-VL-7B-Instruct Factor Agent Evaluation (HF Jobs)")
    print("=" * 60)
    
    # Log environment info
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Initialize environment
    print("\nInitializing environment...")
    try:
        env = FactorImproveEnv(
            data_path="QRAgent_Env/data/ff25_value_weighted.csv",
            test_train_split=0.8,
            timesteps=10,
            baseline_path="QRAgent_Env/factors/baseline.json",
            plot_path="hf_job_plots"
        )
        print("Environment initialized successfully")
    except Exception as e:
        print(f"Error initializing environment: {e}")
        return
    
    # Initialize agent
    print("\nInitializing Qwen agent...")
    try:
        agent = QwenFactorAgent()
        print("Agent initialized successfully")
    except Exception as e:
        print(f"Error initializing agent: {e}")
        return
    
    # Run evaluation
    print("\nRunning evaluation episode...")
    try:
        episode_data = run_evaluation_episode(env, agent, max_steps=10)
        
        # Print results
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        print(f"Total reward: {episode_data['total_reward']:.3f}")
        print(f"Steps taken: {episode_data['steps_taken']}")
        print(f"Average reward per step: {episode_data['total_reward'] / episode_data['steps_taken']:.3f}")
        
        # Print final performance if available
        final_obs = episode_data["observations"][-1]
        if "investment_performance" in final_obs:
            perf = final_obs["investment_performance"]
            print(f"\nFinal Performance:")
            print(f"  Strategy Sharpe (net): {perf.get('strategy_sharpe_net', 0):.3f}")
            print(f"  Strategy Sharpe (gross): {perf.get('strategy_sharpe_gross', 0):.3f}")
            print(f"  Baseline Sharpe: {perf.get('baseline_sharpe', 0):.3f}")
            if 'improvement' in perf:
                print(f"  Improvement: {perf['improvement']:.3f}")
        
        # Save results to file
        results_file = "evaluation_results.json"
        with open(results_file, 'w') as f:
            json.dump(episode_data, f, indent=2, default=str)
        print(f"\nResults saved to: {results_file}")
        
        print("\nEvaluation complete!")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
