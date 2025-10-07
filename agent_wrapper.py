#!/usr/bin/env python3
"""
Clean wrapper for the Qwen2.5-VL-7B-Instruct Factor Agent.
This file provides a simplified interface to the Qwen agent without modifying QRAgent_Env.
"""

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, Any, List, Optional
import sys
import os

# Add QRAgent_Env to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'QRAgent_Env'))

from envs.factor_env import FactorImproveEnv
from factors.validate import validate_program


class QwenFactorAgentWrapper:
    """
    Clean wrapper for the Qwen2.5-VL-7B-Instruct factor evaluation agent.
    Provides a simplified interface for factor strategy development.
    """
    
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

    def create_simple_momentum_factor(self, lookback_days: int = 63) -> Dict[str, Any]:
        """Create a simple momentum factor program."""
        return {
            "nodes": [
                {"id": "x0", "op": "rolling_return", "n": lookback_days},
                {"id": "score", "op": "zscore_xs", "src": "x0"}
            ],
            "output": "score"
        }

    def create_enhanced_momentum_factor(self, long_term: int = 126, short_term: int = 21) -> Dict[str, Any]:
        """Create an enhanced momentum factor with multiple timeframes."""
        return {
            "nodes": [
                {"id": "x0", "op": "rolling_return", "n": long_term},
                {"id": "x1", "op": "rolling_return", "n": short_term},
                {"id": "x2", "op": "sub", "a": "x0", "b": "x1"},
                {"id": "x3", "op": "winsor_quantile", "src": "x2", "q": 0.02},
                {"id": "score", "op": "zscore_xs", "src": "x3"}
            ],
            "output": "score"
        }

    def create_mean_reversion_factor(self) -> Dict[str, Any]:
        """Create a mean reversion factor combining momentum and reversal signals."""
        return {
            "nodes": [
                {"id": "x0", "op": "rolling_return", "n": 126},
                {"id": "x1", "op": "rolling_return", "n": 21},
                {"id": "x2", "op": "sub", "a": "x0", "b": "x1"},
                {"id": "x3", "op": "rolling_return", "n": 5},
                {"id": "x4", "op": "ema", "n": 10, "src": "x3"},
                {"id": "x5", "op": "sub", "a": "x3", "b": "x4"},
                {"id": "x6", "op": "add", "a": "x2", "b": "x5"},
                {"id": "x7", "op": "winsor_quantile", "src": "x6", "q": 0.02},
                {"id": "score", "op": "zscore_xs", "src": "x7"}
            ],
            "output": "score"
        }
