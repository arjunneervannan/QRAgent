# Qwen2.5-VL-7B-Instruct Factor Agent

A clean implementation of a factor evaluation agent using Qwen2.5-VL-7B-Instruct for the QRAgent_Env environment. This agent can observe market data, propose factor strategies, and evaluate their performance through an interactive environment.

## Overview

This implementation provides:
- A clean Qwen2.5-VL-7B-Instruct agent wrapper that can interact with the QRAgent_Env
- Basic action generation based on observations
- Factor strategy evaluation using the Factor DSL
- Simple evaluation loop for testing and development

## Files

- `agent.py` - Simple Qwen agent wrapper with query method
- `evaluation_loop.py` - Basic evaluation loop for running episodes
- `requirements.txt` - Python dependencies
- `README.md` - This file
- `QRAgent_Env/` - The core reinforcement learning environment (unchanged)

## Installation

1. Clone the QRAgent_Env repository:
```bash
git clone https://github.com/arjunneervannan/QRAgent_Env.git
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure you have the QRAgent_Env data files in place (ff25_value_weighted.csv)

## Usage

### Basic Evaluation

Run the evaluation loop:

```bash
python evaluation_loop.py
```

### Using the Agent

```python
from agent import QwenAgent
from evaluation_loop import FactorEvaluationLoop

# Initialize agent
agent = QwenAgent()

# Query the agent with a custom prompt
response = agent.query("What is a good momentum factor strategy?")

# Initialize evaluation loop
eval_loop = FactorEvaluationLoop()

# Run a single episode
episode_data = eval_loop.run_episode(agent)

# Run multiple episodes
all_episodes = eval_loop.run_multiple_episodes(agent, num_episodes=3)
```

## How It Works

### Agent Architecture

The agent uses Qwen2.5-VL-7B-Instruct to generate actions based on environment observations:

1. **Observation Processing**: Formats the current state into a text prompt
2. **Action Generation**: Uses the language model to generate JSON actions
3. **Action Validation**: Ensures generated actions are valid
4. **Fallback Handling**: Uses simple fallback actions if generation fails

### Action Types

The agent can perform three types of actions:

1. **OBSERVE**: Analyze data using built-in tools
   - `describe_data`: Get data statistics
   - `plot_returns`: Generate return distribution plots
   - `analyze_factor_performance`: Analyze factor performance

2. **FACTOR_IMPROVE**: Propose new factor strategies using the Factor DSL

3. **STOP**: End the episode and trigger final evaluation

### Factor DSL

The agent generates factor strategies using a JSON-based Domain Specific Language with operations like:
- `rolling_return`: Calculate rolling returns
- `ema`: Exponential moving average
- `zscore_xs`: Cross-sectional z-scores
- `winsor_quantile`: Winsorize data
- `add`, `sub`, `mul`: Arithmetic operations
- `combine`: Combine multiple signals

## Example Factor Strategy

```json
{
  "nodes": [
    {"id": "x0", "op": "rolling_return", "n": 126},
    {"id": "x1", "op": "rolling_return", "n": 21},
    {"id": "x2", "op": "sub", "a": "x0", "b": "x1"},
    {"id": "x3", "op": "winsor_quantile", "src": "x2", "q": 0.02},
    {"id": "score", "op": "zscore_xs", "src": "x3"}
  ],
  "output": "score"
}
```

## Performance Metrics

The agent is evaluated on:
- **Sharpe Ratio**: Risk-adjusted returns
- **Turnover**: Portfolio rebalancing frequency
- **Information Ratio**: Active return vs tracking error
- **Maximum Drawdown**: Largest peak-to-trough decline

## Limitations

This is a basic implementation without:
- Reinforcement learning
- Memory or state tracking across episodes
- Advanced prompt engineering
- Fine-tuning on factor data
- Sophisticated action selection strategies

## Future Improvements

Potential enhancements:
1. Add memory/context tracking
2. Implement more sophisticated prompt engineering
3. Add few-shot examples of good factor strategies
4. Implement action validation and correction
5. Add performance-based action selection
6. Fine-tune the model on factor data

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Use a smaller model or reduce batch size
2. **JSON Parsing Errors**: The agent has fallback mechanisms for invalid JSON
3. **Environment Errors**: Ensure QRAgent_Env is properly installed and data files are present

### Debug Mode

Enable debug output by modifying the agent to print more detailed information about the generation process.

## License

This project uses the QRAgent_Env library (MIT License) and Qwen2.5-VL-7B-Instruct model.