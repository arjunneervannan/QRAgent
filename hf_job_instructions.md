# Hugging Face Jobs Configuration

## Job Setup

To run the Qwen2.5-VL-7B-Instruct factor agent on Hugging Face Jobs:

### 1. Repository Structure
Ensure your repository has the following structure:
```
your-repo/
├── hf_job_config.py          # Main job script
├── hf_job_requirements.txt   # Dependencies
├── QRAgent_Env/             # Cloned QRAgent_Env repository
│   ├── data/
│   │   └── ff25_value_weighted.csv
│   ├── factors/
│   │   └── baseline.json
│   └── ... (other QRAgent_Env files)
└── README.md
```

### 2. Job Configuration

**Hardware Requirements:**
- Instance: GPU instance (recommended: A100 or similar)
- Memory: At least 16GB GPU memory for 7B model
- Storage: At least 20GB for model and data

**Environment Variables:**
- `HF_TOKEN`: Your Hugging Face token (for model access)
- `CUDA_VISIBLE_DEVICES`: Set to 0 for single GPU

**Command:**
```bash
python hf_job_config.py
```

**Timeout:** 30-60 minutes (depending on evaluation length)

### 3. Expected Outputs

The job will generate:
- `evaluation_results.json`: Complete episode data and performance metrics
- `hf_job_plots/`: Generated plots from factor evaluations
- Console output with step-by-step progress

### 4. Monitoring

Monitor the job through:
- Hugging Face Jobs dashboard
- Console output for real-time progress
- Generated files for results

### 5. Troubleshooting

**Common Issues:**
- **Out of Memory**: Reduce model size or use gradient checkpointing
- **Model Loading**: Ensure HF_TOKEN is set correctly
- **Data Access**: Verify QRAgent_Env data files are present
- **Timeout**: Reduce max_steps in the evaluation loop

**Debug Mode:**
Add `--debug` flag or modify the script to enable verbose logging.

### 6. Customization

You can customize the evaluation by modifying:
- `max_steps`: Number of steps per episode
- `model_name`: Different Qwen model variant
- `temperature`: Generation temperature for more/less randomness
- `max_new_tokens`: Maximum tokens to generate per action

### 7. Results Analysis

After job completion, analyze:
- `evaluation_results.json` for performance metrics
- Generated plots for visual analysis
- Console logs for action sequences and rewards
