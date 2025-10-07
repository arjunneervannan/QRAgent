# Hugging Face Jobs Setup for Qwen Factor Agent

This document provides a complete guide for running the Qwen2.5-VL-7B-Instruct factor agent using [Hugging Face Jobs](https://huggingface.co/docs/huggingface_hub/en/guides/jobs).

## Quick Start

### 1. Setup and Login

```bash
# Install dependencies
pip install huggingface_hub torch transformers

# Login to Hugging Face
huggingface-cli login

# Run setup check
python setup_hf_job.py
```

### 2. Submit Job

```bash
# Direct command line submission
hf jobs run \
  --flavor a10g-small \
  --timeout 60m \
  pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel \
  python hf_job_runner.py
```

### 3. Monitor Job

```bash
# Check job status
hf jobs list

# View logs
hf jobs logs <job_id>
```

## File Structure

```
QRAgent/
├── hf_job_runner.py          # Main job execution script
├── run_hf_job.py             # Programmatic job submission
├── setup_hf_job.py           # Setup and validation script
├── hf_job_requirements.txt   # Job dependencies
├── hf_jobs_commands.md       # Command reference
├── HF_JOBS_SETUP.md          # This file
└── QRAgent_Env/              # Cloned QRAgent_Env repository
    ├── data/
    │   └── ff25_value_weighted.csv
    ├── factors/
    │   └── baseline.json
    └── envs/
        └── factor_env.py
```

## Job Configuration

### Hardware Options

| Flavor | Description | Memory | Use Case |
|--------|-------------|---------|----------|
| `a10g-small` | Small A10G GPU | 24GB | Basic evaluation |
| `a10g-large` | Large A10G GPU | 48GB | Extended evaluation |
| `a100-40gb` | A100 40GB | 40GB | Large model inference |
| `a100-80gb` | A100 80GB | 80GB | Maximum performance |

### Timeout Options

- `30m`: 30 minutes (quick test)
- `60m`: 1 hour (standard evaluation)
- `2h`: 2 hours (extended evaluation)
- `4h`: 4 hours (comprehensive analysis)

## Usage Examples

### Basic Job Submission

```bash
hf jobs run \
  --flavor a10g-small \
  --timeout 60m \
  pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel \
  python hf_job_runner.py
```

### With Environment Variables

```bash
hf jobs run \
  --flavor a10g-small \
  --timeout 60m \
  --env PYTHONPATH=/workspace \
  --env CUDA_VISIBLE_DEVICES=0 \
  pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel \
  python hf_job_runner.py
```

### Programmatic Submission

```python
from huggingface_hub import run_job

job = run_job(
    image="pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel",
    command=["python", "hf_job_runner.py"],
    flavor="a10g-small",
    timeout="60m",
    env={
        "PYTHONPATH": "/workspace",
        "CUDA_VISIBLE_DEVICES": "0"
    }
)

print(f"Job URL: {job.url}")
```

### Multiple Hardware Comparison

```bash
# Submit jobs with different hardware
python run_hf_job.py --multi
```

## Expected Outputs

The job will generate:

1. **Console Output**: Step-by-step progress and results
2. **evaluation_results.json**: Complete episode data with:
   - Actions taken
   - Rewards received
   - Performance metrics
   - Final factor strategy
3. **hf_job_plots/**: Generated plots from factor evaluations
4. **Performance Metrics**:
   - Sharpe ratio (net and gross)
   - Baseline comparison
   - Improvement metrics
   - Turnover analysis

## Monitoring and Debugging

### Check Job Status

```bash
# List all jobs
hf jobs list

# Check specific job
hf jobs status <job_id>

# View logs
hf jobs logs <job_id>
```

### Python API Monitoring

```python
from huggingface_hub import inspect_job, fetch_job_logs

# Check status
job_info = inspect_job(job_id="your_job_id")
print(f"Status: {job_info.status.stage}")

# Get logs
logs = fetch_job_logs(job_id="your_job_id")
for log in logs:
    print(log)
```

## Troubleshooting

### Common Issues

1. **Out of Memory**
   - Use larger GPU flavor (`a10g-large` or `a100-40gb`)
   - Reduce model precision in the code

2. **Import Errors**
   - Ensure QRAgent_Env is properly cloned
   - Check PYTHONPATH environment variable

3. **Authentication Issues**
   - Run `huggingface-cli login`
   - Check your HF token permissions

4. **Timeout Issues**
   - Increase timeout parameter
   - Reduce max_steps in evaluation

### Debug Mode

```bash
# Run with debug output
hf jobs run \
  --flavor a10g-small \
  --timeout 60m \
  pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel \
  python -c "
import sys
sys.path.append('QRAgent_Env')
exec(open('hf_job_runner.py').read())
"
```

## Cost Estimation

Hugging Face Jobs are pay-as-you-go. Approximate costs:

- **a10g-small**: ~$0.50-1.00 per hour
- **a10g-large**: ~$1.00-2.00 per hour
- **a100-40gb**: ~$2.00-4.00 per hour

## Next Steps

1. **Run Setup**: `python setup_hf_job.py`
2. **Submit Job**: Use one of the command examples above
3. **Monitor Progress**: Check job status and logs
4. **Analyze Results**: Review generated outputs and metrics
5. **Iterate**: Modify parameters and resubmit for different evaluations

## Support

- [Hugging Face Jobs Documentation](https://huggingface.co/docs/huggingface_hub/en/guides/jobs)
- [QRAgent_Env Repository](https://github.com/arjunneervannan/QRAgent_Env)
- [Qwen Model Documentation](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
