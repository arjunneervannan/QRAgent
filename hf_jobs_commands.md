# Hugging Face Jobs Commands

## Prerequisites

1. Install and login to Hugging Face:
```bash
pip install huggingface_hub
huggingface-cli login
```

2. Ensure you have the QRAgent_Env repository cloned and accessible

## Direct Command Line Usage

### Basic Job Submission

```bash
# Submit the job directly using hf jobs command
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

### Different Hardware Flavors

```bash
# Small GPU
hf jobs run --flavor a10g-small pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel python hf_job_runner.py

# Large GPU
hf jobs run --flavor a10g-large pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel python hf_job_runner.py

# A100 GPU
hf jobs run --flavor a100-40gb pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel python hf_job_runner.py
```

## Programmatic Submission

### Using Python Script

```bash
# Submit single job
python run_hf_job.py

# Submit multiple jobs with different hardware
python run_hf_job.py --multi
```

### Using Python API Directly

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

## Monitoring Jobs

### Check Job Status

```bash
# List all your jobs
hf jobs list

# Check specific job status
hf jobs status <job_id>

# Get job logs
hf jobs logs <job_id>
```

### Python API Monitoring

```python
from huggingface_hub import inspect_job, fetch_job_logs

# Check job status
job_info = inspect_job(job_id="your_job_id")
print(f"Status: {job_info.status.stage}")

# Get logs
logs = fetch_job_logs(job_id="your_job_id")
for log in logs:
    print(log)
```

## Job Configuration Options

### Hardware Flavors

- `cpu-basic`: Basic CPU
- `a10g-small`: Small A10G GPU (24GB)
- `a10g-large`: Large A10G GPU (48GB)
- `a100-40gb`: A100 40GB GPU
- `a100-80gb`: A100 80GB GPU

### Timeout Options

- `30m`: 30 minutes
- `60m`: 1 hour
- `2h`: 2 hours
- `4h`: 4 hours

### Environment Variables

- `PYTHONPATH`: Set to `/workspace` for proper imports
- `CUDA_VISIBLE_DEVICES`: Set to `0` for single GPU usage
- `HF_TOKEN`: Your Hugging Face token (if needed)

## Troubleshooting

### Common Issues

1. **Out of Memory**: Use a larger GPU flavor or reduce model size
2. **Import Errors**: Ensure QRAgent_Env is properly accessible
3. **Timeout**: Increase timeout for longer evaluations
4. **Authentication**: Make sure you're logged in with `huggingface-cli login`

### Debug Mode

Add debug output to the job runner:

```bash
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

## Expected Outputs

The job will generate:
- `evaluation_results.json`: Complete episode data
- `hf_job_plots/`: Generated plots
- Console output with step-by-step progress
- Performance metrics and final results
