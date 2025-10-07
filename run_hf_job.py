#!/usr/bin/env python3
"""
Script to submit and run the Qwen factor agent as a Hugging Face Job.
This script uses the huggingface_hub library to submit jobs programmatically.
"""

from huggingface_hub import run_job
import os
import sys


def submit_qwen_factor_job():
    """Submit the Qwen factor agent job to Hugging Face."""
    
    print("Submitting Qwen Factor Agent Job to Hugging Face...")
    
    # Job configuration
    job_config = {
        "image": "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel",  # PyTorch image with CUDA
        "command": ["python", "hf_job_runner.py"],
        "flavor": "a10g-small",  # GPU flavor - adjust based on your needs
        "timeout": "60m",  # 60 minutes timeout
        "env": {
            "PYTHONPATH": "/workspace",
            "CUDA_VISIBLE_DEVICES": "0"
        }
    }
    
    print(f"Job configuration:")
    print(f"  Image: {job_config['image']}")
    print(f"  Command: {' '.join(job_config['command'])}")
    print(f"  Flavor: {job_config['flavor']}")
    print(f"  Timeout: {job_config['timeout']}")
    
    try:
        # Submit the job
        job = run_job(**job_config)
        
        print(f"\nJob submitted successfully!")
        print(f"Job ID: {job.id}")
        print(f"Job URL: {job.url}")
        print(f"Status: {job.status.stage}")
        
        return job
        
    except Exception as e:
        print(f"Error submitting job: {e}")
        return None


def submit_with_different_flavors():
    """Submit jobs with different hardware flavors for comparison."""
    
    flavors = [
        ("a10g-small", "Small A10G GPU"),
        ("a10g-large", "Large A10G GPU"),
        ("a100-40gb", "A100 40GB GPU"),
    ]
    
    jobs = []
    
    for flavor, description in flavors:
        print(f"\nSubmitting job with {description}...")
        
        job_config = {
            "image": "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel",
            "command": ["python", "hf_job_runner.py"],
            "flavor": flavor,
            "timeout": "60m",
            "env": {
                "PYTHONPATH": "/workspace",
                "CUDA_VISIBLE_DEVICES": "0"
            }
        }
        
        try:
            job = run_job(**job_config)
            jobs.append((job, description))
            print(f"  Job ID: {job.id}")
            print(f"  URL: {job.url}")
        except Exception as e:
            print(f"  Error: {e}")
    
    return jobs


if __name__ == "__main__":
    print("Hugging Face Jobs Submission Script")
    print("=" * 40)
    
    # Check if user wants to run multiple flavors
    if len(sys.argv) > 1 and sys.argv[1] == "--multi":
        print("Submitting jobs with multiple hardware flavors...")
        jobs = submit_with_different_flavors()
        
        print(f"\nSubmitted {len(jobs)} jobs:")
        for job, description in jobs:
            print(f"  {description}: {job.url}")
    else:
        # Submit single job
        job = submit_qwen_factor_job()
        
        if job:
            print(f"\nMonitor your job at: {job.url}")
            print("Use 'hf jobs logs <job_id>' to view logs")
            print("Use 'hf jobs status <job_id>' to check status")
