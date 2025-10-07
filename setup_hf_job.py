#!/usr/bin/env python3
"""
Setup script to prepare the environment for Hugging Face Jobs.
This script checks dependencies and prepares the job environment.
"""

import os
import sys
import subprocess
import json
from pathlib import Path


def check_dependencies():
    """Check if required dependencies are installed."""
    print("Checking dependencies...")
    
    required_packages = [
        "torch",
        "transformers", 
        "huggingface_hub",
        "numpy",
        "pandas",
        "gymnasium"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {missing_packages}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False
    
    print("All dependencies found!")
    return True


def check_qragent_env():
    """Check if QRAgent_Env is available."""
    print("\nChecking QRAgent_Env...")
    
    qragent_path = Path("QRAgent_Env")
    if not qragent_path.exists():
        print("✗ QRAgent_Env directory not found")
        print("Please clone it with: git clone https://github.com/arjunneervannan/QRAgent_Env.git")
        return False
    
    # Check for required files
    required_files = [
        "QRAgent_Env/envs/factor_env.py",
        "QRAgent_Env/data/ff25_value_weighted.csv",
        "QRAgent_Env/factors/baseline.json"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("✗ Missing required files:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        return False
    
    print("✓ QRAgent_Env found and complete")
    return True


def check_hf_login():
    """Check if user is logged in to Hugging Face."""
    print("\nChecking Hugging Face login...")
    
    try:
        from huggingface_hub import whoami
        user_info = whoami()
        print(f"✓ Logged in as: {user_info['name']}")
        return True
    except Exception as e:
        print(f"✗ Not logged in: {e}")
        print("Please login with: huggingface-cli login")
        return False


def create_job_manifest():
    """Create a job manifest file."""
    print("\nCreating job manifest...")
    
    manifest = {
        "name": "qwen-factor-agent",
        "description": "Qwen2.5-VL-7B-Instruct factor evaluation agent",
        "image": "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel",
        "command": ["python", "hf_job_runner.py"],
        "flavor": "a10g-small",
        "timeout": "60m",
        "env": {
            "PYTHONPATH": "/workspace",
            "CUDA_VISIBLE_DEVICES": "0"
        },
        "files": [
            "hf_job_runner.py",
            "QRAgent_Env/",
            "requirements.txt"
        ]
    }
    
    with open("job_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    
    print("✓ Job manifest created: job_manifest.json")
    return True


def test_local_run():
    """Test if the job can run locally (basic test)."""
    print("\nTesting local execution...")
    
    try:
        # Test basic imports
        sys.path.append("QRAgent_Env")
        from envs.factor_env import FactorImproveEnv
        print("✓ QRAgent_Env imports work")
        
        # Test model loading (if possible)
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", trust_remote_code=True)
            print("✓ Model tokenizer loads successfully")
        except Exception as e:
            print(f"⚠ Model loading test failed: {e}")
            print("  This is expected if you don't have GPU/CUDA")
        
        return True
        
    except Exception as e:
        print(f"✗ Local test failed: {e}")
        return False


def main():
    """Main setup function."""
    print("Hugging Face Jobs Setup")
    print("=" * 30)
    
    checks = [
        ("Dependencies", check_dependencies),
        ("QRAgent_Env", check_qragent_env),
        ("HF Login", check_hf_login),
        ("Job Manifest", create_job_manifest),
        ("Local Test", test_local_run)
    ]
    
    all_passed = True
    
    for check_name, check_func in checks:
        try:
            if not check_func():
                all_passed = False
        except Exception as e:
            print(f"✗ {check_name} check failed: {e}")
            all_passed = False
    
    print("\n" + "=" * 30)
    if all_passed:
        print("✓ Setup complete! Ready to submit jobs.")
        print("\nNext steps:")
        print("1. Run: hf jobs run --flavor a10g-small pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel python hf_job_runner.py")
        print("2. Or use: python run_hf_job.py")
    else:
        print("✗ Setup incomplete. Please fix the issues above.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
