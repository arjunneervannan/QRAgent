#!/usr/bin/env python3
"""
Simple Qwen2.5-VL-7B-Instruct agent wrapper.
Provides a clean interface to query the LLM with custom prompts.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional


class QwenAgent:
    """
    Simple wrapper for Qwen2.5-VL-7B-Instruct model.
    Provides a clean interface to query the LLM with custom prompts.
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

    def query(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.7) -> str:
        """
        Query the LLM with a custom prompt.
        
        Args:
            prompt: The input prompt to send to the model
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature for generation
            
        Returns:
            The model's response as a string
        """
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
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
        
        return new_response
