#!/usr/bin/env python3
"""
Test script for DIME implementation
"""

import sys
import os

# Add the project root to the Python path
sys.path.insert(0, '/home/holmes/projects/thesis/metaworld-algorithms')

try:
    # Test imports
    from metaworld_algorithms.rl.algorithms.dime import DIME, DIMEConfig, DiffusionPolicy
    print("✓ DIME imports successful")
    
    # Test configuration creation
    config = DIMEConfig(
        num_tasks=10,
        gamma=0.99,
        initial_temperature=1.0,
        tau=0.005,
        policy_tau=0.005,
        num_diffusion_steps=16,
        diffusion_hidden_dim=256,
        diffusion_num_layers=3,
        policy_delay=2,
        entropy_coefficient=0.1,
    )
    print("✓ DIMEConfig creation successful")
    
    # Test DiffusionPolicy creation
    policy = DiffusionPolicy(
        action_dim=4,
        hidden_dim=256,
        num_layers=3,
        num_diffusion_steps=16,
    )
    print("✓ DiffusionPolicy creation successful")
    
    print("\n🎉 All DIME implementation tests passed!")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
