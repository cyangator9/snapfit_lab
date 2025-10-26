"""
Script to verify if LSTM is properly instantiated in the model.
Run this after training starts to check model architecture.
"""

import torch
import sys
import os

def check_checkpoint_lstm(checkpoint_path):
    """Check if a checkpoint contains LSTM layers."""
    print(f"\n{'='*80}")
    print(f"🔍 LSTM VERIFICATION TOOL")
    print(f"{'='*80}\n")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        print(f"✅ Checkpoint loaded: {checkpoint_path}\n")
        
        # Check policy (actor)
        if 'policy' in checkpoint:
            policy = checkpoint['policy']
            print("📊 POLICY (Actor) Architecture:")
            print("-" * 80)
            
            lstm_found = False
            for key in policy.keys():
                if 'lstm' in key.lower():
                    print(f"  ✅ LSTM FOUND: {key} - Shape: {policy[key].shape}")
                    lstm_found = True
                elif 'weight' in key or 'bias' in key:
                    print(f"  Layer: {key} - Shape: {policy[key].shape}")
            
            if lstm_found:
                print("\n🎉 SUCCESS: LSTM layers detected in policy!")
            else:
                print("\n❌ WARNING: No LSTM layers found in policy!")
                print("   Model might be using feedforward only (LazyLinear).")
            print()
        
        # Check value (critic)
        if 'value' in checkpoint:
            value = checkpoint['value']
            print("📊 VALUE (Critic) Architecture:")
            print("-" * 80)
            
            lstm_found = False
            for key in value.keys():
                if 'lstm' in key.lower():
                    print(f"  ✅ LSTM FOUND: {key} - Shape: {value[key].shape}")
                    lstm_found = True
                elif 'weight' in key or 'bias' in key:
                    print(f"  Layer: {key} - Shape: {value[key].shape}")
            
            if lstm_found:
                print("\n🎉 SUCCESS: LSTM layers detected in critic!")
            else:
                print("\n❌ WARNING: No LSTM layers found in critic!")
            print()
        
        print(f"{'='*80}\n")
        
    except Exception as e:
        print(f"❌ ERROR: Could not load checkpoint")
        print(f"   {str(e)}\n")
        return False
    
    return True

def find_latest_checkpoint():
    """Find the most recent checkpoint."""
    logs_dir = "logs/skrl/snapfit_lab"
    
    if not os.path.exists(logs_dir):
        print(f"❌ No logs directory found at: {logs_dir}")
        return None
    
    # Find all checkpoint directories
    checkpoints = []
    for run_dir in os.listdir(logs_dir):
        checkpoint_dir = os.path.join(logs_dir, run_dir, "checkpoints")
        if os.path.exists(checkpoint_dir):
            for ckpt_file in os.listdir(checkpoint_dir):
                if ckpt_file.endswith('.pt'):
                    full_path = os.path.join(checkpoint_dir, ckpt_file)
                    checkpoints.append((os.path.getmtime(full_path), full_path))
    
    if not checkpoints:
        print("❌ No checkpoints found!")
        return None
    
    # Return most recent
    checkpoints.sort(reverse=True)
    return checkpoints[0][1]

if __name__ == "__main__":
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
    else:
        print("🔍 Searching for latest checkpoint...")
        checkpoint_path = find_latest_checkpoint()
        
        if checkpoint_path is None:
            print("\n⚠️  No checkpoint found. Start training first!")
            print("   Then run: python scripts/verify_lstm.py")
            sys.exit(1)
    
    check_checkpoint_lstm(checkpoint_path)
