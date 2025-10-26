#!/usr/bin/env python3
"""
Training Monitor Script

Parses training logs and provides real-time insights:
- NaN detection
- Reward trends
- Success rate tracking
- Curriculum stage monitoring
- LSTM health checks

Usage:
    python monitor_training.py --log_file=<path_to_log>
    python monitor_training.py --tensorboard_dir=<path_to_tb_logs>
"""

import argparse
import re
import sys
from pathlib import Path
from collections import deque
import time


class TrainingMonitor:
    def __init__(self, log_file=None, tensorboard_dir=None):
        self.log_file = log_file
        self.tensorboard_dir = tensorboard_dir
        
        # Tracking metrics
        self.step_count = 0
        self.reward_history = deque(maxlen=1000)
        self.success_history = deque(maxlen=1000)
        self.current_stage = 0
        self.lstm_enabled = None
        self.has_nan = False
        
    def parse_log_line(self, line):
        """Parse a single log line and extract metrics."""
        
        # Check for LSTM status
        if "Policy LSTM enabled:" in line:
            self.lstm_enabled = "True" in line
            print(f"✅ LSTM Status: {'Enabled' if self.lstm_enabled else 'DISABLED'}")
        
        # Check for curriculum stage
        stage_match = re.search(r"CURRICULUM STAGE (\d+)/(\d+)", line)
        if stage_match:
            self.current_stage = int(stage_match.group(1))
            print(f"\n🎯 Curriculum Stage: {stage_match.group(1)}/{stage_match.group(2)}")
            
            # Extract entropy if present
            entropy_match = re.search(r"Target Entropy: ([\d.]+)", line)
            if entropy_match:
                print(f"   Entropy: {entropy_match.group(1)}")
        
        # Check for NaN
        if "nan" in line.lower() or "NaN" in line:
            if not self.has_nan:
                print(f"\n❌ WARNING: NaN detected at step {self.step_count}")
                self.has_nan = True
        
        # Extract step count
        step_match = re.search(r"Step (\d+)", line)
        if step_match:
            self.step_count = int(step_match.group(1))
        
        # Extract rewards
        reward_match = re.search(r"Total Reward[:\s]+([-\d.]+)", line)
        if reward_match:
            reward = float(reward_match.group(1))
            self.reward_history.append(reward)
        
        # Extract success rate
        success_match = re.search(r"Success Rate[:\s]+([\d.]+)", line)
        if success_match:
            success = float(success_match.group(1))
            self.success_history.append(success)
    
    def print_summary(self):
        """Print current training summary."""
        print("\n" + "="*80)
        print("TRAINING SUMMARY")
        print("="*80)
        print(f"Steps: {self.step_count:,}")
        print(f"Curriculum Stage: {self.current_stage}")
        print(f"LSTM Enabled: {self.lstm_enabled}")
        
        if self.reward_history:
            avg_reward = sum(self.reward_history) / len(self.reward_history)
            print(f"Avg Reward (last 1000): {avg_reward:.2f}")
        
        if self.success_history:
            avg_success = sum(self.success_history) / len(self.success_history)
            print(f"Avg Success Rate (last 1000): {avg_success:.1%}")
        
        if self.has_nan:
            print(f"⚠️  NaN DETECTED - Training may be unstable!")
        
        print("="*80)
    
    def monitor_file(self):
        """Monitor a log file in real-time."""
        if not self.log_file or not Path(self.log_file).exists():
            print(f"❌ Log file not found: {self.log_file}")
            return
        
        print(f"📊 Monitoring: {self.log_file}")
        print("Press Ctrl+C to stop...\n")
        
        with open(self.log_file, 'r') as f:
            # Read existing lines
            for line in f:
                self.parse_log_line(line)
            
            # Follow new lines
            while True:
                line = f.readline()
                if line:
                    self.parse_log_line(line)
                else:
                    time.sleep(0.1)
    
    def analyze_tensorboard(self):
        """Analyze TensorBoard logs."""
        try:
            from tensorboard.backend.event_processing import event_accumulator
        except ImportError:
            print("❌ TensorBoard not installed. Install with: pip install tensorboard")
            return
        
        if not self.tensorboard_dir:
            print("❌ No TensorBoard directory specified")
            return
        
        print(f"📊 Analyzing TensorBoard logs: {self.tensorboard_dir}")
        
        # Find event files
        event_files = list(Path(self.tensorboard_dir).rglob("events.out.tfevents.*"))
        if not event_files:
            print("❌ No TensorBoard event files found")
            return
        
        for event_file in event_files:
            print(f"\n📁 Processing: {event_file.name}")
            ea = event_accumulator.EventAccumulator(str(event_file))
            ea.Reload()
            
            # List available tags
            print(f"   Tags: {', '.join(ea.Tags()['scalars'][:5])}...")
            
            # Extract key metrics
            if 'Reward/mean' in ea.Tags()['scalars']:
                rewards = ea.Scalars('Reward/mean')
                print(f"   Total reward events: {len(rewards)}")
                if rewards:
                    latest = rewards[-1]
                    print(f"   Latest reward: {latest.value:.2f} at step {latest.step}")


def main():
    parser = argparse.ArgumentParser(description="Monitor PPO training progress")
    parser.add_argument("--log_file", type=str, help="Path to training log file")
    parser.add_argument("--tensorboard_dir", type=str, help="Path to TensorBoard logs")
    parser.add_argument("--summary", action="store_true", help="Print summary and exit")
    
    args = parser.parse_args()
    
    if not args.log_file and not args.tensorboard_dir:
        print("❌ Please specify either --log_file or --tensorboard_dir")
        parser.print_help()
        return
    
    monitor = TrainingMonitor(log_file=args.log_file, tensorboard_dir=args.tensorboard_dir)
    
    try:
        if args.tensorboard_dir:
            monitor.analyze_tensorboard()
        elif args.log_file:
            monitor.monitor_file()
        
        if args.summary:
            monitor.print_summary()
    
    except KeyboardInterrupt:
        print("\n\n⏹️  Monitoring stopped by user")
        monitor.print_summary()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
