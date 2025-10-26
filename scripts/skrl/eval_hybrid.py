"""
Evaluate hybrid RL+Scripted policy.
Test different blend ratios to find optimal balance.
"""

import argparse
import torch
from datetime import datetime

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate hybrid policy")
parser.add_argument("--policy_file", type=str, required=True, help="Trained policy checkpoint")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--num_episodes", type=int, default=100)
parser.add_argument("--mode", type=str, default="BLEND", choices=["BLEND", "FALLBACK", "STAGED"])
parser.add_argument("--alpha", type=float, default=0.7, help="RL weight for BLEND mode")
parser.add_argument("--confidence_threshold", type=float, default=0.3, help="Threshold for FALLBACK mode")
parser.add_argument("--headless", action="store_true")
parser.add_argument("--device", type=str, default="cuda:0")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from snapfit_lab.tasks.direct.snapfit_lab import SnapfitLabEnv, SnapfitLabEnvCfg
from snapfit_lab.tasks.direct.snapfit_lab.hybrid_controller import HybridController


def load_policy(checkpoint_path, device):
    """Load trained policy from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract policy model
    if 'policy' in checkpoint:
        policy = checkpoint['policy']
    elif 'model_state_dict' in checkpoint:
        # BC-style checkpoint
        from train_bc import BCPolicy
        policy = BCPolicy(
            obs_dim=checkpoint['obs_dim'],
            action_dim=checkpoint['action_dim']
        )
        policy.load_state_dict(checkpoint['model_state_dict'])
    else:
        raise ValueError("Unknown checkpoint format")
    
    policy.eval()
    return policy


def main():
    """Evaluate hybrid policy with different configurations"""
    
    print(f"\n{'='*80}")
    print(f"🔄 HYBRID POLICY EVALUATION")
    print(f"{'='*80}")
    print(f"✅ Policy: {args_cli.policy_file}")
    print(f"✅ Mode: {args_cli.mode}")
    if args_cli.mode == "BLEND":
        print(f"✅ Alpha: {args_cli.alpha:.2f} (RL weight)")
    elif args_cli.mode == "FALLBACK":
        print(f"✅ Confidence threshold: {args_cli.confidence_threshold:.2f}")
    print(f"✅ Environments: {args_cli.num_envs}")
    print(f"✅ Episodes: {args_cli.num_episodes}")
    print(f"{'='*80}\n")
    
    # Load policy
    print("📂 Loading policy...")
    policy = load_policy(args_cli.policy_file, args_cli.device)
    print("✅ Policy loaded\n")
    
    # Create environment
    env_cfg = SnapfitLabEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env = SnapfitLabEnv(cfg=env_cfg)
    
    # Create hybrid controller
    controller = HybridController(
        env=env,
        rl_policy=policy,
        mode=args_cli.mode,
        blend_alpha=args_cli.alpha,
        confidence_threshold=args_cli.confidence_threshold,
    )
    
    # Reset
    obs, _ = env.reset()
    controller.reset()
    
    # Tracking
    episodes_completed = 0
    successes = 0
    episode_rewards = []
    episode_lengths = []
    
    current_rewards = torch.zeros(args_cli.num_envs, device=args_cli.device)
    current_lengths = torch.zeros(args_cli.num_envs, dtype=torch.long, device=args_cli.device)
    
    print("🚀 Running evaluation...\n")
    
    while episodes_completed < args_cli.num_episodes:
        # Get hybrid action
        actions = controller.get_action(obs)
        
        # Step
        obs, rewards, dones, truncated, info = env.step(actions)
        
        current_rewards += rewards
        current_lengths += 1
        
        # Process completed episodes
        if dones.any():
            done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
            
            for done_id in done_ids:
                if episodes_completed < args_cli.num_episodes:
                    ep_reward = current_rewards[done_id].item()
                    ep_length = current_lengths[done_id].item()
                    
                    episode_rewards.append(ep_reward)
                    episode_lengths.append(ep_length)
                    
                    if ep_reward > 100:  # Success threshold
                        successes += 1
                    
                    episodes_completed += 1
                    
                    current_rewards[done_id] = 0
                    current_lengths[done_id] = 0
                    
                    # Progress update
                    if episodes_completed % 10 == 0:
                        success_rate = successes / episodes_completed * 100
                        avg_reward = sum(episode_rewards) / len(episode_rewards)
                        stats = controller.get_stats()
                        
                        print(f"Episodes: {episodes_completed}/{args_cli.num_episodes} | "
                              f"Success: {success_rate:.1f}% | "
                              f"Avg Reward: {avg_reward:.1f} | "
                              f"RL Usage: {stats['rl_percentage']:.1f}%")
            
            # Reset controller for done envs
            controller.reset(done_ids)
    
    # Final statistics
    stats = controller.get_stats()
    
    print(f"\n{'='*80}")
    print(f"🏁 EVALUATION RESULTS")
    print(f"{'='*80}")
    print(f"Mode: {args_cli.mode}")
    if args_cli.mode == "BLEND":
        print(f"RL Weight: {args_cli.alpha:.2f}")
    print(f"\n📊 Performance:")
    print(f"  Success Rate: {successes}/{args_cli.num_episodes} ({successes/args_cli.num_episodes*100:.1f}%)")
    print(f"  Avg Reward: {sum(episode_rewards)/len(episode_rewards):.2f} ± {torch.tensor(episode_rewards).std().item():.2f}")
    print(f"  Avg Length: {sum(episode_lengths)/len(episode_lengths):.1f} steps")
    print(f"\n🔄 Controller Usage:")
    print(f"  RL: {stats['rl_percentage']:.1f}%")
    print(f"  Scripted: {stats['scripted_percentage']:.1f}%")
    if args_cli.mode == "FALLBACK":
        print(f"  Avg Confidence: {stats['avg_confidence']:.3f}")
    print(f"{'='*80}\n")
    
    # Cleanup
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
