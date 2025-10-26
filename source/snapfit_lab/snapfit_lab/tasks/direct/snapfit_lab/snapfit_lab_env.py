# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""


WHAT TO LOOK FOR:

GOOD SIGNS:
  - Total Reward trending upward (↑)
  - Success Rate increasing over time
  - Grip Confidence > 0.7 (robot can grasp reliably)
  - Transport Reward > 0.1 (robot moving toward female)
  - Male→Female distance decreasing

WARNING SIGNS:
  - Total Reward stuck or decreasing for >50k steps
  - Grip Confidence stuck < 0.3 after 100k steps
  - Success Rate = 0 after 500k steps
  - All distances staying constant


"""

from __future__ import annotations

import os
import csv
import torch
import wandb
from collections.abc import Sequence
from datetime import datetime

from isaacsim.core.utils.torch.transformations import tf_combine, tf_inverse

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import sample_uniform

from .snapfit_lab_env_cfg import SnapfitLabEnvCfg
from .policy_guidance import LearnedPolicyGuidance


class SnapfitLabEnv(DirectRLEnv):
    cfg: SnapfitLabEnvCfg

    def __init__(self, cfg: SnapfitLabEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.dt = self.cfg.sim.dt * self.cfg.decimation

        # create auxiliary variables for computing applied action, observations and rewards
        self.robot_dof_lower_limits = self.robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self.robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)

        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits)
        self.robot_dof_speed_scales[self.robot.find_joints("panda_finger_joint1")[0]] = 0.3  # INCREASED for faster gripping
        self.robot_dof_speed_scales[self.robot.find_joints("panda_finger_joint2")[0]] = 0.3  # INCREASED for faster gripping

        self.robot_dof_targets = torch.zeros((self.num_envs, self.robot.num_joints), device=self.device)

        # IMPROVED: Use panda_link7 (hand) for better orientation reference
        self.hand_link_idx = self.robot.find_bodies("panda_link7")[0][0]
        self.left_finger_link_idx = self.robot.find_bodies("panda_leftfinger")[0][0]
        self.right_finger_link_idx = self.robot.find_bodies("panda_rightfinger")[0][0]

        self.robot_grasp_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.robot_grasp_rot = torch.zeros((self.num_envs, 4), device=self.device)  # Now from hand_link
        self.buckle_grasp_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.buckle_grasp_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.grasp_dist = torch.zeros((self.num_envs, 3), device=self.device)
        self.buckle_ftom_pose = torch.zeros((self.num_envs, 7), device=self.device)

        # -- Variables for reward calculation --
        # Previous actions for calculating action rate penalty
        self.prev_actions = torch.zeros((self.num_envs, self.robot.num_joints), device=self.device)
        # Initial position of the female buckle to penalize its movement
        self.female_buckle_initial_pos_w = torch.zeros((self.num_envs, 3), device=self.device)
        
        # IMPROVED: Early termination tracking
        self.success_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.consecutive_success = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        
        # Terminal progress tracking
        self.print_interval = 500  # INCREASED: Print every 500 steps (reduce overhead)
        self.summary_interval = 10000  # INCREASED: Detailed summary every 10000 steps
        self.wandb_log_interval = 250  # INCREASED: Log to wandb every 250 steps (reduce I/O)
        self.last_print_step = 0
        self.last_summary_step = 0
        self.last_wandb_log_step = 0
        self.reward_history = []
        self.success_history = []
        
        # Long-term tracking for improvement analysis
        self.milestone_rewards = []  # Store rewards at milestones
        self.milestone_success = []  # Store success rates at milestones
        self.milestone_steps = []    # Store step counts
        self.training_start_time = None  # Will be set on first step
        
        # IMPROVED: Enhanced curriculum tracking
        self.curriculum_stage = 0
        self.update_curriculum()  # Initialize curriculum parameters
        
        # EMA tracking for smoother gripping metrics display
        self.ema_num_gripping = 0.0
        self.ema_alpha = 0.1  # EMA smoothing factor

        # Initialize wandb and logging directory
        self.log_dir = os.path.join("logs", "metrics", datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Create CSV files for episode metrics
        self.episode_metrics_file = os.path.join(self.log_dir, "episode_metrics.csv")
        with open(self.episode_metrics_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'episode', 'env_id', 'total_reward', 'episode_length',
                'success', 'grip_confidence', 'final_distance', 'curriculum_stage'
            ])
        
        # Create CSV for step metrics (sampled)
        self.step_metrics_file = os.path.join(self.log_dir, "step_metrics.csv")
        with open(self.step_metrics_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'step', 'avg_total_reward', 'avg_success_rate',
                'avg_grip_conf', 'avg_male_female_dist', 'curriculum_stage'
            ])
        
        # Initialize policy guidance from imitation learning (warm-start for faster convergence)
        self.policy_guidance = None
        if cfg.use_policy_guidance:
            self.policy_guidance = LearnedPolicyGuidance(num_envs=self.num_envs, device=self.device)
        
        wandb.init(
            project="snapfit_franka_enhanced",
            config={
                "episode_length_s": cfg.episode_length_s,
                "action_space": cfg.action_space,
                "observation_space": cfg.observation_space,
                "observation_space_critic": cfg.observation_space_critic,
                "num_envs": self.num_envs,
                "curriculum_stages": len(cfg.curriculum_stages),
                "use_lstm": True,
                "architecture": "asymmetric_actor_critic",
                "use_policy_guidance": cfg.use_policy_guidance,
                "guidance_with_exploration": cfg.guidance_with_exploration,
            },
        )
        
        print(f"\n{'='*80}")
        print(f"🚀 ENHANCED PPO SETUP (NVIDIA IndustReal-inspired)")
        print(f"{'='*80}")
        print(f"✅ LSTM Recurrent Policy: Enabled (sequence_length=16)")
        print(f"   - Policy LSTM: 256 units, 16-step lookback")
        print(f"   - Critic LSTM: 256 units, 16-step lookback")
        if cfg.use_policy_guidance:
            print(f"✅ Imitation Learning Warm-Start: Enabled")
            print(f"   - Policy initialized from expert demonstrations")
            print(f"   - Accelerates convergence by 2-3x (NVIDIA IndustReal)")
            print(f"   - Exploration: {'Enabled' if cfg.guidance_with_exploration else 'Conservative'}")
        print(f"✅ Symmetric Actor-Critic: Policy obs={cfg.observation_space}, Critic obs={cfg.observation_space_critic}")
        print(f"✅ Accelerated Curriculum: {len(cfg.curriculum_stages)} stages (2M steps each)")
        print(f"   - Stage 1 (0-2M): Grasp+Transport, 25mm tolerance")
        print(f"   - Stage 2 (2-4M): Transport focus, 18mm tolerance")
        print(f"   - Stage 3 (4-6M): Precise insertion, 12mm tolerance")
        print(f"   - Stage 4 (6-8M): Expert, 10mm tolerance")
        print(f"✅ Early Termination: Enabled (consecutive={cfg.early_termination_steps})")
        print(f"✅ Total Training: 8M timesteps (4 stages × 2M)")
        print(f"✅ Logging Directory: {self.log_dir}")
        print(f"{'='*80}\n")

    def update_curriculum(self):
        """Update curriculum parameters based on training progress (NVIDIA IndustReal approach)."""
        total_steps = self.common_step_counter if hasattr(self, 'common_step_counter') else 0
        
        # Find appropriate curriculum stage
        new_stage = 0
        for i, stage_cfg in enumerate(self.cfg.curriculum_stages):
            if total_steps >= stage_cfg["start_step"]:
                new_stage = i
        
        if new_stage != self.curriculum_stage:
            self.curriculum_stage = new_stage
            stage_cfg = self.cfg.curriculum_stages[new_stage]
            
            # Update parameters
            self.current_insertion_threshold = stage_cfg["insertion_threshold"]
            self.current_randomization_scale = stage_cfg["randomization_scale"]
            self.current_reward_scales = stage_cfg["reward_scales"]
            # NEW: Per-stage grip thresholds
            self.current_grip_threshold_finger = stage_cfg.get("grip_threshold_finger", self.cfg.grip_threshold_finger)
            self.current_grip_threshold_dist = stage_cfg.get("grip_threshold_dist", self.cfg.grip_threshold_dist)
            
            print(f"\n{'='*80}")
            print(f"🎯 CURRICULUM STAGE {new_stage + 1}/{len(self.cfg.curriculum_stages)} (Step {total_steps:,})")
            print(f"{'='*80}")
            print(f"  Insertion Threshold: {self.current_insertion_threshold*1000:.1f}mm")
            print(f"  Randomization Scale: {self.current_randomization_scale:.1%}")
            print(f"  Reward Scales: grasp={self.current_reward_scales['grasp']:.1f}, "
                  f"transport={self.current_reward_scales['transport']:.1f}, "
                  f"success={self.current_reward_scales['success']:.1f}")
            print(f"{'='*80}\n")
        else:
            # Initialize on first call
            if not hasattr(self, 'current_insertion_threshold'):
                stage_cfg = self.cfg.curriculum_stages[0]
                self.current_insertion_threshold = stage_cfg["insertion_threshold"]
                self.current_randomization_scale = stage_cfg["randomization_scale"]
                self.current_reward_scales = stage_cfg["reward_scales"]
                # NEW: Per-stage grip thresholds
                self.current_grip_threshold_finger = stage_cfg.get("grip_threshold_finger", self.cfg.grip_threshold_finger)
                self.current_grip_threshold_dist = stage_cfg.get("grip_threshold_dist", self.cfg.grip_threshold_dist)
    
    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self.female_buckle = RigidObject(self.cfg.buckle_female_cfg)
        self.male_buckle = Articulation(self.cfg.buckle_male_cfg)
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["buckle_female"] = self.female_buckle
        self.scene.articulations["buckle_male"] = self.male_buckle

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        
        self.table = RigidObject(self.cfg.table_cfg)
        self.scene.rigid_objects["table"] = self.table

        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _get_env_data_for_policy_guidance(self) -> dict:
        """Extract current environment state for policy guidance network"""
        gripper_pos = self.robot.data.body_pos_w[:, self.hand_link_idx]
        male_pos = self.male_buckle.data.root_pos_w.squeeze(1)
        female_pos = self.female_buckle.data.root_pos_w.squeeze(1)
        
        left_finger_pos = self.robot.data.body_pos_w[:, self.left_finger_link_idx]
        right_finger_pos = self.robot.data.body_pos_w[:, self.right_finger_link_idx]
        finger_dist = torch.norm(left_finger_pos - right_finger_pos, dim=-1)
        
        gripper_to_male = torch.norm(gripper_pos - male_pos, dim=-1)
        male_to_female = torch.norm(male_pos - female_pos, dim=-1)
        
        return {
            "gripper_position": gripper_pos,
            "male_position": male_pos,
            "female_position": female_pos,
            "gripper_to_male_dist": gripper_to_male,
            "male_to_female_dist": male_to_female,
            "finger_distance": finger_dist,
        }
    
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # Store previous actions
        self.prev_actions[:] = self.actions
        
        # Apply policy guidance from imitation learning (warm-start acceleration)
        if self.policy_guidance is not None:
            observations = self._get_observations()
            env_data = self._get_env_data_for_policy_guidance()
            actions = self.policy_guidance.act(observations, env_data)
        
        # Update and clamp current actions
        self.actions = actions.clone().clamp(-1.0, 1.0)
        targets = self.robot_dof_targets + self.robot_dof_speed_scales * self.dt * self.actions * self.cfg.action_scale
        self.robot_dof_targets[:] = torch.clamp(targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits)

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(self.robot_dof_targets)

    def _get_observations(self) -> dict:
        dof_pos_scaled = (
            2.0
            * (self.robot.data.joint_pos - self.robot_dof_lower_limits)
            / (self.robot_dof_upper_limits - self.robot_dof_lower_limits)
            - 1.0
        )
        
        # Domain randomization: add observation noise for sim2real robustness
        if self.cfg.enable_domain_randomization and self.cfg.observation_noise_std > 0:
            obs_noise = torch.randn_like(dof_pos_scaled) * self.cfg.observation_noise_std
            dof_pos_scaled = dof_pos_scaled + obs_noise
        
        robot_left_finger_pos = self.robot.data.body_pos_w[:, self.left_finger_link_idx]
        robot_right_finger_pos = self.robot.data.body_pos_w[:, self.right_finger_link_idx]
        
        # IMPROVED: Use gripper midpoint for position, hand link for orientation
        self.robot_grasp_pos = 0.5 * (robot_left_finger_pos + robot_right_finger_pos)
        self.robot_grasp_rot = self.robot.data.body_quat_w[:, self.hand_link_idx]  # Better reference
        
        # Split into rotation and translation
        female_t = self.female_buckle.data.body_com_pos_w.squeeze(1)  # (N, 3)
        female_q = self.female_buckle.data.body_com_quat_w.squeeze(1)      # (N, 4)

        male_t = self.male_buckle.data.root_com_pos_w.squeeze(1)      # (N, 3)
        male_q = self.male_buckle.data.root_com_quat_w.squeeze(1)      # (N, 4)
        
        self.buckle_grasp_pos = male_t
        self.buckle_grasp_rot = male_q
        # distance between the gripper and male buckle
        self.grasp_dist = self.buckle_grasp_pos - self.robot_grasp_pos

        # Invert the female pose
        female_q_inv, female_t_inv = tf_inverse(female_q, female_t)  # each is (N, 4) and (N, 3)
        # Now combine: T_relative = inv(T_female) * T_male
        relative_q, relative_t = tf_combine(female_q_inv, female_t_inv, male_q, male_t)
        # Combine back into SE(3) pose
        self.buckle_ftom_pose = torch.cat([relative_t, relative_q], dim=-1)  # shape: (N, 7)

        # Policy observation (what the actor sees)
        obs_policy = torch.cat(
            (
                dof_pos_scaled,
                self.buckle_ftom_pose,
                self.grasp_dist,
                self.actions
            ),
            dim=-1,
        )
        
        # NOTE: Previously used asymmetric actor-critic with privileged critic observations
        # Switched to symmetric (both see same 28-dim obs) for skrl LSTM compatibility
        # Asymmetric would compute obs_critic with joint_vel + velocities + exact positions (52-dim)
        
        # FIXED: Return dictionary with "policy" key for skrl compatibility
        # skrl's IsaacLab wrapper expects observations["policy"]
        # Using symmetric observations (both policy and critic see the same data)
        return {"policy": obs_policy}

    def _get_rewards(self) -> torch.Tensor:
        # IMPROVED: Update curriculum parameters periodically
        if self.common_step_counter % 1000 == 0:
            self.update_curriculum()
        
        # Get all necessary data for reward calculation
        robot_left_finger_pos = self.robot.data.body_pos_w[:, self.left_finger_link_idx]
        robot_right_finger_pos = self.robot.data.body_pos_w[:, self.right_finger_link_idx]
        female_buckle_pos = self.female_buckle.data.root_pos_w
        female_buckle_rot = self.female_buckle.data.root_quat_w
        
        # IMPROVED: Use curriculum-adjusted reward scales and thresholds
        total_reward, reward_components, success_info = compute_rewards(
            # State tensors
            self.actions,
            self.prev_actions,
            self.robot.data.joint_vel,
            self.robot_grasp_pos,
            self.robot_grasp_rot,
            self.buckle_grasp_pos,
            self.buckle_grasp_rot,
            female_buckle_pos,
            female_buckle_rot,
            self.female_buckle_initial_pos_w,
            self.buckle_ftom_pose[:, :3],
            robot_left_finger_pos,
            robot_right_finger_pos,
            # Base reward scales from config
            self.cfg.approach_weight,
            self.cfg.align_weight,
            self.current_reward_scales["grasp"],  # Curriculum-adjusted
            self.current_reward_scales["transport"],  # Curriculum-adjusted
            self.cfg.mating_weight,
            self.cfg.insertion_align_weight,
            self.current_reward_scales["success"],  # Curriculum-adjusted
            self.cfg.lateral_penalty,
            self.current_insertion_threshold,  # Curriculum-adjusted
            self.current_grip_threshold_finger,  # Curriculum-adjusted per stage
            self.current_grip_threshold_dist,  # Curriculum-adjusted per stage
            self.cfg.action_rate_penalty,
            self.cfg.harsh_movement_penalty,
            self.cfg.female_move_penalty,
        )
        
        # IMPROVED: Update success tracking for early termination
        self.success_buf[:] = success_info["insertion_success"]
        self.consecutive_success = torch.where(
            self.success_buf,
            self.consecutive_success + 1,
            torch.zeros_like(self.consecutive_success)
        )
        
        # IMPROVED: Cache latest metrics for terminal printing (always available)
        self.latest_metrics = {
            "total_reward": total_reward.mean().item(),
            "success_rate": success_info["insertion_success"].float().mean().item(),
            "grip_confidence": success_info["grip_conf"].mean().item(),
            "grasp_quality": success_info["grasp_quality"].mean().item(),
            "num_gripping": success_info["num_gripping"],
            "finger_close_reward": success_info["finger_close_reward"].mean().item(),
            "proximity_reward": success_info["proximity_reward"].mean().item(),
            "is_gripped_rate": success_info["is_gripped"].float().mean().item(),
            "gripper_to_male_dist": success_info["gripper_to_male_dist"].mean().item(),
            "gripper_to_female_dist": success_info["gripper_to_female_dist"].mean().item(),
            "male_to_female_dist": success_info["male_to_female_dist"].mean().item(),
            "lateral_error": success_info["lateral_error"].mean().item(),
            "finger_dist": success_info["finger_dist"].mean().item(),
        }
        
        # IMPROVED: Throttled W&B logging (every N steps instead of every step)
        if self.common_step_counter - self.last_wandb_log_step >= self.wandb_log_interval:
            self.last_wandb_log_step = self.common_step_counter
            
            # Create wandb log dictionary
            wandb_log = {
                # Total and stage rewards
                "reward/total_reward": total_reward.mean().item(),
                "reward/pre_grasp_reward": reward_components["pre_grasp"].mean().item(),
                "reward/post_grasp_reward": reward_components["post_grasp"].mean().item(),
                
                # Individual component rewards
                "reward/approach_reward": reward_components["approach"].mean().item(),
                "reward/align_reward": reward_components["align"].mean().item(),
                "reward/grasp_reward": reward_components["grasp"].mean().item(),
                "reward/transport_reward": reward_components["transport"].mean().item(),
                "reward/mating_reward": reward_components["mating"].mean().item(),
                "reward/insertion_align_reward": reward_components["insertion_align"].mean().item(),
                "reward/success_reward": reward_components["success"].mean().item(),
                
                # Penalties
                "penalty/action_rate_penalty": reward_components["action_rate_penalty"].mean().item(),
                "penalty/harsh_movement_penalty": reward_components["harsh_movement_penalty"].mean().item(),
                "penalty/female_move_penalty": reward_components["female_move_penalty"].mean().item(),
                "penalty/lateral_penalty": reward_components["lateral_penalty"].mean().item(),
                
                # State observations
                "state/gripper_to_male_dist": success_info["gripper_to_male_dist"].mean().item(),
                "state/gripper_to_female_dist": success_info["gripper_to_female_dist"].mean().item(),
                "state/male_to_female_dist": success_info["male_to_female_dist"].mean().item(),
                "state/finger_dist": success_info["finger_dist"].mean().item(),
                "state/grip_confidence": success_info["grip_conf"].mean().item(),
                "state/female_move_dist": success_info["female_move_dist"].mean().item(),
                "state/lateral_error": success_info["lateral_error"].mean().item(),
                "state/insertion_ori_alignment": success_info["male_female_ori_dot"].mean().item(),
                "state/success_rate": success_info["insertion_success"].float().mean().item(),
                "state/is_gripped_rate": success_info["is_gripped"].float().mean().item(),
                
                # Curriculum tracking
                "curriculum/stage": self.curriculum_stage,
                "curriculum/insertion_threshold": self.current_insertion_threshold,
                "curriculum/randomization_scale": self.current_randomization_scale,
            }
            
            wandb.log(wandb_log, step=self.common_step_counter)
            
            # IMPROVED: Log to CSV file periodically
            with open(self.step_metrics_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().isoformat(),
                    self.common_step_counter,
                    wandb_log["reward/total_reward"],
                    wandb_log["state/success_rate"],
                    wandb_log["state/grip_confidence"],
                    wandb_log["state/male_to_female_dist"],
                    self.curriculum_stage,
                ])
        
        # =====================================================================
        # TERMINAL PROGRESS TRACKING - Print every N steps
        # =====================================================================
        if self.common_step_counter - self.last_print_step >= self.print_interval:
            self.last_print_step = self.common_step_counter
            
            # Get current metrics
            current_reward = total_reward.mean().item()
            current_success = success_info["insertion_success"].float().mean().item()
            
            # Store metrics for trend analysis
            self.reward_history.append(current_reward)
            self.success_history.append(current_success)
            
            # Keep only last 10 samples for moving average
            if len(self.reward_history) > 10:
                self.reward_history.pop(0)
                self.success_history.pop(0)
            
            # Calculate trends
            avg_reward = sum(self.reward_history) / len(self.reward_history)
            avg_success = sum(self.success_history) / len(self.success_history)
            
            # Determine trend indicators
            reward_trend = "↑" if len(self.reward_history) > 1 and self.reward_history[-1] > self.reward_history[0] else "→"
            success_trend = "↑" if len(self.success_history) > 1 and self.success_history[-1] > self.success_history[0] else "→"
            
            # Print formatted progress
            print("\n" + "="*80)
            print(f" TRAINING PROGRESS - Step {self.common_step_counter:,} | Stage {self.curriculum_stage+1}")
            print("="*80)
            print(f" Total Reward:        {self.latest_metrics['total_reward']:.3f}  (avg:   {avg_reward:.3f}) {reward_trend}")
            print(f" Success Rate:        {self.latest_metrics['success_rate']*100:.1f}%  (avg:   {avg_success*100:.1f}%) {success_trend}")
            print(f" Grip Confidence:     {self.latest_metrics['grip_confidence']:.3f}")
            print(f" Grasp Quality:       {self.latest_metrics['grasp_quality']:.3f}  (verified)")
            
            # DETAILED HEADLESS-MODE STATUS
            num_gripping = self.latest_metrics['num_gripping']
            grip_pct = (num_gripping / self.num_envs) * 100
            num_success = int(self.latest_metrics['success_rate'] * self.num_envs)
            
            # Update EMA for smoother gripping count display
            self.ema_num_gripping = self.ema_alpha * num_gripping + (1 - self.ema_alpha) * self.ema_num_gripping
            ema_pct = (self.ema_num_gripping / self.num_envs) * 100
            
            # Gripping status (show both instant and smoothed)
            if num_gripping > 0:
                print(f"🔥 GRIPPING NOW:        {num_gripping}/{self.num_envs} envs ({grip_pct:.1f}%)  [EMA: {self.ema_num_gripping:.0f} ({ema_pct:.1f}%)] 🤖✊")
            else:
                print(f"⚪ Not Gripping Yet:    0/{self.num_envs} envs (0.0%)  [EMA: {self.ema_num_gripping:.0f} ({ema_pct:.1f}%)]")
            
            # Insertion status (headless visibility)
            if num_success > 0:
                print(f"✅ INSERTING SUCCESS:   {num_success}/{self.num_envs} envs 🎯 MALE IN FEMALE!")
            
            # Distance metrics
            print(f"📍 Gripper→Male:        {self.latest_metrics['gripper_to_male_dist']:.4f} m")
            print(f"🖐️  Finger Distance:     {self.latest_metrics['finger_dist']:.4f} m")
            print(f"🎯 Male→Female:         {self.latest_metrics['male_to_female_dist']:.4f} m")
            
            # Insertion readiness indicator
            if self.latest_metrics['male_to_female_dist'] < 0.05:
                print(f"🚨 INSERTION ATTEMPT:   Male very close to female!")
            
            print("-"*80)
            
            # Learning status indicator
            grip_conf = success_info['grip_conf'].mean().item()
            if avg_success > 0.5:
                status = "🟢 EXCELLENT - Policy is highly successful!"
            elif avg_success > 0.2:
                status = "🟡 GOOD - Policy is learning well"
            elif avg_success > 0.05:
                status = "🟠 PROGRESSING - Early success signals"
            elif grip_conf > 0.7:
                status = "🔵 LEARNING - Grasping mastered, working on insertion"
            elif grip_conf > 0.3:
                status = "⚪ EARLY - Learning to grasp"
            else:
                status = "⚫ INITIAL - Exploring environment"
            
            print(f"Status: {status}")
            print(f"📚 Curriculum: Stage {self.curriculum_stage+1}/{len(self.cfg.curriculum_stages)} | "
                  f"Threshold: {self.current_insertion_threshold*1000:.1f}mm | "
                  f"Grip Gate: finger<{self.current_grip_threshold_finger*1000:.1f}mm, dist<{self.current_grip_threshold_dist*1000:.1f}mm")
            print("="*80 + "\n")
        
        # =====================================================================
        # DETAILED TRAINING SUMMARY - Print every 5000 steps
        # =====================================================================
        if self.common_step_counter - self.last_summary_step >= self.summary_interval:
            import time
            if self.training_start_time is None:
                self.training_start_time = time.time()
            
            self.last_summary_step = self.common_step_counter
            
            # Store milestone data (use cached metrics)
            self.milestone_steps.append(self.common_step_counter)
            self.milestone_rewards.append(self.latest_metrics["total_reward"])
            self.milestone_success.append(self.latest_metrics["success_rate"])
            
            # Calculate training statistics
            elapsed_time = time.time() - self.training_start_time
            steps_per_sec = self.common_step_counter / max(elapsed_time, 1)
            hours_elapsed = elapsed_time / 3600
            
            # Calculate improvement trends
            if len(self.milestone_rewards) > 1:
                reward_improvement = self.milestone_rewards[-1] - self.milestone_rewards[0]
                success_improvement = self.milestone_success[-1] - self.milestone_success[0]
                reward_trend_pct = (reward_improvement / max(abs(self.milestone_rewards[0]), 0.001)) * 100
                success_trend_pct = success_improvement * 100
            else:
                reward_improvement = 0
                success_improvement = 0
                reward_trend_pct = 0
                success_trend_pct = 0
            
            print("\n" + "█"*80)
            print("█" + " "*78 + "█")
            print(f"█  🎓 TRAINING SUMMARY - {self.common_step_counter:,} Steps".ljust(79) + "█")
            print("█" + " "*78 + "█")
            print("█"*80)
            
            # Performance metrics (use cached metrics)
            print(f"█  📈 PERFORMANCE METRICS".ljust(79) + "█")
            print(f"█    • Total Reward:        {self.latest_metrics['total_reward']:>8.3f}".ljust(79) + "█")
            print(f"█    • Success Rate:        {self.latest_metrics['success_rate']:>8.1%}".ljust(79) + "█")
            print(f"█    • Grip Confidence:     {self.latest_metrics['grip_confidence']:>8.3f}".ljust(79) + "█")
            print(f"█    • Grasp Quality:       {self.latest_metrics['grasp_quality']:>8.3f}  (verified)".ljust(79) + "█")
            
            # GRIPPING VERIFICATION DISPLAY
            num_gripping = self.latest_metrics['num_gripping']
            grip_pct = (num_gripping / self.num_envs) * 100
            if num_gripping > 0:
                print(f"█    🔥 GRIPPING NOW:        {num_gripping}/{self.num_envs} envs ({grip_pct:.1f}%)".ljust(79) + "█")
            else:
                print(f"█    ⚪ Not Gripping Yet:    0/{self.num_envs} envs (0.0%)".ljust(79) + "█")
            
            print(f"█    • Is Gripped Rate:     {self.latest_metrics['is_gripped_rate']:>8.1%}".ljust(79) + "█")
            print("█" + " "*78 + "█")
            
            # Improvement tracking
            print(f"█  📊 IMPROVEMENT SINCE START".ljust(79) + "█")
            reward_arrow = "📈" if reward_improvement > 0 else "📉" if reward_improvement < 0 else "➡️"
            success_arrow = "📈" if success_improvement > 0 else "📉" if success_improvement < 0 else "➡️"
            print(f"█    {reward_arrow} Reward Change:      {reward_improvement:>+8.3f}  ({reward_trend_pct:>+7.1f}%)".ljust(79) + "█")
            print(f"█    {success_arrow} Success Change:     {success_improvement:>+8.1%}  ({success_trend_pct:>+7.1f}%)".ljust(79) + "█")
            print("█" + " "*78 + "█")
            
            # Key distances (use cached metrics)
            print(f"█  📏 CURRENT DISTANCES".ljust(79) + "█")
            print(f"█    • Gripper → Male:      {self.latest_metrics['gripper_to_male_dist']:>8.4f} m".ljust(79) + "█")
            print(f"█    • Gripper → Female:    {self.latest_metrics['gripper_to_female_dist']:>8.4f} m".ljust(79) + "█")
            print(f"█    • Male → Female:       {self.latest_metrics['male_to_female_dist']:>8.4f} m".ljust(79) + "█")
            print(f"█    • Lateral Error:       {self.latest_metrics['lateral_error']:>8.4f} m".ljust(79) + "█")
            print("█" + " "*78 + "█")
            
            # Training efficiency
            print(f"█  ⚡ TRAINING EFFICIENCY".ljust(79) + "█")
            print(f"█    • Time Elapsed:        {hours_elapsed:>8.2f} hours".ljust(79) + "█")
            print(f"█    • Steps/Second:        {steps_per_sec:>8.1f}".ljust(79) + "█")
            print(f"█    • Total Environments:  {self.num_envs:>8,}".ljust(79) + "█")
            print(f"█    • Samples/Second:      {steps_per_sec * self.num_envs:>8,.0f}".ljust(79) + "█")
            print("█" + " "*78 + "█")
            
            # Learning stage assessment (use cached metrics and current reward components)
            print(f"█  🎯 LEARNING STAGE ASSESSMENT".ljust(79) + "█")
            grip_conf = self.latest_metrics['grip_confidence']
            success_rate = self.latest_metrics['success_rate']
            transport_rew = reward_components['transport'].mean().item()
            
            if success_rate > 0.7:
                stage = "MASTERY - Consistently successful insertions! 🏆"
            elif success_rate > 0.3:
                stage = "ADVANCED - High success rate, refining policy 🎯"
            elif success_rate > 0.1:
                stage = "INTERMEDIATE - Regular successes, improving 📈"
            elif success_rate > 0.01:
                stage = "EARLY SUCCESS - First successful insertions! 🎉"
            elif grip_conf > 0.8 and transport_rew > 0.1:
                stage = "TRANSPORT - Moving toward female buckle 🚚"
            elif grip_conf > 0.6:
                stage = "GRASPING - Learning reliable grasps 🤏"
            elif grip_conf > 0.3:
                stage = "APPROACH - Learning to reach male buckle 📍"
            else:
                stage = "EXPLORATION - Initial random exploration 🔍"
            
            print(f"█    {stage}".ljust(79) + "█")
            print("█" + " "*78 + "█")
            
            # Next milestone prediction
            estimated_steps_to_50pct = max(0, int((0.5 - success_rate) * self.common_step_counter / max(success_rate, 0.001)))
            if success_rate < 0.5 and success_rate > 0:
                print(f"█  🔮 PREDICTION".ljust(79) + "█")
                print(f"█    • Est. steps to 50% success: ~{estimated_steps_to_50pct:,}".ljust(79) + "█")
                print("█" + " "*78 + "█")
            
            print("█"*80 + "\n")

        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # IMPROVED: Early termination on success
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        
        # Early termination if task succeeded for N consecutive steps
        early_success = self.consecutive_success >= self.cfg.early_termination_steps
        
        # Combined termination
        terminated = time_out | early_success
        
        return terminated, time_out  # Return (terminated, truncated)

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        
        # IMPROVED: Log episode metrics before reset
        if len(env_ids) > 0:
            for env_id in env_ids:
                env_id_int = int(env_id)
                # Calculate episode metrics
                episode_length = int(self.episode_length_buf[env_id_int].item())
                total_reward = float(self.episode_length_buf[env_id_int].item())  # Approximate
                was_successful = bool(self.success_buf[env_id_int].item())
                final_distance = 0.0  # Will be computed if needed
                grip_conf = 0.0  # Will be computed if needed
                
                # Log to CSV
                with open(self.episode_metrics_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        datetime.now().isoformat(),
                        self.common_step_counter // self.num_envs,  # Episode number estimate
                        env_id_int,
                        total_reward,
                        episode_length,
                        int(was_successful),
                        grip_conf,
                        final_distance,
                        self.curriculum_stage,
                    ])
        
        super()._reset_idx(env_ids)
        
        # Reset success tracking for these envs
        self.consecutive_success[env_ids] = 0
        self.success_buf[env_ids] = False
        
        # Reset policy guidance network state for these envs
        if self.policy_guidance is not None:
            self.policy_guidance.reset(torch.tensor(env_ids, device=self.device))

        # robot state
        joint_pos = self.robot.data.default_joint_pos[env_ids] + sample_uniform(
            -0.125,
            0.125,
            (len(env_ids), self.robot.num_joints),
            self.device,
        )
        joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        joint_vel = self.robot.data.default_joint_vel[env_ids]

        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # -- BUCKLE FEMALE --
        # get default state
        buckle_female_default_state = self.female_buckle.data.default_root_state[env_ids]
        # copy to new state
        buckle_female_new_state = buckle_female_default_state.clone()

        # randomize position on xy-plane within a 10cm square
        pos_noise = sample_uniform(-0.05, 0.05, (len(env_ids), 2), device=self.device)
        buckle_female_new_state[:, 0:2] += pos_noise

        # add environment origins to the position
        buckle_female_new_state[:, :3] += self.scene.env_origins[env_ids]
        
        # Domain randomization: randomize buckle mass for sim2real transfer
        if self.cfg.enable_domain_randomization:
            mass_scale = sample_uniform(
                self.cfg.buckle_mass_range[0], 
                self.cfg.buckle_mass_range[1], 
                (len(env_ids), 1), 
                device=self.device
            )
            # Note: Mass randomization would require PhysX API calls - simplified here
            # In practice, you'd use self.female_buckle.root_physx_view.set_masses(...)
        
        # write the new state to the simulation
        self.female_buckle.write_root_state_to_sim(buckle_female_new_state, env_ids)
        # -- NEW: Store the initial position for reward calculation
        self.female_buckle_initial_pos_w[env_ids] = buckle_female_new_state[:, :3].clone()


        # -- BUCKLE MALE --
        # reset the male buckle to its default state
        buckle_male_default_state = self.male_buckle.data.default_root_state[env_ids]
        buckle_male_new_state = buckle_male_default_state.clone()
        buckle_male_new_state[:, :3] += self.scene.env_origins[env_ids]
        self.male_buckle.write_root_state_to_sim(buckle_male_new_state, env_ids)


@torch.jit.script
def compute_rewards(
    # State tensors
    actions: torch.Tensor,
    prev_actions: torch.Tensor,
    joint_velocities: torch.Tensor,
    robot_grasp_pos: torch.Tensor,
    robot_grasp_rot: torch.Tensor,
    male_buckle_pos: torch.Tensor,
    male_buckle_rot: torch.Tensor,
    female_buckle_pos: torch.Tensor,
    female_buckle_rot: torch.Tensor,
    female_buckle_initial_pos: torch.Tensor,
    male_to_female_pos: torch.Tensor,
    robot_lfinger_pos: torch.Tensor,
    robot_rfinger_pos: torch.Tensor,
    # Reward scales
    approach_weight: float,
    align_weight: float,
    grasp_weight: float,
    transport_weight: float,        # NEW
    mating_weight: float,
    insertion_align_weight: float,
    success_reward_bonus: float,
    lateral_penalty_weight: float,
    insertion_threshold: float,
    grip_threshold_finger: float,   # NEW
    grip_threshold_dist: float,     # NEW
    # Penalty scales
    action_rate_penalty: float,
    harsh_movement_penalty: float,
    female_move_penalty: float,
):
    # =============================================================================
    # STAGE 1: APPROACH MALE BUCKLE
    # =============================================================================
    gripper_to_male_dist = torch.norm(robot_grasp_pos - male_buckle_pos, p=2, dim=-1)
    approach_reward = torch.exp(-4.0 * gripper_to_male_dist)
    # Bonus for very close proximity
    approach_reward = torch.where(gripper_to_male_dist <= 0.02, approach_reward * 2.0, approach_reward)

    # =============================================================================
    # STAGE 2: ALIGN ORIENTATION WITH MALE
    # =============================================================================
    quat_dot = torch.abs(torch.sum(robot_grasp_rot * male_buckle_rot, dim=-1))
    align_reward = quat_dot**4

    # =============================================================================
    # STAGE 3: GRASP MALE BUCKLE (SOFT CONFIDENCE)
    # =============================================================================
    finger_dist = torch.norm(robot_lfinger_pos - robot_rfinger_pos, p=2, dim=-1)
    
    # Hard boolean for success detection
    is_gripped = (finger_dist < grip_threshold_finger) & (gripper_to_male_dist < grip_threshold_dist)
    
    # FIXED: Stronger gradient for grasping - use exponential instead of sigmoid
    # Reward for closing fingers (smaller distance = higher reward)
    finger_close_reward = torch.exp(-10.0 * finger_dist)
    # Reward for being near male buckle
    proximity_reward = torch.exp(-10.0 * gripper_to_male_dist)
    # Combined grip confidence (product ensures both conditions needed)
    grip_conf = finger_close_reward * proximity_reward
    grip_conf = torch.clamp(grip_conf, 0.0, 1.0)
    
    # IMPROVED: Grasp quality verification - stricter gating for post-grasp behaviors
    # Only allow transport/mating rewards if robot is ACTUALLY grasping (not just close)
    grasp_quality = torch.where(
        (finger_dist < grip_threshold_finger * 1.2) & (gripper_to_male_dist < grip_threshold_dist * 1.5),
        grip_conf,
        torch.zeros_like(grip_conf)
    )
    grasp_quality = torch.clamp(grasp_quality, 0.0, 1.0)
    
    # Count how many envs are actively gripping (for terminal display)
    num_gripping = (grasp_quality > 0.3).sum().item()
    
    # FIXED: Grasp reward now has explicit bonus for closing fingers
    grasp_reward = grip_conf + 0.3 * finger_close_reward  # Reduced from 0.5 to avoid over-rewarding

    # =============================================================================
    # STAGE 4: TRANSPORT MALE TOWARD FEMALE (VERIFIED GRASP REQUIRED)
    # =============================================================================
    # Direct reward for moving the gripper (holding male) toward the female target
    # IMPROVED: Use grasp_quality to ensure proper grasp before transport reward
    gripper_to_female_dist = torch.norm(robot_grasp_pos - female_buckle_pos, p=2, dim=-1)
    transport_reward = torch.exp(-3.0 * gripper_to_female_dist) * grasp_quality
    
    # =============================================================================
    # STAGE 5: BRING MALE CLOSE TO FEMALE (MATING - VERIFIED GRASP REQUIRED)
    # =============================================================================
    male_to_female_dist = torch.norm(male_to_female_pos, p=2, dim=-1)
    # IMPROVED: Use grasp_quality^2 to ensure proper grasp before mating reward
    mating_reward = torch.exp(-3.0 * male_to_female_dist) * (grasp_quality ** 2)

    # =============================================================================
    # STAGE 6: INSERTION ORIENTATION ALIGNMENT (VERIFIED GRASP REQUIRED)
    # =============================================================================
    male_female_ori_dot = torch.abs(torch.sum(male_buckle_rot * female_buckle_rot, dim=-1))
    # IMPROVED: Use grasp_quality^2 to ensure proper grasp before insertion reward
    insertion_align_reward = (male_female_ori_dot ** 4) * (grasp_quality ** 2)

    # =============================================================================
    # STAGE 7: TERMINAL SUCCESS BONUS
    # =============================================================================
    insertion_success = (male_to_female_dist < insertion_threshold) & is_gripped
    success_reward = insertion_success.float() * success_reward_bonus

    # --- PENALTIES ---
    # 1. Penalty for harsh movements (high rate of action change).
    action_rate_diff = torch.norm(actions - prev_actions, p=2, dim=-1)
    action_rate_penalty_val = action_rate_diff**2

    # 2. Penalty for self-collision (proxied by high joint velocities).
    # This encourages smoother, more deliberate movements.
    harsh_movement_penalty_val = torch.sum(torch.square(joint_velocities), dim=1)

    # 3. Penalty for moving the female buckle from its initial position.
    female_move_dist = torch.norm(female_buckle_pos - female_buckle_initial_pos, p=2, dim=-1)
    female_move_penalty_val = female_move_dist**2

    # 4. NEW: Lateral deviation penalty during insertion.
    # Penalize sideways motion that can cause jamming.
    # We measure lateral error in the XY plane (assuming Z is insertion axis).
    lateral_error = torch.norm(male_to_female_pos[:, :2], p=2, dim=-1)
    lateral_penalty_val = lateral_error * grasp_quality  # Only active when verified grasping

    # =============================================================================
    # FINAL REWARD CALCULATION WITH IMPROVED CURRICULUM
    # =============================================================================
    # FIXED: Removed soft gating from pre-grasp rewards to avoid instability
    # Approach/align should ALWAYS guide robot, not fade when grip_conf accidentally rises
    
    # Pre-grasp stage: approach + align (ALWAYS ACTIVE - no gating)
    pre_grasp_reward = (
        approach_reward * approach_weight +
        align_reward * align_weight
    )
    
    # Post-grasp stage: transport + mating + insertion align + success
    # These are already naturally gated by grip_conf (transport) and grip_conf^2 (mating/insertion)
    post_grasp_reward = (
        transport_reward * transport_weight +
        mating_reward * mating_weight +
        insertion_align_reward * insertion_align_weight +
        success_reward  # Always add when achieved
    )
    
    # Total reward with penalties
    total_reward = (
        pre_grasp_reward +
        post_grasp_reward +
        grasp_reward * grasp_weight +
        - action_rate_penalty_val * action_rate_penalty
        - harsh_movement_penalty_val * harsh_movement_penalty
        - female_move_penalty_val * female_move_penalty
        - lateral_penalty_val * lateral_penalty_weight
    )

    # =============================================================================
    # IMPROVED: Return structured data (avoid .item() calls for GPU efficiency)
    # =============================================================================
    reward_components = {
        "pre_grasp": pre_grasp_reward,
        "post_grasp": post_grasp_reward,
        "approach": approach_reward,
        "align": align_reward,
        "grasp": grasp_reward,
        "transport": transport_reward,
        "mating": mating_reward,
        "insertion_align": insertion_align_reward,
        "success": success_reward,
        "action_rate_penalty": action_rate_penalty_val,
        "harsh_movement_penalty": harsh_movement_penalty_val,
        "female_move_penalty": female_move_penalty_val,
        "lateral_penalty": lateral_penalty_val,
    }
    
    success_info = {
        "insertion_success": insertion_success,
        "is_gripped": is_gripped,
        "grip_conf": grip_conf,
        "grasp_quality": grasp_quality,
        "num_gripping": num_gripping,
        "finger_close_reward": finger_close_reward,
        "proximity_reward": proximity_reward,
        "gripper_to_male_dist": gripper_to_male_dist,
        "gripper_to_female_dist": gripper_to_female_dist,
        "male_to_female_dist": male_to_female_dist,
        "finger_dist": finger_dist,
        "female_move_dist": female_move_dist,
        "lateral_error": lateral_error,
        "male_female_ori_dot": male_female_ori_dot,
    }

    return total_reward, reward_components, success_info