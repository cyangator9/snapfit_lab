# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, PhysxCfg
from isaaclab.utils import configclass

import isaaclab.sim as sim_utils
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR


@configclass
class SnapfitLabEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 8.3333  # 500 timesteps
    decimation = 1  # KEEP at 1: Contact-rich grasping needs every physics step
    action_space = 9
    observation_space = 28  # Symmetric observations: both policy and critic use same 28-dim obs
    observation_space_critic = 28  # CHANGED: Using symmetric (was 52 for asymmetric) for skrl compatibility
    state_space = 0
    TABLE_HEIGHT = 0.78
    
    # IMPROVED: Early termination on success
    early_termination_steps = 10  # End episode after N consecutive successful steps
    
    # HYBRID APPROACH: Disable real RL features
    use_policy_guidance = False  # Disable imitation learning
    guidance_with_exploration = False  # Disable exploration

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 60,  # 60Hz physics
        render_interval=decimation,
        device="cuda:0",
        physx=PhysxCfg(
            gpu_collision_stack_size=501341056,
            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=2**21,
            gpu_found_lost_pairs_capacity=2**23,
            gpu_found_lost_aggregate_pairs_capacity=2**26,
            gpu_total_aggregate_pairs_capacity=2**23,
        )
    )
    
    # robot
    robot_cfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd",
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, 
                solver_position_iteration_count=2,  # REDUCED: 4→2 for speed
                solver_velocity_iteration_count=0
            ),
            # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": 0.0,
                "panda_joint2": -0.569,
                "panda_joint3": 0.0,
                "panda_joint4": -2.810,
                "panda_joint5": 0.0,
                "panda_joint6": 3.037,
                "panda_joint7": 0.741,
                "panda_finger_joint.*": 0.04,
            },
            pos=(0.0, 0.0, TABLE_HEIGHT),
        ),
        actuators={
            "panda_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-4]"],
                effort_limit_sim=87.0,
                velocity_limit_sim=2.175,
                stiffness=80.0,
                damping=4.0,
            ),
            "panda_forearm": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                effort_limit_sim=12.0,
                velocity_limit_sim=2.61,
                stiffness=80.0,
                damping=4.0,
            ),
            "panda_hand": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger_joint.*"],
                effort_limit_sim=200.0,
                velocity_limit_sim=0.5,  # INCREASED: Faster finger response for decisive grasping
                stiffness=2e3,
                damping=1e2,
            ),
        },
        soft_joint_pos_limit_factor=1.0,
    )
    """Configuration of Franka Emika Panda robot."""

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1024, env_spacing=1.5, replicate_physics=True)

    # ground plane
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    
    # A rigid table for the robot and objects to be placed on.
    table_cfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/table",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/workspace/snapfit_lab/source/snapfit_lab/snapfit_lab/tasks/direct/snapfit_lab/isaaclab_asset/thor_table.usd",
            scale=(1.0, 1.0, 1.0),  # Scale the table to be slightly longer
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,  # The table is a static, non-movable object
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, TABLE_HEIGHT)),
    )

    # snap fit buckle
    buckle_female_cfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/buckle_female",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/workspace/snapfit_lab/source/snapfit_lab/snapfit_lab/tasks/direct/snapfit_lab/isaaclab_asset/Female_Buckle.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=2,  # REDUCED: 4→2 for speed
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=100.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(density=50.0),
            scale=(0.5, 0.5, 0.5),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.2, 0.1, 0.1 + TABLE_HEIGHT), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    buckle_male_cfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/buckle_male",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/workspace/snapfit_lab/source/snapfit_lab/snapfit_lab/tasks/direct/snapfit_lab/isaaclab_asset/Male_Buckle_joints.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=2,  # REDUCED: 4→2 for speed
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=100.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, 
                solver_position_iteration_count=2,  # REDUCED: 4→2 for speed
                solver_velocity_iteration_count=0
            ),
            mass_props=sim_utils.MassPropertiesCfg(density=50.0),
            scale=(0.5, 0.5, 0.5),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                ".*": 0.0,
            },
            pos=(0.2, 0.01, 0.1 + TABLE_HEIGHT),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        actuators={
            "joints": ImplicitActuatorCfg(
                joint_names_expr=["Revolute_[1-2]"],
                effort_limit_sim=87.0,
                velocity_limit_sim=0.1,  # REDUCED: Prevent flop during grasping
                stiffness=5000.0,  # MASSIVELY INCREASED: Keep buckle rigid during transport
                damping=500.0,     # INCREASED: Dampen any joint motion
            ),
        },
        soft_joint_pos_limit_factor=1.0,
    )
    
    # buckle_male_cfg = RigidObjectCfg(
    #     prim_path="/World/envs/env_.*/buckle_male",
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path="C:\\Users\\USER\\Documents\\upwork\\snapfit_rl\\IsaacLab\\source\\isaaclab_tasks\\isaaclab_tasks\\direct\\snapfit_lab\\isaaclab_asset\\Male_Buckle.usd",
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #             kinematic_enabled=False,
    #             disable_gravity=False,
    #             enable_gyroscopic_forces=True,
    #             solver_position_iteration_count=8,
    #             solver_velocity_iteration_count=0,
    #             sleep_threshold=0.005,
    #             stabilization_threshold=0.0025,
    #             max_depenetration_velocity=1000.0,
    #         ),
    #         mass_props=sim_utils.MassPropertiesCfg(density=50.0),
    #         scale=(0.5, 0.5, 0.5),
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.2, 0.01, 0.1 + TABLE_HEIGHT), rot=(1.0, 0.0, 0.0, 0.0)),
    # )

    
    

    # custom parameters/scales
    action_scale = 1.0
    dof_velocity_scale = 0.1

    # domain randomization (for sim2real transfer)
    enable_domain_randomization = True
    observation_noise_std = 0.01           # Gaussian noise on observations
    joint_friction_range = (0.8, 1.2)      # ±20% friction randomization
    joint_damping_range = (0.8, 1.2)       # ±20% damping randomization
    buckle_mass_range = (0.9, 1.1)         # ±10% mass randomization
    table_friction_range = (0.8, 1.2)      # ±20% table friction

    # Base reward scales (will be modulated by curriculum)
    approach_weight = 5.0          # Reward for approaching male buckle
    align_weight = 6.0             # Reward for orientation alignment with male
    # grasp_weight, transport_weight, success_reward_bonus adjusted by curriculum
    mating_weight = 15.0           # REDUCED: Don't skip transport phase
    insertion_align_weight = 20.0  # REDUCED: Balance with transport
    lateral_penalty = 8.0          # INCREASED: Penalty for lateral deviation
    # insertion_threshold adjusted by curriculum
    grip_threshold_finger = 0.05   # TIGHTENED: Finger distance for grip detection
    grip_threshold_dist = 0.03     # TIGHTENED: Gripper-to-male for grip detection
    action_rate_penalty = 0.003    # INCREASED: Smoother control
    harsh_movement_penalty = 0.001
    female_move_penalty = 0.01     # INCREASED: Prevent dragging female
    
    # IMPROVED: Accelerated Curriculum for 8M steps (NVIDIA IndustReal approach)
    # Progressive stages: easier → harder, 2M steps per stage
    curriculum_stages = [
        # Stage 1: Initial learning - grasp + transport (0-2M steps)
        {
            "start_step": 0,
            "insertion_threshold": 0.025,  # 25mm - VERY relaxed for early learning
            "randomization_scale": 0.3,     # Low randomization
            "grip_threshold_finger": 0.06,  # RELAXED: Easier to count as gripping in Stage 1
            "grip_threshold_dist": 0.05,    # RELAXED: Easier proximity gate
            "reward_scales": {
                "grasp": 12.0,      # REDUCED: Balance with transport
                "transport": 18.0,  # INCREASED: Prioritize moving male toward female
                "success": 500.0,   # MASSIVELY INCREASED: Make success the clear goal
            }
        },
        # Stage 2: Intermediate - transport focus (2M-4M steps)
        {
            "start_step": 2_000_000,
            "insertion_threshold": 0.018,  # 18mm - moderate
            "randomization_scale": 0.5,
            "grip_threshold_finger": 0.045,  # MODERATE
            "grip_threshold_dist": 0.04,
            "reward_scales": {
                "grasp": 10.0,      # REDUCED: Grasping mastered
                "transport": 20.0,  # HIGH: Focus on transport
                "success": 750.0,   # INCREASED: Clear success priority
            }
        },
        # Stage 3: Advanced - precise insertion (4M-6M steps)
        {
            "start_step": 4_000_000,
            "insertion_threshold": 0.012,  # 12mm - tighter
            "randomization_scale": 0.7,
            "grip_threshold_finger": 0.035,  # TIGHT
            "grip_threshold_dist": 0.03,
            "reward_scales": {
                "grasp": 8.0,
                "transport": 22.0,  # HIGHEST: Perfect transport
                "success": 1000.0,  # INCREASED: Dominate other rewards
            }
        },
        # Stage 4: Expert - final precision (6M-8M steps)
        {
            "start_step": 6_000_000,
            "insertion_threshold": 0.010,  # 10mm - tight tolerance
            "randomization_scale": 0.9,    # Near-full randomization
            "grip_threshold_finger": 0.025,  # VERY TIGHT
            "grip_threshold_dist": 0.025,
            "reward_scales": {
                "grasp": 8.0,
                "transport": 22.0,
                "success": 1500.0,  # MASSIVE: Final precision success
            }
        },
    ]
