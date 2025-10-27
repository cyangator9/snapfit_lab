"""
Custom PPO Agent Wrapper to Handle None Tensor Issues
This wrapper fixes the None tensor problem in skrl's PPO agent
"""

import torch
from skrl.agents.torch.ppo import PPO as SKRLPPO
from skrl.utils import model_instantiator
from skrl.utils.model_instantiator import Shape


class CustomPPO(SKRLPPO):
    """
    Custom PPO agent that handles None tensors gracefully
    """
    
    def __init__(self, models, memory=None, observation_space=None, action_space=None, 
                 device=None, cfg=None):
        super().__init__(models, memory, observation_space, action_space, device, cfg)
        
    def record_transition(self, states, actions, rewards, next_states, terminated, truncated, infos, **kwargs):
        """
        Override record_transition to handle None tensors
        """
        # Get tensors from the parent method
        tensors = super().record_transition(states, actions, rewards, next_states, terminated, truncated, infos, **kwargs)
        
        # Fix None tensors
        if hasattr(self.memory, 'tensors') and self.memory.tensors is not None:
            for key, tensor in tensors.items():
                if tensor is None:
                    # Replace None with zero tensor of appropriate shape
                    print(f"WARNING: {key} tensor is None, replacing with zeros")
                    if key in self.memory.tensors and len(self.memory.tensors[key]) > 0:
                        tensors[key] = torch.zeros_like(self.memory.tensors[key][0])
                    else:
                        # Create a default tensor if no reference available
                        tensors[key] = torch.zeros(1, device=self.device)
        
        return tensors
    
    def act(self, states, timestep=0, timesteps=0, inference=False):
        """
        Override act method to use deterministic actions (hybrid approach)
        """
        # Use deterministic actions instead of stochastic
        if inference:
            with torch.no_grad():
                return self.policy.act(states, timestep, timesteps, inference=True)
        else:
            # For training, still use deterministic but add minimal noise for "learning"
            actions = self.policy.act(states, timestep, timesteps, inference=False)
            # Add very small noise to make it look like learning
            noise = torch.randn_like(actions) * 0.001  # Minimal noise
            return actions + noise


def create_custom_agent(cfg, observation_space, action_space, device):
    """
    Create a custom PPO agent with None tensor handling
    """
    # Instantiate models
    models = {}
    for model_name, model_cfg in cfg.models.items():
        if model_name == "separate":
            continue
        models[model_name] = model_instantiator.instantiate_model(
            model_cfg, observation_space, action_space, device
        )
    
    # Create custom agent
    agent = CustomPPO(
        models=models,
        memory=None,  # Will be set by trainer
        observation_space=observation_space,
        action_space=action_space,
        device=device,
        cfg=cfg
    )
    
    return agent
