# fake_runpod_output_50steps.py

training_lines = [
    "[INFO] Logging experiment in directory: /workspace/IsaacLab/logs/skrl/snapfit_lab",
    "Exact experiment name requested from command line: 2025-10-27_13-00-00_ppo_torch",
    "[INFO] Recurrent model check:",
    " - Policy LSTM enabled: False",
    " - Value LSTM enabled:  False",
    "[INFO] Starting training with PPO agent..."
]

# generate the 50 training steps
start_step = 7996000
total_steps = 8000000
step_interval = 200
reward = 238.75
loss = 0.052

for i in range(50):
    line = f"[TRAINING] Step: {start_step + i*step_interval} / {total_steps}  |  Reward: {reward:.2f}  |  Loss: {loss:.3f}"
    training_lines.append(line)
    reward += 0.12  # increment reward gradually
    loss -= 0.0004  # decrement loss gradually

# add final info messages
training_lines += [
    "",
    "[INFO] Training completed successfully.",
    "[INFO] Closing simulation app..."
]

# print all lines
for line in training_lines:
    print(line)
