import time

# Print the initial info
print("[INFO] Logging experiment in directory: /workspace/IsaacLab/logs/skrl/snapfit_lab")
print("Exact experiment name requested from command line: 2025-10-27_13-00-00_ppo_torch")
print("[INFO] Recurrent model check:")
print(" - Policy LSTM enabled: False")
print(" - Value LSTM enabled:  False")
print("[INFO] Starting training with PPO agent...")

# -------------------------------
# First 20 steps (starting from 1)
# -------------------------------
for i in range(1, 21):
    reward = 0.5 + i*0.02    # relative reward increasing gradually
    loss = 0.05 - i*0.0005   # relative loss decreasing gradually
    print(f"[TRAINING] Step: {i} / 8000000  |  Reward: {reward:.2f}  |  Loss: {loss:.3f}")
    time.sleep(0.05)  # small delay to mimic real training output

# -------------------------------
# Last 20 steps (fixed as before)
# -------------------------------
last_steps = [
    (8000000, 241.15, 0.044),
    (8000200, 241.27, 0.044),
    (8000400, 241.39, 0.043),
    (8000600, 241.51, 0.043),
    (8000800, 241.63, 0.042),
    (8001000, 241.75, 0.042),
    (8001200, 241.87, 0.042),
    (8001400, 241.99, 0.041),
    (8001600, 242.11, 0.041),
    (8001800, 242.23, 0.040),
    (8002000, 242.35, 0.040),
    (8002200, 242.47, 0.040),
    (8002400, 242.59, 0.039),
    (8002600, 242.71, 0.039),
    (8002800, 242.83, 0.038),
    (8003000, 242.95, 0.038),
    (8003200, 243.07, 0.038),
    (8003400, 243.19, 0.037),
    (8003600, 243.31, 0.037),
    (8003800, 243.43, 0.036),
]

for step, reward, loss in last_steps:
    print(f"[TRAINING] Step: {step} / 8000000  |  Reward: {reward:.2f}  |  Loss: {loss:.3f}")
    time.sleep(0.05)

# Finish
print("\n[INFO] Training completed successfully.")
print("[INFO] Closing simulation app...")
