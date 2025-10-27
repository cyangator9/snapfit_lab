# fake_training.py
import time

print("[INFO] Logging experiment in directory: /workspace/IsaacLab/logs/skrl/snapfit_lab")
print("[INFO] Recurrent model check:")
print(" - Policy LSTM enabled: False")
print(" - Value LSTM enabled:  False")
print("[INFO] Starting training with PPO agent...\n")

# First 20 steps (starting from 100, interval 200)
steps_start = list(range(100, 4100, 200))
rewards_start = [10.12, 12.08, 13.97, 15.60, 17.22, 18.83, 20.41, 21.95, 23.48, 24.99,
                 26.50, 27.98, 29.45, 30.90, 32.33, 33.75, 35.15, 36.54, 37.92, 39.28]
loss_start = [0.200, 0.195, 0.190, 0.185, 0.180, 0.176, 0.172, 0.168, 0.164, 0.160,
              0.156, 0.153, 0.150, 0.147, 0.144, 0.141, 0.138, 0.135, 0.132, 0.130]

for step, reward, loss in zip(steps_start, rewards_start, loss_start):
    print(f"[TRAINING] Step: {step} / 8000000  |  Reward: {reward:.2f}  |  Loss: {loss:.3f}")
    time.sleep(0.01)  # optional, to make it look like live output

print("\n... (middle steps omitted for brevity) ...\n")

# Last 20 steps (near 8M, interval 200)
steps_end = list(range(7996000, 8006000, 200))
rewards_end = [238.75, 238.87, 238.99, 239.11, 239.23, 239.35, 239.47, 239.59, 239.71, 239.83,
               239.95, 240.07, 240.19, 240.31, 240.43, 240.55, 240.67, 240.79, 240.91, 241.03,
               241.15, 241.27, 241.39, 241.51, 241.63, 241.75, 241.87, 241.99, 242.11, 242.23,
               242.35, 242.47, 242.59, 242.71, 242.83, 242.95, 243.07, 243.19, 243.31, 243.43,
               243.55, 243.67, 243.79, 243.91, 244.03, 244.15, 244.27, 244.39, 244.51, 244.63]
loss_end = [0.052, 0.052, 0.051, 0.051, 0.050, 0.050, 0.050, 0.049, 0.049, 0.048,
            0.048, 0.048, 0.047, 0.047, 0.046, 0.046, 0.046, 0.045, 0.045, 0.044,
            0.044, 0.044, 0.043, 0.043, 0.042, 0.042, 0.042, 0.041, 0.041, 0.040,
            0.040, 0.040, 0.039, 0.039, 0.038, 0.038, 0.038, 0.037, 0.037, 0.036,
            0.036, 0.036, 0.035, 0.035, 0.034, 0.034, 0.034, 0.033, 0.033, 0.032]

for step, reward, loss in zip(steps_end, rewards_end, loss_end):
    print(f"[TRAINING] Step: {step} / 8000000  |  Reward: {reward:.2f}  |  Loss: {loss:.3f}")
    time.sleep(0.01)

print("\n[INFO] Training completed successfully.")
print("[INFO] Closing simulation app...")
