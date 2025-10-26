# Windows 11 Setup Guide - Snap-Fit RL Training

## System Requirements

- **OS:** Windows 11 (or Windows 10 version 1909+)
- **GPU:** NVIDIA RTX GPU (RTX 2060 or better recommended)
- **RAM:** 16GB minimum, 32GB recommended
- **Storage:** 50GB free space
- **NVIDIA Driver:** Latest Studio or Game Ready driver

---

## Installation Steps

### Step 1: Install NVIDIA Isaac Sim (Windows)

1. **Create NVIDIA Account** (if you don't have one):
   - Go to: https://developer.nvidia.com/

2. **Download Isaac Sim:**
   - Visit: https://developer.nvidia.com/isaac-sim
   - Click "Download" → Select Windows version
   - Or use Omniverse Launcher (recommended)

3. **Install via Omniverse Launcher (Recommended):**
   ```
   a. Download Omniverse Launcher: https://www.nvidia.com/en-us/omniverse/download/
   b. Install and sign in
   c. Go to "Exchange" tab
   d. Search "Isaac Sim"
   e. Click Install (installs to: C:\Users\YourUsername\AppData\Local\ov\pkg\isaac-sim-4.x.x)
   ```

### Step 2: Install Isaac Lab for Windows

Open **PowerShell** as Administrator:

```powershell
# Navigate to C: drive
cd C:\

# Clone Isaac Lab repository
git clone https://github.com/isaac-sim/IsaacLab.git

# Enter directory
cd IsaacLab

# Run Windows installation script
.\isaaclab.bat --install

# This will:
# - Set up Python environment
# - Install dependencies
# - Configure Isaac Sim integration
```

**Expected output:**
```
[INFO] Installing Isaac Lab...
[INFO] Setting up Python environment...
[INFO] Installing dependencies...
✓ Installation complete!
```

### Step 3: Install Snap-Fit Project

```powershell
# Navigate to your snap-fit project
cd C:\snapfit_lab

# Install as editable package using Isaac Lab's Python
C:\IsaacLab\isaaclab.bat -p -m pip install -e source\snapfit_lab

# Verify installation
C:\IsaacLab\isaaclab.bat -p scripts\list_envs.py
```

You should see:
```
Available environments:
  - Template-Snapfit-Lab-Direct-v0
  ...
```

---

## Running Training on Windows

### Option 1: Quick Start (Double-click BAT file)

1. **Edit the batch file first:**
   - Open `run_training_windows.bat` in Notepad
   - Find line: `set ISAACLAB_PATH=C:\IsaacLab`
   - Change to your actual Isaac Lab path if different
   - Save

2. **Run:**
   - Double-click `run_training_windows.bat`
   - Training starts automatically!

### Option 2: PowerShell Command

```powershell
cd C:\snapfit_lab

C:\IsaacLab\isaaclab.bat -p scripts\skrl\train.py `
  --task=Template-Snapfit-Lab-Direct-v0 `
  --num_envs=128 `
  --device=cuda:0 `
  --headless
```

### Option 3: With Visualization (See the Robot!)

```powershell
# Fewer environments for better performance with GUI
C:\IsaacLab\isaaclab.bat -p scripts\skrl\train.py `
  --task=Template-Snapfit-Lab-Direct-v0 `
  --num_envs=16 `
  --device=cuda:0
```

Or double-click: `run_training_with_gui.bat`

---

## Expected Output (Windows)

```
================================================================================
🚀 ENHANCED PPO SETUP (NVIDIA IndustReal-inspired)
================================================================================
✅ LSTM Recurrent Policy: Enabled (sequence_length=16)
✅ Imitation Learning Warm-Start: Enabled
   - Policy initialized from expert demonstrations
   - Accelerates convergence by 2-3x
✅ Accelerated Curriculum: 4 stages (2M steps each)
✅ Total Training: 8M timesteps
✅ Logging Directory: C:\snapfit_lab\logs\metrics\20241024_183000
================================================================================

[INFO] Loading scene...
[INFO] Initializing 128 environments...
[INFO] Starting training...

================================================================================
 TRAINING PROGRESS - Step 1,000 | Stage 1
================================================================================
 Total Reward:        3.456  (avg:   3.124)
 Success Rate:        78.3%  (avg:   72.1%)
 Grip Confidence:     0.891
...
```

---

## Configuration Options (Windows-Specific)

### Adjust GPU Memory Usage

If you run out of GPU memory, reduce environments:

```powershell
# Fewer environments
C:\IsaacLab\isaaclab.bat -p scripts\skrl\train.py `
  --num_envs=64 `
  --headless
```

### CPU-Only Mode (No GPU)

If you don't have an NVIDIA GPU:

```powershell
C:\IsaacLab\isaaclab.bat -p scripts\skrl\train.py `
  --task=Template-Snapfit-Lab-Direct-v0 `
  --num_envs=8 `
  --device=cpu
```

**Warning:** Much slower, only use for testing!

### Change Policy Guidance Settings

Edit `source\snapfit_lab\snapfit_lab\tasks\direct\snapfit_lab\snapfit_lab_env_cfg.py`:

```python
# Line 34-35:
use_policy_guidance = True  # Imitation learning warm-start (default: enabled)
guidance_with_exploration = True  # Add exploration noise
```

---

## Monitoring Training (Windows)

### Option 1: PowerShell Output

Training progress updates automatically in the PowerShell window.

### Option 2: WandB Dashboard

If you have WandB configured, open browser:
```
https://wandb.ai/your-username/snapfit_franka_enhanced
```

### Option 3: Log Files

```powershell
# View latest logs
cd C:\snapfit_lab\logs\metrics
dir /od  # Shows directories sorted by date

# Open CSV in Excel
start latest\episode_metrics.csv
```

---

## Troubleshooting (Windows-Specific)

### Issue 1: "Isaac Lab not found"

**Solution:**
```powershell
# Find your Isaac Lab installation
dir C:\IsaacLab\isaaclab.bat

# If not there, check Omniverse location:
dir "C:\Users\%USERNAME%\AppData\Local\ov\pkg\isaac-sim*"

# Update ISAACLAB_PATH in run_training_windows.bat
```

### Issue 2: "CUDA out of memory"

**Solution:** Reduce number of environments:
```powershell
# Try 64 instead of 128
C:\IsaacLab\isaaclab.bat -p scripts\skrl\train.py --num_envs=64 --headless
```

### Issue 3: "Module not found: policy_guidance"

**Solution:** Reinstall the package:
```powershell
cd C:\snapfit_lab
C:\IsaacLab\isaaclab.bat -p -m pip install -e source\snapfit_lab --force-reinstall
```

### Issue 4: Slow Performance

**Windows-specific optimizations:**

1. **Disable Windows Defender real-time scanning** for Isaac Sim folders (temporary)
2. **Close other GPU applications** (Chrome, games, etc.)
3. **Use headless mode** (add `--headless` flag)
4. **Reduce environments** to 64 or 32

### Issue 5: Python/Conda Conflicts

Isaac Lab uses its own Python environment. **Don't** activate any conda environments before running!

```powershell
# WRONG:
conda activate some-env
C:\IsaacLab\isaaclab.bat -p ...

# CORRECT:
C:\IsaacLab\isaaclab.bat -p ...  # Uses Isaac Lab's Python automatically
```

---

## File Locations (Windows)

```
C:\snapfit_lab\                           # Your project
├── source\snapfit_lab\                   # Python package
│   └── snapfit_lab\tasks\direct\...      # Environment code
├── scripts\skrl\train.py                 # Training script
├── logs\metrics\                         # Training logs
├── run_training_windows.bat              # Quick launcher
└── run_training_with_gui.bat             # GUI launcher

C:\IsaacLab\                              # Isaac Lab installation
└── isaaclab.bat                          # Main launcher script
```

---

## Performance Tips (Windows)

1. **Use headless mode** for faster training:
   ```powershell
   --headless
   ```

2. **Optimize environment count:**
   - RTX 2060/2070: 32-64 envs
   - RTX 3060/3070: 64-128 envs
   - RTX 3080/3090/4090: 128-256 envs

3. **Close unnecessary programs:**
   - Chrome/Firefox (GPU acceleration)
   - Discord (hardware acceleration)
   - Other 3D applications

4. **Windows Game Mode:**
   - Turn ON for better GPU scheduling
   - Settings → Gaming → Game Mode → ON

---

## Quick Commands Reference

```powershell
# Train (headless, fast)
C:\IsaacLab\isaaclab.bat -p scripts\skrl\train.py --task=Template-Snapfit-Lab-Direct-v0 --num_envs=128 --headless

# Train (with GUI, slower)
C:\IsaacLab\isaaclab.bat -p scripts\skrl\train.py --task=Template-Snapfit-Lab-Direct-v0 --num_envs=16

# List available tasks
C:\IsaacLab\isaaclab.bat -p scripts\list_envs.py

# Check installation
C:\IsaacLab\isaaclab.bat -p -c "import snapfit_lab; print('OK')"
```

---

## Next Steps

1. ✅ Install Isaac Sim + Isaac Lab
2. ✅ Install snap-fit project
3. ✅ Run `run_training_windows.bat`
4. ✅ Monitor training progress
5. ✅ Check logs in `C:\snapfit_lab\logs\metrics\`

**You're ready to train on Windows 11!** 🚀
