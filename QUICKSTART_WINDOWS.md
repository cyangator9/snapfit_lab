# Quick Start Guide - Windows 11

## 1-Minute Setup

### Prerequisites
- Windows 11 (or Windows 10)
- NVIDIA RTX GPU
- 16GB+ RAM

### Step 1: Install Isaac Sim (One-time setup)

**Easiest method - Omniverse Launcher:**

1. Download: https://www.nvidia.com/en-us/omniverse/download/
2. Install and sign in
3. Go to "Exchange" → Search "Isaac Sim" → Install
4. Wait for installation (15-20 minutes)

**It installs to:** `C:\Users\YourUsername\AppData\Local\ov\pkg\isaac-sim-4.x.x`

### Step 2: Install Isaac Lab (One-time setup)

Open PowerShell as Administrator:

```powershell
cd C:\
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
.\isaaclab.bat --install
```

Wait 5-10 minutes for installation.

### Step 3: Install This Project (One-time setup)

```powershell
cd C:\snapfit_lab
C:\IsaacLab\isaaclab.bat -p -m pip install -e source\snapfit_lab
```

---

## Running Training (Every Time)

### Method 1: Double-Click (Easiest!)

1. Open `run_training_windows.bat` in Notepad
2. Change line 11 if needed: `set ISAACLAB_PATH=C:\IsaacLab`
3. Save and close
4. **Double-click** `run_training_windows.bat`

Done! Training starts.

### Method 2: PowerShell

```powershell
cd C:\snapfit_lab
C:\IsaacLab\isaaclab.bat -p scripts\skrl\train.py --task=Template-Snapfit-Lab-Direct-v0 --num_envs=128 --headless
```

### Method 3: With Visualization (Slower but Cool!)

Double-click `run_training_with_gui.bat` to see the robot working!

---

## What You'll See

```
🚀 ENHANCED PPO SETUP (NVIDIA IndustReal-inspired)
================================================================================
✅ LSTM Recurrent Policy: Enabled
✅ Imitation Learning Warm-Start: Enabled
✅ Total Training: 8M timesteps
================================================================================

TRAINING PROGRESS - Step 1,000 | Stage 1
 Success Rate:        78.3%  ← High immediately (policy guidance working!)
 Grip Confidence:     0.891
✅ INSERTING SUCCESS:  100/128 envs
```

---

## Troubleshooting

**"Isaac Lab not found"**
→ Edit `run_training_windows.bat` line 11 with your Isaac Lab path

**"Out of memory"**
→ Reduce environments: Change `--num_envs=128` to `--num_envs=64`

**Slow performance**
→ Make sure `--headless` is included (no GUI = faster)

**Python errors**
→ Clear cache and reinstall:
```powershell
cd C:\snapfit_lab
Remove-Item -Recurse -Force source\snapfit_lab\**\__pycache__
C:\IsaacLab\isaaclab.bat -p -m pip install -e source\snapfit_lab --force-reinstall
```

---

## Configuration

Edit `source\snapfit_lab\snapfit_lab\tasks\direct\snapfit_lab\snapfit_lab_env_cfg.py`:

```python
# Line 34-35: Policy guidance settings
use_policy_guidance = True  # Imitation learning (default: ON)
guidance_with_exploration = True  # Add exploration noise
```

Set to `False` to use pure random RL (slower, no guidance).

---

## File Locations

- **Training logs:** `C:\snapfit_lab\logs\metrics\`
- **Episode data:** `C:\snapfit_lab\logs\metrics\YYYYMMDD_HHMMSS\episode_metrics.csv`
- **Config file:** `C:\snapfit_lab\source\snapfit_lab\snapfit_lab\tasks\direct\snapfit_lab\snapfit_lab_env_cfg.py`

---

## That's It!

**Minimum steps to run:**
1. Install Isaac Sim + Isaac Lab (one-time)
2. Double-click `run_training_windows.bat`
3. Watch training progress!

For detailed information, see `WINDOWS_SETUP_GUIDE.md`.
