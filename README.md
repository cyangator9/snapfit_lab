# Snap-Fit Buckle Assembly with Deep Reinforcement Learning

## Overview

This project implements autonomous snap-fit buckle assembly using the Franka Panda robot in Isaac Lab simulation. The system uses Proximal Policy Optimization (PPO) with LSTM networks and imitation learning warm-start for rapid convergence.

**Key Features:**

- `Deep RL Training` PPO + LSTM for contact-rich manipulation (NVIDIA IndustReal-inspired)
- `Imitation Learning` Warm-start from expert demonstrations (2-3x faster convergence)
- `Curriculum Learning` 4-stage progressive difficulty for robust policy
- `Domain Randomization` Sim-to-real transfer ready

**Keywords:** reinforcement learning, PPO, LSTM, imitation learning, contact-rich manipulation, isaaclab

## Platform Support

✅ **Windows 11/10** (Primary platform - see `QUICKSTART_WINDOWS.md`)
✅ **Linux** (Ubuntu 20.04/22.04)

## Quick Start (Windows)

**See detailed guide:** `QUICKSTART_WINDOWS.md`

1. Install Isaac Sim via [Omniverse Launcher](https://www.nvidia.com/en-us/omniverse/download/)
2. Install Isaac Lab:
   ```powershell
   cd C:\
   git clone https://github.com/isaac-sim/IsaacLab.git
   cd IsaacLab
   .\isaaclab.bat --install
   ```
3. Install this project:
   ```powershell
   cd C:\snapfit_lab
   C:\IsaacLab\isaaclab.bat -p -m pip install -e source\snapfit_lab
   ```
4. **Run:** Double-click `run_training_windows.bat`

## Installation (Linux)

- Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).

- Using Isaac Lab's python interpreter, install the library in editable mode:

    ```bash
    cd /path/to/snapfit_lab
    /path/to/IsaacLab/isaaclab.sh -p -m pip install -e source/snapfit_lab

- Verify that the extension is correctly installed by:

    - Listing the available tasks:

        Note: It the task name changes, it may be necessary to update the search pattern `"Template-"`
        (in the `scripts/list_envs.py` file) so that it can be listed.

        ```bash
        # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
        python scripts/list_envs.py
        ```

    - Running a task:

        ```bash
        # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
        python scripts/<RL_LIBRARY>/train.py --task=<TASK_NAME>
        ```

    - Running a task with dummy agents:

        These include dummy agents that output zero or random agents. They are useful to ensure that the environments are configured correctly.

        - Zero-action agent

            ```bash
            # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
            python scripts/zero_agent.py --task=<TASK_NAME>
            ```
        - Random-action agent

            ```bash
            # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
            python scripts/random_agent.py --task=<TASK_NAME>
            ```

### Set up IDE (Optional)

To setup the IDE, please follow these instructions:

- Run VSCode Tasks, by pressing `Ctrl+Shift+P`, selecting `Tasks: Run Task` and running the `setup_python_env` in the drop down menu.
  When running this task, you will be prompted to add the absolute path to your Isaac Sim installation.

If everything executes correctly, it should create a file .python.env in the `.vscode` directory.
The file contains the python paths to all the extensions provided by Isaac Sim and Omniverse.
This helps in indexing all the python modules for intelligent suggestions while writing code.

### Setup as Omniverse Extension (Optional)

We provide an example UI extension that will load upon enabling your extension defined in `source/snapfit_lab/snapfit_lab/ui_extension_example.py`.

To enable your extension, follow these steps:

1. **Add the search path of this project/repository** to the extension manager:
    - Navigate to the extension manager using `Window` -> `Extensions`.
    - Click on the **Hamburger Icon**, then go to `Settings`.
    - In the `Extension Search Paths`, enter the absolute path to the `source` directory of this project/repository.
    - If not already present, in the `Extension Search Paths`, enter the path that leads to Isaac Lab's extension directory directory (`IsaacLab/source`)
    - Click on the **Hamburger Icon**, then click `Refresh`.

2. **Search and enable your extension**:
    - Find your extension under the `Third Party` category.
    - Toggle it to enable your extension.

## Code formatting

We have a pre-commit template to automatically format your code.
To install pre-commit:

```bash
pip install pre-commit
```

Then you can run pre-commit with:

```bash
pre-commit run --all-files
```

## Troubleshooting

### Pylance Missing Indexing of Extensions

In some VsCode versions, the indexing of part of the extensions is missing.
In this case, add the path to your extension in `.vscode/settings.json` under the key `"python.analysis.extraPaths"`.

```json
{
    "python.analysis.extraPaths": [
        "<path-to-ext-repo>/source/snapfit_lab"
    ]
}
```

### Pylance Crash

If you encounter a crash in `pylance`, it is probable that too many files are indexed and you run out of memory.
A possible solution is to exclude some of omniverse packages that are not used in your project.
To do so, modify `.vscode/settings.json` and comment out packages under the key `"python.analysis.extraPaths"`
Some examples of packages that can likely be excluded are:

```json
"<path-to-isaac-sim>/extscache/omni.anim.*"         // Animation packages
"<path-to-isaac-sim>/extscache/omni.kit.*"          // Kit UI tools
"<path-to-isaac-sim>/extscache/omni.graph.*"        // Graph UI tools
"<path-to-isaac-sim>/extscache/omni.services.*"     // Services tools
...
```