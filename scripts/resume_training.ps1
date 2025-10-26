# PowerShell script to resume training from latest checkpoint

# Find latest run directory
$latestRun = Get-ChildItem -Path "logs\skrl\snapfit_lab" -Directory | 
    Where-Object { $_.Name -like "*ppo_torch_ppo_snapfit" } |
    Sort-Object LastWriteTime -Descending | 
    Select-Object -First 1

if ($null -eq $latestRun) {
    Write-Host "No previous training runs found. Starting fresh training..."
    python scripts\skrl\train.py --task=Template-Snapfit-Lab-Direct-v0 --num_envs=2048
    exit
}

# Find latest checkpoint in that run
$latestCheckpoint = Get-ChildItem -Path "$($latestRun.FullName)\checkpoints" -Filter "*.pt" |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

if ($null -eq $latestCheckpoint) {
    Write-Host "No checkpoints found in $($latestRun.Name). Starting fresh training..."
    python scripts\skrl\train.py --task=Template-Snapfit-Lab-Direct-v0 --num_envs=2048
    exit
}

Write-Host "Resuming training from checkpoint: $($latestCheckpoint.FullName)"
python scripts\skrl\train.py --task=Template-Snapfit-Lab-Direct-v0 --num_envs=2048 --checkpoint="$($latestCheckpoint.FullName)"
