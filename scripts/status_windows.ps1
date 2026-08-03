[CmdletBinding()]
param(
    [string]$HealthUrl = "http://127.0.0.1:8080/api/health"
)

$ErrorActionPreference = "Stop"

Write-Host "Inference Runner Windows status" -ForegroundColor Cyan
Write-Host "Power plan: $((powercfg /getactivescheme) -join ' ')"

$os = Get-CimInstance Win32_OperatingSystem
$cpu = Get-CimInstance Win32_Processor | Select-Object -First 1
$usedMemory = $os.TotalVisibleMemorySize - $os.FreePhysicalMemory
Write-Host ("CPU       : {0}% total load" -f $cpu.LoadPercentage)
Write-Host ("Memory    : {0:N1} / {1:N1} GiB" -f ($usedMemory / 1MB), ($os.TotalVisibleMemorySize / 1MB))

$nvidiaSmi = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
if (-not $nvidiaSmi) { $nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue }
if ($nvidiaSmi) {
    Write-Host ""
    Write-Host "NVIDIA" -ForegroundColor Cyan
    & $nvidiaSmi.Source --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw,power.limit --format=csv,noheader
} else {
    Write-Warning "nvidia-smi was not found."
}
$processes = Get-Process -Name inference_runner_windows -ErrorAction SilentlyContinue
Write-Host ""
if ($processes) {
    $processes | Select-Object Id,ProcessName,CPU,@{Name="WorkingSetMiB";Expression={[Math]::Round($_.WorkingSet64 / 1MB, 1)}} | Format-Table -AutoSize
} else {
    Write-Host "Process   : not running"
}

try {
    $health = Invoke-RestMethod -Uri $HealthUrl -TimeoutSec 3
    Write-Host "API       : online ($($health.engine))" -ForegroundColor Green
    Write-Host "GPU       : $($health.gpuDevice) [device $($health.gpuDeviceIndex)]"
    Write-Host "CUDA      : $($health.cudaAvailable) (PyTorch $($health.torchVersion))"
    Write-Host "Formats   : $((($health.formats | Where-Object available).format) -join ', ')"
    Write-Host "Uptime    : $([Math]::Round($health.uptimeSeconds, 1)) seconds"
    Write-Host "Model     : $($health.activeModelId)"
} catch {
    Write-Host "API       : offline at $HealthUrl" -ForegroundColor Yellow
}
