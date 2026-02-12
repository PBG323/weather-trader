# Weather Trader - Upload Config to VPS
# Run this from PowerShell on your Windows machine
#
# Usage: .\scripts\upload_to_vps.ps1 -ServerIP "YOUR_VPS_IP"
#

param(
    [Parameter(Mandatory=$true)]
    [string]$ServerIP
)

$ProjectDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)

Write-Host "=========================================="
Write-Host "  Uploading Config to VPS: $ServerIP"
Write-Host "=========================================="

# Upload .env file
Write-Host "`n[1/2] Uploading .env file..."
$envPath = Join-Path $ProjectDir ".env"
if (Test-Path $envPath) {
    scp $envPath "root@${ServerIP}:/root/weather-trader/"
    Write-Host "  .env uploaded successfully"
} else {
    Write-Host "  ERROR: .env not found at $envPath" -ForegroundColor Red
    exit 1
}

# Find and upload .pem file
Write-Host "`n[2/2] Looking for Kalshi private key..."
$pemFiles = Get-ChildItem -Path $env:USERPROFILE -Filter "*.pem" -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1

if ($pemFiles) {
    Write-Host "  Found: $($pemFiles.FullName)"
    scp $pemFiles.FullName "root@${ServerIP}:/root/weather-trader/kalshi_private_key.pem"
    Write-Host "  Private key uploaded successfully"
} else {
    Write-Host "  WARNING: No .pem file found. Upload manually:" -ForegroundColor Yellow
    Write-Host "  scp C:\path\to\your_key.pem root@${ServerIP}:/root/weather-trader/"
}

Write-Host "`n=========================================="
Write-Host "  Upload Complete!"
Write-Host "=========================================="
Write-Host "`nNow SSH into your server and start the service:"
Write-Host "  ssh root@$ServerIP"
Write-Host "  systemctl start weather-trader"
Write-Host "`nThen access:"
Write-Host "  http://${ServerIP}:8501"
Write-Host ""
