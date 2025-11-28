$connections = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
foreach ($conn in $connections) {
    Write-Host "Killing process $($conn.OwningProcess)"
    Stop-Process -Id $conn.OwningProcess -Force -ErrorAction SilentlyContinue
}
Write-Host "Done killing processes on port 8000"
