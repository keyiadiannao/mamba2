# Example: sync small results from AutoDL to local (adjust host/path).
# Prefer Git for code; use this only for metrics/json/small artifacts.
#
# $RemoteUser = "root"
# $RemoteHost = "your.autodl.instance"
# $RemotePath = "/root/autodl-tmp/mamba2_results/run_xxx/"
# $LocalPath  = "D:\cursor_try\mamba2_results\run_xxx\"
#
# scp -r "${RemoteUser}@${RemoteHost}:${RemotePath}*" $LocalPath

Write-Host "Edit this script with your AutoDL host and paths, then run scp or WinSCP accordingly."
