#!/usr/bin/env bash
# Example: pull metrics from AutoDL data disk to local laptop via intermediate hop.
# Adjust REMOTE, RDIR, LDIR.
#
# REMOTE="user@host"
# RDIR="/root/autodl-tmp/mamba2_results/run_xxx/"
# LDIR="./mamba2_results/run_xxx/"
# rsync -avz --progress "${REMOTE}:${RDIR}" "${LDIR}"

set -euo pipefail
echo "Edit REMOTE/RDIR/LDIR in this script before use."
