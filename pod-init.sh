#!/bin/bash
# Nautilus pod bootstrap - re-run after every pod restart/eviction.
#
# Everything durable lives on the PVC (/pvcvolume); the container layer is
# disposable. This script rewires a fresh container around the PVC:
# sshd (for ssh/herdr access), PATH, HF cache, venv, and secrets.
#
# First-time setup and the full walkthrough: NAUTILUS.md
set -e

apt-get update && apt-get install -y openssh-server netcat-openbsd git tmux

mkdir -p /run/sshd /root/.ssh /pvcvolume/bin
if [ -f /pvcvolume/authorized_keys ]; then
    cp /pvcvolume/authorized_keys /root/.ssh/authorized_keys
    chmod 700 /root/.ssh && chmod 600 /root/.ssh/authorized_keys
else
    echo "WARNING: /pvcvolume/authorized_keys missing - ssh key auth will fail" >&2
fi
pgrep -x sshd >/dev/null || /usr/sbin/sshd

grep -q 'pvcvolume/bin' /root/.bashrc || cat >> /root/.bashrc <<'RC'
export PATH=/pvcvolume/bin:$PATH
export HF_HOME=/pvcvolume/hf_cache
[ -f /pvcvolume/venv/bin/activate ] && source /pvcvolume/venv/bin/activate
[ -f /pvcvolume/secrets.env ] && source /pvcvolume/secrets.env
RC

echo "pod-init done - sshd up; PATH, HF_HOME, venv, secrets wired into ~/.bashrc"
