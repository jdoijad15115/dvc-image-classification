#!/bin/bash
# EC2 user data (Amazon Linux 2023): runs once at first boot.
# Installs Docker and git, and DVC in an isolated virtualenv (a system-wide
# pip install would break the OS-packaged AWS CLI). Deployments are pushed by
# GitHub Actions through AWS Systems Manager (SSM) Run Command.
set -eux
dnf install -y docker git
systemctl enable --now docker
usermod -aG docker ec2-user
python3 -m venv /opt/dvc-venv
/opt/dvc-venv/bin/pip install --upgrade pip
/opt/dvc-venv/bin/pip install "dvc[s3]"
git clone https://github.com/jdoijad15115/dvc-image-classification.git /opt/image-classifier || true
