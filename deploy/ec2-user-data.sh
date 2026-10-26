#!/bin/bash
# EC2 user data (Amazon Linux 2023): runs once at first boot.
# Installs Docker, git and DVC; first deployment is triggered by GitHub Actions via SSM.
set -eux
dnf install -y docker git python3-pip
systemctl enable --now docker
usermod -aG docker ec2-user
pip3 install "dvc[s3]"
git clone https://github.com/jdoijad15115/dvc-image-classification.git /opt/image-classifier
