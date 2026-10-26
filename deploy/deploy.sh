#!/usr/bin/env bash
# Runs ON the EC2 instance (invoked by GitHub Actions through AWS SSM Run Command).
# 1. sync code to the commit being deployed
# 2. fetch the release manifest (dvc.lock written by CI) and `dvc pull` the model from S3
# 3. rebuild the Docker image and restart the container on port 80
set -euo pipefail
export HOME=${HOME:-/root}   # SSM runs commands without HOME; git and dvc need it

APP_DIR=/opt/image-classifier
REPO_URL=${REPO_URL:-https://github.com/jdoijad15115/dvc-image-classification.git}
SHA=${1:-main}

echo "==> [1/4] sync code to $SHA"
if [ ! -d "$APP_DIR/.git" ]; then
  git clone "$REPO_URL" "$APP_DIR"
fi
cd "$APP_DIR"
git fetch --quiet origin
git checkout --quiet --force "$SHA"
BUCKET=$(grep -oP "url = s3://\K[^/]+" .dvc/config)

echo "==> [2/4] pull model from DVC remote s3://$BUCKET"
aws s3 cp "s3://$BUCKET/releases/$SHA/dvc.lock" dvc.lock --only-show-errors \
  || aws s3 cp "s3://$BUCKET/releases/latest/dvc.lock" dvc.lock --only-show-errors
dvc pull models/model.pkl
test -s models/model.pkl

echo "==> [3/4] docker build"
docker build -q -t image-classifier:"${SHA:0:7}" -t image-classifier:latest .

echo "==> [4/4] restart container on :80"
docker rm -f image-classifier >/dev/null 2>&1 || true
docker run -d --name image-classifier --restart unless-stopped \
  -p 80:5000 -e HOSTNAME="$(hostname)" image-classifier:latest
for i in $(seq 1 20); do
  if curl -fsS http://localhost/health >/dev/null; then
    echo "deployed ${SHA:0:7}: $(curl -fsS http://localhost/health)"
    docker image prune -f >/dev/null
    exit 0
  fi
  sleep 2
done
echo "health check failed"; docker logs image-classifier | tail -30; exit 1
