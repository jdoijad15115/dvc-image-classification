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
VENV=/opt/dvc-venv
DVC=$VENV/bin/dvc

echo "==> [0/4] prerequisites"
# DVC lives in its own virtualenv: a system-wide `pip install dvc` upgrades shared
# Python packages and breaks the OS-packaged AWS CLI (No module named 'dateutil').
if [ ! -x "$DVC" ]; then
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install -q --upgrade pip
  "$VENV/bin/pip" install -q "dvc[s3]"
fi
"$DVC" --version

echo "==> [1/4] sync code to $SHA"
if [ ! -d "$APP_DIR/.git" ]; then
  git clone "$REPO_URL" "$APP_DIR"
fi
cd "$APP_DIR"
git fetch --quiet origin
git checkout --quiet --force "$SHA"
BUCKET=$(grep -oP "url = s3://\K[^/]+" .dvc/config)

echo "==> [2/4] pull model from DVC remote s3://$BUCKET"
# Release manifest = the dvc.lock CI produced for this commit, so EC2 pulls exactly the
# model that CI tested. Fetched with botocore from the DVC venv (uses the instance role).
"$VENV/bin/python" - "$BUCKET" "$SHA" <<'PY'
import sys, botocore.session
bucket, sha = sys.argv[1], sys.argv[2]
s3 = botocore.session.get_session().create_client("s3", region_name="ap-south-1")
for key in (f"releases/{sha}/dvc.lock", "releases/latest/dvc.lock"):
    try:
        body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    except s3.exceptions.NoSuchKey:
        continue
    open("dvc.lock", "wb").write(body)
    print(f"release manifest: s3://{bucket}/{key}")
    break
else:
    sys.exit("no release manifest found in S3")
PY
"$DVC" pull models/model.pkl
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
