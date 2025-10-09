#!/bin/bash
set -euo pipefail

# ---------------- Configuration ---------------- #
IMAGE="hamzakarim07/flwr_client_hfl:latest"
SERVER_IP="10.226.47.97"
SERVER_PORT="8080"
TOTAL_CLIENTS=20
EPOCHS=5
MODEL="lstm"
ALGO="fedavg"

# ---------------- Devices ---------------- #
# Format: [IP]="USERNAME START END"
declare -A DEVICES=(
  [10.226.47.97]="c2srnano07 1 2"
# [10.226.47.254]="c2srnano08 5 8"
# [10.226.47.102]="c2srnano09 9 12"
# [10.226.47.103]="c2srnano10 13 16"
# [10.226.47.105]="c2srnano11 17 20"
)

# ---------------- Deployment ---------------- #
for DEVICE_IP in "${!DEVICES[@]}"; do
  read USERNAME START END <<< "${DEVICES[$DEVICE_IP]}"
  echo "---------------------------------------------------------"
  echo "🔹 Deploying clients $START–$END on $DEVICE_IP ($USERNAME)"
  echo "---------------------------------------------------------"

  # Prepare the remote commands as a single block
  REMOTE_COMMANDS="set -e

echo 'Pulling latest image...'
docker pull $IMAGE >/dev/null

echo 'Removing any existing flwr-client containers...'
docker ps -a --filter 'name=flwr-client' --format '{{.Names}}' | xargs -r docker rm -f

for id in \$(seq $START $END); do
  cname=\"flwr-client\$id\"
  echo 'Starting new container:' \$cname
  docker run -d --name \$cname \
    --runtime=nvidia --gpus all \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e CLIENT_ID=\$id \
    -e TOTAL_CLIENTS=$TOTAL_CLIENTS \
    -e EPOCHS=$EPOCHS \
    -e MODEL=$MODEL \
    -e ALGO=$ALGO \
    -e SERVER_IP=$SERVER_IP \
    -e SERVER_PORT=$SERVER_PORT \
    $IMAGE >/dev/null

  echo '✅' \$cname 'started successfully.'
done"

  # SSH once and execute all commands
  ssh -o StrictHostKeyChecking=no $USERNAME@$DEVICE_IP "$REMOTE_COMMANDS" || \
    echo "⚠️ Failed to deploy clients on $DEVICE_IP"

done

echo "=========================================================="
echo "🎯 Deployment completed for all reachable devices."
echo "=========================================================="
