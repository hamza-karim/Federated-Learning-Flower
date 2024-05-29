#!/bin/bash

# Function to copy files from container to host
copy_files() {
  local container_id=$1
  local destination_dir="."

  # List of files to copy
  local files=(
    "combined_anomalies_plot.png"
    "combined_dataset_plot.png"
    "test_mape_histogram.png"
    "train_mape_histogram.png"
    "combined_MAPE_vs_threshold.png"
    "combined_confusion_matrix.png"
    "test_mae_histogram.png"
    "train_mae_histogram.png"
  )

  # Copy each file from the container to the host
  for file in "${files[@]}"; do
    echo "Copying $file from container $container_id..."
    sudo docker cp "${container_id}:/app/src/${file}" "$destination_dir"
  done

  echo "All files copied successfully."
}

# Function to delete files from the host
delete_files() {
  local files=(
    "combined_anomalies_plot.png"
    "combined_dataset_plot.png"
    "test_mape_histogram.png"
    "train_mape_histogram.png"
    "combined_MAPE_vs_threshold.png"
    "combined_confusion_matrix.png"
    "test_mae_histogram.png"
    "train_mae_histogram.png"
  )

  # Delete each file from the host
  for file in "${files[@]}"; do
    if [ -f "$file" ]; then
      echo "Deleting $file..."
      yes | rm "$file"
    else
      echo "File $file does not exist, skipping..."
    fi
  done

  echo "All files deleted successfully."
}

# Check if the user provided an action argument
if [ $# -ne 1 ]; then
  echo "Usage: $0 <copy|delete>"
  exit 1
fi

action=$1

# Get the container ID of the flwr_client container
container_id=$(sudo docker ps --filter "ancestor=hamzakarim07/flwr_client:latest" --format "{{.ID}}")

# Check if the container ID is found
if [ -z "$container_id" ] && [ "$action" == "copy" ]; then
  echo "Container with image hamzakarim07/flwr_client:latest is not running."
  exit 1
fi

echo "Container ID: $container_id"

# Perform the requested action
case "$action" in
  copy)
    copy_files "$container_id"
    ;;
  delete)
    delete_files
    ;;
  *)
    echo "Invalid action: $action"
    echo "Usage: $0 <copy|delete>"
    exit 1
    ;;
esac