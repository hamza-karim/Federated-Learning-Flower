#!/bin/bash

read -p "Do you want to copy or delete? (copy/delete): " action

# Validate action
if [[ "$action" != "copy" && "$action" != "delete" ]]; then
  echo "❌ Invalid option. Please enter 'copy' or 'delete'."
  exit 1
fi

read -p "Is this a server or client? (server/client): " role

# Validate role
if [[ "$role" != "server" && "$role" != "client" ]]; then
  echo "❌ Invalid role. Please enter 'server' or 'client'."
  exit 1
fi

read -p "Enter ${role} number (e.g., 1, 2, 3...): " number
container_name="flwr-${role}${number}"
destination_dir="./${container_name}"

# Handle copy
if [ "$action" == "copy" ]; then
  container_id=$(sudo docker ps --filter "name=${container_name}" --format "{{.ID}}")

  if [ -z "$container_id" ]; then
    echo "❌ No running container found with name: $container_name"
    exit 1
  fi

  echo "✅ Found container $container_name (ID: $container_id)"
  mkdir -p "$destination_dir"

  files=(
    "combined_anomalies_plot.png"
    "combined_dataset_plot.png"
    "test_mape_histogram.png"
    "train_mape_histogram.png"
    "combined_MAPE_vs_threshold.png"
    "combined_confusion_matrix.png"
    "test_mae_histogram.png"
    "train_mae_histogram.png"
  )

  for file in "${files[@]}"; do
    echo "Copying $file..."
    if sudo docker cp "${container_id}:/app/src/${file}" "${destination_dir}/" 2>/dev/null; then
      echo "✅ Copied $file to ${destination_dir}/"
    else
      echo "⚠️  $file not found in container."
    fi
  done

  echo "✅ Copy operation completed for $container_name."

# Handle delete
elif [ "$action" == "delete" ]; then
  if [ -d "$destination_dir" ]; then
    read -p "Are you sure you want to delete the folder '$destination_dir'? (y/n): " confirm
    if [[ "$confirm" == "y" || "$confirm" == "Y" ]]; then
      rm -rf "$destination_dir"
      echo "🗑️  Deleted folder: $destination_dir"
    else
      echo "❎ Deletion cancelled."
    fi
  else
    echo "⚠️  Folder $destination_dir does not exist."
  fi
fi
