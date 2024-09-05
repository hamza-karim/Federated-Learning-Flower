#!/bin/bash

log_file="tegra_stats_log.csv"
echo "time,RAM_usage,SWAP_usage,CPU_usage,EMC_FREQ,GR3D_FREQ,GPU_temp,CPU_temp" > $log_file

# Loop to collect data every 5 seconds
while true; do
    current_time=$(date "+%Y-%m-%d %H:%M:%S")

    # Run tegrastats
    tegrastats_output=$(tegrastats --interval 1000 | head -n 1)

    # Print
    echo "Tegrastats output: $tegrastats_output"

    # Extract the desired information
    ram_usage=$(echo "$tegrastats_output" | grep -oP 'RAM \K[0-9]+/[0-9]+MB')
    swap_usage=$(echo "$tegrastats_output" | grep -oP 'SWAP \K[0-9]+/[0-9]+MB')
    cpu_usage=$(echo "$tegrastats_output" | grep -oP 'CPU \[\K[^\]]+')
    emc_freq=$(echo "$tegrastats_output" | grep -oP 'EMC_FREQ \K[0-9]+%')
    gr3d_freq=$(echo "$tegrastats_output" | grep -oP 'GR3D_FREQ \K[0-9]+%')
    gpu_temp=$(echo "$tegrastats_output" | grep -oP 'GPU@\K[0-9.]+C')
    cpu_temp=$(echo "$tegrastats_output" | grep -oP 'CPU@\K[0-9.]+C')

    echo "Extracted data: $current_time,$ram_usage,$swap_usage,$cpu_usage,$emc_freq,$gr3d_freq,$gpu_temp,$cpu_temp"

    # Log the data
    echo "$current_time,$ram_usage,$swap_usage,$cpu_usage,$emc_freq,$gr3d_freq,$gpu_temp,$cpu_temp" >> $log_file

    # Wait 5 seconds
    sleep 5
done