#!/bin/bash

# Read the SSH config file
CONFIG_FILE="$HOME/.ssh/config"

# Extract HostName and User for the specified host
HOST="calan"
HOSTNAME=$(awk -v host="$HOST" '$1 == "Host" && $2 == host {found=1} found && $1 == "HostName" {print $2; exit}' "$CONFIG_FILE")
USER=$(awk -v host="$HOST" '$1 == "Host" && $2 == host {found=1} found && $1 == "User" {print $2; exit}' "$CONFIG_FILE")

# Check if HostName and User were found
if [[ -z "$HOSTNAME" || -z "$USER" ]]; then
    echo "Error: Could not find HostName or User for host '$HOST' in $CONFIG_FILE"
    exit 1
fi

# Construct the SCP command
SCP_COMMAND="scp -i ~/.ssh/transistor5.pem -r ${USER}@${HOSTNAME}:/home/ubuntu/joule/src/fairchem/core/datasets/worst_mae ~/Desktop/worst_errors"

# Execute the SCP command
echo "Executing: $SCP_COMMAND"
eval "$SCP_COMMAND"