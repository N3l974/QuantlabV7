#!/bin/bash

# Script to monitor diagnostic V5b and shutdown PC when complete
PID=178931

echo "Monitoring diagnostic V5b (PID: $PID)..."
echo "Will shutdown PC when process completes."

while kill -0 $PID 2>/dev/null; do
    echo "[$(date)] Diagnostic still running... (PID: $PID)"
    sleep 300  # Check every 5 minutes
done

echo "[$(date)] Diagnostic completed! Shutting down in 60 seconds..."
sleep 60

# Shutdown command
echo "Shutting down now..."
sudo shutdown -h now
