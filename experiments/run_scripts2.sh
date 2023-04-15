#!/bin/bash

# Define the list of Python scripts to run sequentially
scripts=(
    "A2C35.py"
    "PPO5Suburban.py"
    "PPO5City.py"
    "PPO15Suburban.py"
    "PPO15City.py"
    "A2C5Suburban.py"
)

# Iterate through the list of scripts and run them sequentially
for script in "${scripts[@]}"; do
  echo "Running $script"
  python "$script"
  echo "$script completed"
done

echo "All scripts executed successfully."



