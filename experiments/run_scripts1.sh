#!/bin/bash

# Define the list of Python scripts to run sequentially
scripts=(
    "PPO15.py"
    "PPO25.py"
    "PPO35.py"
    "A2C5.py"
    "A2C15.py"
    "A2C25.py"
)

# Iterate through the list of scripts and run them sequentially
for script in "${scripts[@]}"; do
  echo "Running $script"
  python "$script"
  echo "$script completed"
done

echo "All scripts executed successfully."




