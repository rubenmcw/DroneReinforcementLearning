#!/bin/bash

# Define the list of Python scripts to run sequentially
scripts=(
    "A2C5City.py"
    "A2C15Suburban.py"
    "A2C15City.py"
    "A2C25Suburban.py"
    "A2C25City.py"
    "A2C35Suburban.py"
    "A2C35City.py"
)

# Iterate through the list of scripts and run them sequentially
for script in "${scripts[@]}"; do
  echo "Running $script"
  python "$script"
  echo "$script completed"
done

echo "All scripts executed successfully."
