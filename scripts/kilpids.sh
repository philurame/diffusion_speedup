#!/bin/bash

file_numbers=("$@")

# If "all" is the first arg, use a fixed set of indices
if [[ "${1:-}" == "all" ]]; then
  file_numbers=(0 1 2 3 4 5 6 7)
fi

for i in "${file_numbers[@]}"; do
  file="logs/train_${i}.txt"

  if [[ -f "$file" ]]; then
    # Read only the FIRST line as the PID
    if IFS= read -r pid < "$file"; then
      # Trim whitespace
      pid="${pid//[[:space:]]/}"

      if [[ -n "$pid" && "$pid" =~ ^[0-9]+$ ]]; then
        if kill "$pid" 2>/dev/null; then
          echo "Killed $pid from $file"
        else
          echo "Failed to kill $pid (process may not exist or insufficient permissions)"
        fi
      else
        echo "No valid PID in first line of $file"
      fi
    else
      echo "Could not read $file"
    fi

    rm -f "$file"
  else
    echo "File $file not found"
  fi
done


# for pid in `cat pids/pid_output_*.txt`; do kill $pid; done
