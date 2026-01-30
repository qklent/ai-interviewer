#!/bin/bash

# Check if USE_EXAMPLE_SCRIPT environment variable is set to "true"
if [ "$USE_EXAMPLE_SCRIPT" = "true" ]; then
    echo "Running with example_script.txt..."
    exec python main.py example_script.txt
else
    # Run in interactive mode or with provided arguments
    exec python main.py "$@"
fi
