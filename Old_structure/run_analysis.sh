#!/bin/bash

echo "Starting Chess Performance Analysis at $(date)"
echo "=========================================="

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Install requirements if needed
if [ -f "requirements.txt" ]; then
    echo "Installing Python requirements..."
    pip3 install -r requirements.txt
fi

# Create necessary directories
mkdir -p Data/raw_data Data/processed Data/summaries logs

# Run the analysis
echo "Starting chess data analysis..."
python3 main.py

echo "Analysis completed at $(date)"
echo "Check the Data/ folder for results"
echo "Check the logs/ folder for detailed logs"