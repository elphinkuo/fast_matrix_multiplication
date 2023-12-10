#!/bin/bash

# Stop the script if any command fails
set -e

# Load SYCL/DPC++ environment (modify as needed for your setup)
# module load dpcpp

# Compile the SYCL program
dpcpp -std=c++17 -O3 matmul.cpp -o matmul

# Run the program
./matmul

echo "Matrix multiplication completed."
