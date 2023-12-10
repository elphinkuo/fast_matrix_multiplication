#!/bin/bash

# Stop the script if any command fails
set -e

# Define the Halide version
HALIDE_VERSION="release_2023_04_15"

# Install dependencies
echo "Installing dependencies..."
sudo apt-get update
sudo apt-get install -y g++ make libjpeg-dev libpng-dev

# Clone Halide repository
echo "Cloning Halide..."
git clone https://github.com/halide/Halide.git
cd Halide
git checkout $HALIDE_VERSION

# Build Halide
echo "Building Halide..."
make -j$(nproc)

# Go back to the parent directory
cd ..

# Halide program is named "matmul.cpp"
echo "Compiling your Halide program..."
g++ matmul.cpp -g -std=c++11 -I Halide/include -L Halide/bin -lHalide -ldl -lpthread -o matmul

# Run the program
echo "Running the program..."
./matmul

echo "Done."
