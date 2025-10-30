#!/bin/bash

# Build CUDA-Q kernels
rm -rf CMakeCache.txt CMakeFiles
cd build
cmake -G "Ninja" ..
cmake --build .
cd ..

# Build Classical one
cd classical_optimization
rm -rf CMakeCache.txt CMakeFiles
cd build
cmake -G "Ninja" ..
cmake --build .
./program.x

cd ..
cd ..
cd build

./program_FD_LNN.x 7 3 1048576
./program_SD.x 7 3 1048576

cd ..

python3 generate_history.py