# VQA_POISSON_CUDAQ

This directory hosts the CUDA-Q implementation of the variational quantum algorithm (VQA) used in **VQA_POISSON**. It mirrors the hybrid quantum-classical workflow of the Python toolkit while targeting NVIDIA's CUDA Quantum stack (`nvq++`) to run SD and FD (LNN) methods of VQA_POISSON.

## Features

- **CUDA-Q kernels for Poisson solvers** – `VQA_FD_LNN.cpp` and `VQA_SD.cpp` implement the FD (LNN) and SD method presented in the parent README.
- **Reusable ansatz & QFT utilities** – `VQA_qpu.hpp` defines the variational ansätze preparation, controlled-phase primitives, and Park & Ahn (2023) inspired LNN QFT routine shared across kernels.
- **Shared classical helpers** – `include/` provides state preparation, Laplacian assembly, linear algebra utilities, and CSV IO helpers used by both the CUDA-Q executables and the classical baseline.
- **Classical baseline** – `classical_optimization/` reproduces the cost function in pure C++/NLopt for benchmarking against the quantum executions.
- **Logging & visualization** – execution traces, L2 errors, and trace distances are written under `log/` and plotted via `generate_history.py` into `history/` for side-by-side comparisons.

## Directory layout

```
CUDA_Q/
├── CMakeLists.txt                 # top-level build rules for CUDA-Q programs
├── VQA_FD_LNN.cpp                 # FD method of VQA_POISSON
├── VQA_SD.cpp                     # SD method of VQA_POISSON
├── VQA.hpp                        # shared declarations for classical helpers & IO
├── VQA_qpu.hpp                    # ansatz, inverse ansatz, QFT, numerator kernels
├── include/                       # reusable classical helper implementations
├── classical_optimization/        # COBYLA baseline optimizer & its build files
├── init/                          # initial parameter seeds used in the original paper (num_qubits_depth.csv)
├── log/                           # NLopt progress logs for CUDA-Q runs
├── history/                       # matplotlib figures generated from logs
├── build/                         # default build directory for CUDA-Q binaries
├── run.sh                         # convenience script to build & run everything
└── generate_history.py            # plots EVs / trace distance / L2 histories
```

## Requirements

- CUDA Quantum (`nvq++`) toolchain with an `nvidia` target available.
- CMake ≥ 3.10 and Ninja (or replace `-G "Ninja"` with your generator of choice).
- NLopt C library (set `NLOPT_INCLUDE_DIR` / `NLOPT_LIBRARY_DIR` for CMake or export before running `run.sh`).
- Python ≥ 3.8 with `numpy` and `matplotlib` for log visualization.

## Building

1. Ensure the NLopt paths are discoverable, e.g.
   ```bash
   export NLOPT_INCLUDE_DIR=/path/to/nlopt/include
   export NLOPT_LIBRARY_DIR=/path/to/nlopt/lib
   ```
2. Configure and build the CUDA-Q executables:
   ```bash
   cd CUDA_Q
   rm -rf CMakeCache.txt CMakeFiles
   cd build
   cmake -G "Ninja" ..
   cmake --build .
   cd ..
   ```
   This produces `program_FD_LNN.x` and `program_SD.x` in `CUDA_Q/build`.【F:CUDA_Q/CMakeLists.txt†L21-L36】
3. (Optional) Build the classical baseline:
   ```bash
   cd CUDA_Q
   cd classical_optimization
   rm -rf CMakeCache.txt CMakeFiles
   cd build
   cmake -G "Ninja" ..
   cmake --build .
   ```
   The classical executable `program.x` ends up under `classical_optimization/build/`.

The top-level `run.sh` automates all three steps, rebuilding from scratch before launching the binaries.

## Running the solvers

The CUDA-Q executables accept optional CLI arguments:

```bash
./program_FD_LNN.x [num_qubits] [ansatz_depth] [shots]
./program_SD.x [num_qubits] [ansatz_depth] [shots]
```

- Defaults are `num_qubits=9`, `ansatz_depth=4`, `shots=524288`.【F:CUDA_Q/VQA_FD_LNN.cpp†L156-L211】【F:CUDA_Q/VQA_SD.cpp†L200-L255】
- Each run loads initial parameters from `init/init_<num_qubits>_<depth>.csv`; ensure the file exists or provide your own seeds in that format.【F:CUDA_Q/VQA_FD_LNN.cpp†L220-L236】
- NLopt (COBYLA) termination criteria can be tuned inside the source (`nlopt_set_*` calls) if tighter tolerances are needed.【F:CUDA_Q/VQA_FD_LNN.cpp†L240-L272】

To compare against the classical baseline:

```bash
./classical_optimization/build/program.x
```

This binary reads the same initialization file, performs COBYLA entirely on the classical surrogate, and logs the cost, trace distance, and L2 error after each `(dim + 1)` evaluations.【F:CUDA_Q/classical_optimization/classical_optimization.cpp†L168-L240】

## Output & post-processing

- Quantum runs write `evs`, `trace`, and `l2` logs under `CUDA_Q/log/` with suffix `<num_qubits>_<shots>` to match the execution configuration.【F:CUDA_Q/VQA_FD_LNN.cpp†L272-L304】【F:CUDA_Q/VQA_SD.cpp†L272-L304】
- The classical optimizer mirrors the same metrics under `CUDA_Q/classical_optimization/log/`.【F:CUDA_Q/classical_optimization/classical_optimization.cpp†L204-L236】
- Execute `python3 generate_history.py` to generate the history plots in `CUDA_Q/history/`, overlaying SD, FD-LNN, and classical curves for easy benchmarking.【F:CUDA_Q/generate_history.py†L1-L74】

## License & citation

This directory inherits the MIT license and citation guidance from the parent VQA_POISSON repository. Please cite the references listed in the top-level README when publishing results based on this CUDA-Q implementation.

## Contact

For questions or collaboration inquiries, reach out to **achung3312@yonsei.ac.kr**, as in the main project README.
