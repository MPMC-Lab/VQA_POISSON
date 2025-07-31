from lib.QuantumCalculator import LaplacianEVProcessor1D, InnerProductProcessor
from lib.QuantumOptimizer import VQA_PoissonOptimizer1D
from source_function_input import f_source
from lib.classical_functions import *
from lib.quantum_functions import *
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ionq import IonQProvider
from qiskit.circuit import ParameterVector

from qiskit_ibm_runtime import SamplerV2 as Sampler, SamplerOptions
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import yaml

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

# ================================
# Load Configuration
# ================================
print("Loading configuration from config.yaml...")
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

grid_num = config["grid_num"]
ansatz_depth = config["ansatz_depth"]
boundary_condition = config["boundary_condition"]
num_shots = config["num_shots"]
backend = config["backend"]

print(f"Configuration:")
print(f"  - grid_num: {grid_num}")
print(f"  - ansatz_depth: {ansatz_depth}")
print(f"  - boundary_condition: {boundary_condition}")
print(f"  - num_shots: {num_shots}")
print(f"  - backend: {backend}")

num_qubits = int(np.log2(grid_num))
param_num = num_qubits * ansatz_depth

x_grid = np.linspace(0,1,grid_num + 2, endpoint = True)
x_grid = x_grid[1:-1]
f_vector = np.vectorize(f_source)(x_grid)
f_normalized = f_vector / np.linalg.norm(f_vector)
dx = 1 / (grid_num + 1)

parameters = ParameterVector(r'$\boldsymbol{\theta}$', length=param_num)
psi_param_circuit = make_LNN_ansatz(num_qubits, ansatz_depth, parameters)

print("Parameterized circuit constructed.")
print(f"  - Number of qubits: {num_qubits}")
print(f"  - Number of parameters: {param_num}")

# ================================
# Backend Setup
# ================================
print("Initializing backends...")

if backend == 'simulator':
    # IonQ Simulator
    # os.environ['IONQ_API_KEY'] = 'your api key'
    provider = IonQProvider(os.getenv("IONQ_API_KEY"))
    simulator_backend = provider.get_backend("ionq_simulator", gateset = 'native')
    simulator_backend.set_options(noise_model="ideal")
    
    print(f"Using IonQ simulator: {simulator_backend.name()}")
    used_backend = simulator_backend
    is_simulator = True
    
    laplacian_processor = LaplacianEVProcessor1D(
        ansatz_list=[psi_param_circuit],
        boundary_condition_list=[boundary_condition],
        backend=used_backend,
        num_shots=num_shots,
        is_simulator = is_simulator,
    )

    numerator_processor = InnerProductProcessor(
        ansatz_list=[psi_param_circuit],
        numerator_list = [f_normalized],
        backend = used_backend,
        num_shots = num_shots,
        is_simulator = is_simulator,
    )
elif backend == 'hardware':
    
    # IBM Hardware
    hardware_backend = QiskitRuntimeService().least_busy()
    print(f"Using IBM hardware: {hardware_backend.name}")
    
    used_backend = hardware_backend
    options_sampler = SamplerOptions()
    options_sampler.default_shots = num_shots
    sampler = Sampler(mode = hardware_backend, options = options_sampler)
    is_simulator = False
    
    laplacian_processor = LaplacianEVProcessor1D(
        ansatz_list=[psi_param_circuit],
        boundary_condition_list=[boundary_condition],
        backend=used_backend,
        num_shots=num_shots,
        is_simulator = is_simulator,
        sampler = sampler
    )

    numerator_processor = InnerProductProcessor(
        ansatz_list=[psi_param_circuit],
        numerator_list = [f_normalized],
        backend = used_backend,
        num_shots = num_shots,
        is_simulator = is_simulator,
        sampler = sampler
    )
else:
    raise ValueError("Invalid backend specified in config.yaml. Choose either 'simulator' or 'hardware'.")

# ================================
# Optimization
# ================================
print("🚀 Starting optimization...")
optimizer = VQA_PoissonOptimizer1D(
    laplacian_processor = laplacian_processor,
    numerator_processor = numerator_processor,
    dx = dx,
    ansatz = psi_param_circuit
)

initial_params = np.random.rand(param_num) * 4 * np.pi
result = optimizer.optimize(initial_params = initial_params)

print("Optimization complete.")
print(f"  - Final cost value: {result.fun:.6f}")
print(f"  - Optimizer status: {result.message}")
print(f"  - Number of iterations: {result.nit}")

# ================================
# Post-processing
# ================================
optimal_params = result.x
optimal_state = make_classical_psi(num_qubits, ansatz_depth, optimal_params)

plt.figure(figsize = (6,6))
plt.plot(optimal_state, marker = 'o', label = 'VQA Solution')
plt.legend()
plt.show()