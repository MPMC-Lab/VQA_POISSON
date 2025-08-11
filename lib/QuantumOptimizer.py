from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ClassicalRegister
from typing import List, Union, Optional
import numpy as np
from scipy.optimize import minimize
from functools import partial

from lib.QuantumCalculator import LaplacianEVProcessor1D, LaplacianEVProcessor2D, InnerProductProcessor
from lib.classical_functions import *


class VQA_PoissonOptimizer:
    """
    Variational Quantum Algorithm Optimizer for 1D Poisson equation.
    
    This class handles the cost evaluation and classical optimization loop 
    using two quantum processors: LaplacianEVProcessor1D and InnerProductProcessor.
    """
    def __init__(
        self,
        laplacian_processor: Union[LaplacianEVProcessor1D, LaplacianEVProcessor2D],
        numerator_processor: InnerProductProcessor
    ):
        
        self.laplacian_processor = laplacian_processor
        self.numerator_processor = numerator_processor
        
    def cost_function(
        self,
        params: np.ndarray,
        optimization_level: int,
        is_simulator = True
    ) -> float:
        """
        Computes the variational energy for current parameters.

        Args:
            params: Parameters for the ansatz circuit.
            optimization_level: Transpiler optimization level.
            epsilon: Stability constant, not used for 'D' (homogeneous Dirichlet boundaries).
            is_simulator: Whether to run on simulator or hardware.

        Returns:
            Scalar cost value (float).
        """
        denom_circuit = self.laplacian_processor.make_circuits()
        transpiled = self.laplacian_processor.transpile(denom_circuit, params_list = [params], optimization_level = optimization_level)
        if is_simulator:
            counts = self.laplacian_processor.simulator_execute(transpiled)
        else:
            counts = self.laplacian_processor.hardware_execute(transpiled)
        
        bc = self.laplacian_processor.boundary_condition_list[0]
        
        denom = self.laplacian_processor.make_evs(counts)[0]
        
        num_circuit = self.numerator_processor.make_circuits()
        transpiled = self.numerator_processor.transpile(num_circuit, params_list = [params], optimization_level = optimization_level)
        if is_simulator:
            counts = self.numerator_processor.simulator_execute(transpiled)
        else:
            counts = self.numerator_processor.hardware_execute(transpiled)

        num = self.numerator_processor.make_evs(counts)[0]

        qevs = (num / denom) * 0.5
        
        return qevs
    
    def optimize(
        self,
        initial_params: np.ndarray,
        method: str = "COBYLA",
        optimization_level: int = 1,
        options = None,
        callback = None
    ):
        """
        Runs classical optimizer to minimize the cost function.

        Args:
            initial_params: Starting parameter vector.
            method: Optimization method (default 'COBYLA').
            epsilon: Stability constant for Neumann/Periodic.
            optimization_level: Qiskit transpile optimization level.
            options: Dictionary of optimizer-specific options.

        Returns:
            OptimizationResult object from scipy.optimize.
        """
        
        cost_fn = partial(self.cost_function, 
                          optimization_level=optimization_level,
                          is_simulator=self.laplacian_processor.is_simulator)
        
        result = minimize(cost_fn, 
                          initial_params, 
                          method=method, 
                          options=options,
                          callback = callback)
        
        return result
    
    def get_amplitudes(
        self,
        optimal_params: np.ndarray,
        optimization_level: int = 1,
    ) -> np.ndarray:
        
        """
        Executes the optimized ansatz circuit and extracts amplitude information
        from quantum measurement results.

        Args:
            optimal_params: The optimal parameters obtained after optimization.
            optimization_level: Transpiler optimization level to control circuit compilation.

        Returns:
            np.ndarray: A 1D array of square-rooted normalized measurement counts,
                        corresponding to the amplitudes of each computational basis state.
                        Bit-reversal is applied by default for qubit order correction.
        """
        
        num_qubits = self.laplacian_processor.num_qubits_list[0]
        grid_num = 2**num_qubits
        ansatz_gate = self.laplacian_processor.ansatz_list[0].to_gate()
        
        qc = QuantumCircuit(num_qubits)
        qc.append(ansatz_gate, qc.qubits)
        c = ClassicalRegister(num_qubits, 'my_creg')
        qc.add_register(c)
        
        qc.measure([i for i in range (num_qubits)], c)
    
        parameters = qc.parameters
        
        backend = self.laplacian_processor.backend
        
        qc_transpiled = transpile(qc, backend = backend, optimization_level = optimization_level)
        qc_transpiled = qc_transpiled.assign_parameters({parameters[i]: optimal_params[i] for i in range (len(parameters))})
        
        if self.laplacian_processor.is_simulator:
            counts = backend.run(qc_transpiled, shots = self.laplacian_processor.num_shots).get_counts()
        else:
            sampler = self.laplacian_processor.sampler
            job = sampler.run([qc_transpiled])
            print(f"  - Extracting amplitues of ansatz..")
            print(f"  - Job ID: {job.job_id()}")
            
            result = job.result()[0]
            counts = result.data.my_creg.get_counts()
        
        return count_list(counts, grid_num, self.laplacian_processor.num_shots, num_qubits, reverse = False)