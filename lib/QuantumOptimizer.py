from qiskit import QuantumCircuit
from typing import List, Union, Optional
import numpy as np
from scipy.optimize import minimize
from functools import partial

from lib.QuantumCalculator import LaplacianEVProcessor1D, InnerProductProcessor
from lib.classical_functions import *


class VQA_PoissonOptimizer1D:
    """
    Variational Quantum Algorithm Optimizer for 1D Poisson equation.
    
    This class handles the cost evaluation and classical optimization loop 
    using two quantum processors: LaplacianEVProcessor1D and InnerProductProcessor.
    """
    def __init__(
        self,
        laplacian_processor: LaplacianEVProcessor1D,
        numerator_processor: InnerProductProcessor,
        dx : float,
        ansatz: QuantumCircuit
    ):
        
        self.laplacian_processor = laplacian_processor
        self.numerator_processor = numerator_processor
        self.dx = dx
        self.ansatz = ansatz
    
        
    def cost_function(
        self,
        params: np.ndarray,
        optimization_level: int,
        epsilon : float = 0.001,
        is_simulator = True
    ) -> float:
        """
        Computes the variational energy for current parameters.

        Args:
            params: Parameters for the ansatz circuit.
            optimization_level: Transpiler optimization level.
            epsilon: Stability constant (only for 'P' and 'N' BCs).
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
        
        denom = self.laplacian_processor.make_evs(counts)[0] / ((self.dx) ** 2)
        
        bc = self.laplacian_processor.boundary_condition_list[0]
        if bc == 'P' or bc == 'N':
            denom = denom + epsilon
        
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
        epsilon: float = 0.001,
        optimization_level: int = 1,
        is_simulator: bool = True,
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
            is_simulator: Flag to select simulator/hardware.
            options: Dictionary of optimizer-specific options.

        Returns:
            OptimizationResult object from scipy.optimize.
        """
        
        cost_fn = partial(self.cost_function, 
                          optimization_level=optimization_level, 
                          epsilon = epsilon, 
                          is_simulator=is_simulator)
        
        result = minimize(cost_fn, 
                          initial_params, 
                          method=method, 
                          options=options,
                          callback = callback)
        
        return result
