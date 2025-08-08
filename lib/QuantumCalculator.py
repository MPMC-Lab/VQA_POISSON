from abc import ABC, abstractmethod
from typing import List, Union, Optional, Dict
from qiskit import QuantumCircuit
from qiskit.circuit import ClassicalRegister
from qiskit.circuit.library import StatePreparation
from qiskit_ibm_runtime import SamplerV2
from qiskit.providers.backend import Backend, BackendV2
import numpy as np
from typing import Optional
from lib.QuantumComputer import QuantumComputer
from lib.quantum_functions import QFT_LNN

class LaplacianEVProcessor1D(QuantumComputer):
    def __init__(self,
                 ansatz_list: List[QuantumCircuit],
                 boundary_condition_list: List[str],
                 dx_list: List[float],
                 backend: Union[Backend, BackendV2],
                 num_shots: int,
                 is_simulator: bool,
                 sampler: Optional[SamplerV2] = None,
                 alpha: Optional[float] = 0.0,
                 beta: Optional[float] = 0.0,
    ):
        """
        Abstract base class for computing Laplacian expectation values.

        Args:
            ansatz_list: List of parameterized ansatz (QuantumCircuit)
            boundary_condition_list: List of boundary conditions for each evs (Either 'P' or 'R')
            dx_list: List of grid sizes of the discretized domains.
            backend: Qiskit backend for simulation or hardware.
            num_shots: Number of shots per circuit execution.
            is_simulator: Flag indicating whether the backend is a simulator.
            sampler: Sampler object (only used for IBM hardware).
            
            alpha, beta, gamma_1, gamma_2 : Constants used for Robin boundary conditions (R) :
            \alpha U_0 + \beta \frac{\partial U_0}{\partial n} = \gamma_1
            \alpha U_N - \beta \frac{\partial U_N}{\partial n} = \gamma_2
        """
        num_qubits_list = []
        for ansatz in ansatz_list:
            num_qubits_list.append(ansatz.num_qubits)
        self.num_qubits_list = num_qubits_list
        self.boundary_condition_list = boundary_condition_list
        self.dx_list = dx_list
        self.alpha = alpha
        self.beta = beta
        
        # Determine how many circuits are needed per EV depending on BC type:.
        num_circuits_per_evs = []
        for bc, num_qubits in zip(boundary_condition_list, num_qubits_list):
            if bc == 'P':
                num_circuits_per_evs.append(1)
            elif (bc == 'R') and (self.beta == 0.0):
                num_circuits_per_evs.append(num_qubits)
            else:
                num_circuits_per_evs.append(num_qubits + 1)
            
        super().__init__(
            num_evs = len(boundary_condition_list),
            num_circuits_per_evs = num_circuits_per_evs,
            ansatz_list = ansatz_list,
            backend = backend,
            num_shots = num_shots,
            is_simulator = is_simulator,
            sampler = sampler,
        )
        

    def make_circuits(self) -> List[QuantumCircuit]:
        """
        Build quantum circuits to measure Laplacian expectation values
        for each ansatz under specified boundary conditions.
        
        P: Periodic condition (from Liu et al. (2025))
        R: Robin condition

        Returns:
            circuits: List of quantum circuits for all EV evaluations
        """
        
        ansatz_gate_list = [self.ansatz_list[i].to_gate() for i in range (self.num_evs)]
        circuits = []
        
        for circuit_idx in range (self.num_evs):
            ansatz_gate = ansatz_gate_list[circuit_idx]
            num_qubits = self.num_qubits_list[circuit_idx]
            
            if self.boundary_condition_list[circuit_idx] == 'R':
                
                for j in range (num_qubits):
                    qc = QuantumCircuit(num_qubits)
                    c = ClassicalRegister(num_qubits - j, 'my_creg')
                    qc.add_register(c)
                    qc.append(ansatz_gate, qc.qubits)
                    
                    for i in range (num_qubits - 1 - j):
                        qc.cx(i + 1, i)
                        
                    qc.h(num_qubits - 1 - j)
                    qc.measure([(index) for index in range (num_qubits - j)], c)
                    circuits.append(qc)
                
                if self.beta != 0.0:
                    c = ClassicalRegister(num_qubits, 'my_creg')
                    qc = QuantumCircuit(num_qubits)
                    qc.add_register(c)
                    qc.append(ansatz_gate, qc.qubits)
                    qc.measure([i for i in range (num_qubits)], c)
                    circuits.append(qc)
                    
            elif self.boundary_condition_list[circuit_idx] == 'P':
                qc = QuantumCircuit(num_qubits + 1)
                c = ClassicalRegister(1, 'my_creg')
                qc.add_register(c)
                
                qc.append(ansatz_gate, qc.qubits[1:])
                qc = QFT_LNN(qc, num_qubits, 1)
                qc.h(0)
                for idx in range(num_qubits):
                    # qc.cp((2 * np.pi / (2**num_qubits)) * (2**idx), 0, idx + 1)
                    qc.cp((2 * np.pi / (2**num_qubits)) * (2**idx), 0, num_qubits - idx)
                qc.h(0)
                qc.measure(0, c)
                circuits.append(qc)
            
        return circuits

    def make_evs(self, 
                 count_list : List[Dict[str, int]]
                ) -> List[float]:
        """
        Post-process measurement counts into expectation values.
        
        Args:
            count_list: List of count dictionaries (from simulator/hardware)
            
        Returns:
            qevs_list: Computed expectation values (float) for each problem
        """
        
        trace_idx = 0
        qevs_list = []
        
        for evs_idx in range (self.num_evs):
            num_qubits = self.num_qubits_list[evs_idx]
            dx = self.dx_list[evs_idx]
            
            qevs = 0
            
            if self.boundary_condition_list[evs_idx] == 'R':
                
                
                for j in range (num_qubits):
                    counts = count_list[trace_idx + j]
                    
                    if j == num_qubits - 1:
                        qevs += (counts.get('0', 0) - counts.get('1', 0)) / self.num_shots
                        
                    else:
                        qevs += (counts.get('0' + '1' + '0' * (num_qubits - j - 2), 0) - counts.get('1' + '1' + '0' * (num_qubits - j - 2), 0)) / self.num_shots
                
                if self.beta != 0.0:
                    coefficient = self.beta / (self.beta - self.alpha * dx)
                    counts = count_list[trace_idx + num_qubits]
                    correction = (counts.get('0' * num_qubits, 0) + counts.get('1' * num_qubits, 0)) / self.num_shots
                    qevs += correction * coefficient
                
            elif self.boundary_condition_list[evs_idx] == 'P':
                counts = count_list[trace_idx]
                qevs = 2 * (counts.get('0', 0) - counts.get('1', 0)) / self.num_shots
                
            qevs -= 2
            # qevs = qevs / (dx * dx)
            qevs_list.append(qevs)
            
            trace_idx += self.num_circuits_per_evs[evs_idx]
            
        return qevs_list
    
    
class InnerProductProcessor(QuantumComputer):
    """
    Processor for evaluating squared overlaps <num|psi(theta)>^2 via state preparation and inverse ansatz.
    Used to compute the numerator of variational energy estimators.
    """
    def __init__(self,
                 ansatz_list: List[QuantumCircuit],
                 numerator_list: List[np.ndarray],
                 backend: Union[Backend, BackendV2],
                 num_shots: int,
                 is_simulator: bool,
                 sampler: Optional[SamplerV2] = None,
    ):
        self.num_evs = len(numerator_list)
        
        num_qubits_list = []
        for ansatz in ansatz_list:
            num_qubits_list.append(ansatz.num_qubits)
        self.num_qubits_list = num_qubits_list
        self.numerator_list = numerator_list
        num_circuits_per_evs = [1] * self.num_evs
        
        super().__init__(
            num_evs = self.num_evs,
            num_circuits_per_evs = num_circuits_per_evs,
            ansatz_list = ansatz_list,
            backend = backend,
            num_shots = num_shots,
            is_simulator = is_simulator,
            sampler = sampler,
        )
        

    def make_circuits(self) -> List[QuantumCircuit]:
        
        ansatz_gate_list = [self.ansatz_list[i].to_gate() for i in range (self.num_evs)]
        circuits = []
        
        for circuit_idx in range (self.num_evs):
            ansatz_gate = ansatz_gate_list[circuit_idx]
            num_qubits = self.num_qubits_list[circuit_idx]
            numerator = self.numerator_list[circuit_idx]
            
            qc = QuantumCircuit(num_qubits)
            qc.append(StatePreparation(numerator), qc.qubits)
            qc.append(ansatz_gate.inverse(), qc.qubits)
            c = ClassicalRegister(num_qubits, 'my_creg')
            qc.add_register(c)
            qc.measure([i for i in range (num_qubits)], c)
            
            circuits.append(qc)
        return circuits

    def make_evs(self, count_list):
        qevs_list = []
        
        for evs_idx in range (self.num_evs):
            num_qubits = self.num_qubits_list[evs_idx]
            counts = count_list[evs_idx]
            qevs = (counts.get('0' * num_qubits, 0)) / (self.num_shots)
            qevs_list.append(qevs)
                
        return qevs_list