from abc import ABC, abstractmethod
from typing import List, Union, Optional
from qiskit import QuantumCircuit
from qiskit.circuit import ClassicalRegister
from qiskit.circuit.library import StatePreparation
from qiskit_ibm_runtime import SamplerV2
from qiskit.providers.backend import Backend, BackendV2
import numpy as np
from typing import Optional
from lib.QuantumComputer import QuantumComputer
from functions import QFT_LNN

class LaplacianEVProcessor1D(QuantumComputer):
    def __init__(self,
                 ansatz_list: List[QuantumCircuit],
                 boundary_condition_list: List[str],
                 backend: Union[Backend, BackendV2],
                 num_shots: int,
                 is_simulator: bool,
                 params_list: List[np.ndarray],
                 sampler: Optional[SamplerV2] = None,
    ):
        """
        Abstract base class for computing Laplacian expectation values.

        Args:
            params_list: List of parameter vectors (each for one circuit).
            ansatz_list: List of parameterized ansatz (QuantumCircuit)
            boundary_condition_list: List of boundary conditions for each evs (Either 'P', 'N', or 'D')
            backend: Qiskit backend for simulation or hardware.
            num_shots: Number of shots per circuit execution.
            is_simulator: Flag indicating whether the backend is a simulator.
            dim: Dimension of problem.
            sampler: Sampler object (only used for IBM hardware).
        """
        num_qubits_list = []
        for ansatz in ansatz_list:
            num_qubits_list.append(ansatz.num_qubits)
        self.num_qubits_list = num_qubits_list
        self.boundary_condition_list = boundary_condition_list
        num_circuits_per_evs = []
        
        for bc, num_qubits in zip(boundary_condition_list, num_qubits_list):
            if bc == 'P':
                num_circuits_per_evs.append(1)
            elif bc == 'N':
                num_circuits_per_evs.append(num_qubits + 1)
            elif bc == 'D':
                num_circuits_per_evs.append(num_qubits)
            else:
                raise ValueError("Boundary condition must be either 'N', 'P', or 'D'")
            
        super().__init__(
            num_evs = len(boundary_condition_list),
            num_circuits_per_evs = num_circuits_per_evs,
            params_list = params_list,
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
            
            if self.boundary_condition_list[circuit_idx] == 'D':
                
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
            
            elif self.boundary_condition_list[circuit_idx] == 'N':
                for j in range (num_qubits):
                    c = ClassicalRegister(num_qubits - j, 'my_creg')
                    
                    qc = QuantumCircuit(num_qubits)
                    qc.append(ansatz_gate, qc.qubits)
                    
                    qc.add_register(c)
                    
                    for i in range (num_qubits - 1 - j):
                        qc.cx(i + 1, i)
                    
                    qc.h(num_qubits - 1 - j)
                    qc.measure([(index) for index in range (num_qubits - j)], c)
                    circuits.append(qc)
                    
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

    def make_evs(self, count_list):
        trace_idx = 0
        qevs_list = []
        
        for evs_idx in range (self.num_evs):
            num_qubits = self.num_qubits_list[evs_idx]
            qevs = 0
            
            if self.boundary_condition_list[evs_idx] == 'D':
                for j in range (num_qubits):
                    counts = count_list[trace_idx + j]
                    
                    if j == num_qubits - 1:
                        qevs += (counts.get('0', 0) - counts.get('1', 0)) / self.num_shots
                    else:
                        qevs += (counts.get('0' + '1' + '0' * (num_qubits - j - 2), 0) - counts.get('1' + '1' + '0' * (num_qubits - j - 2), 0)) / self.num_shots
                
            elif self.boundary_condition_list[evs_idx] == 'N':
                for j in range (num_qubits + 1):
                    counts = count_list[trace_idx + j]
                    
                    if j == num_qubits - 1:
                        qevs += (counts.get('0', 0) - counts.get('1', 0)) / self.num_shots
                    elif j == num_qubits:
                        qevs += (counts.get('0' * num_qubits, 0) + counts.get('1' * num_qubits, 0)) / self.num_shots
                    else:
                        qevs += (counts.get('0' + '1' + '0' * (num_qubits - j - 2), 0) - counts.get('1' + '1' + '0' * (num_qubits - j - 2), 0)) / self.num_shots
                
            elif self.boundary_condition_list[evs_idx] == 'P':
                counts = count_list[trace_idx]
                qevs = 2 * (counts.get('0', 0) - counts.get('1', 0)) / self.num_shots
                
            qevs -= 2
            qevs_list.append(qevs)
            trace_idx += self.num_circuits_per_evs[evs_idx]
                
        return qevs_list
    
class InnerProductProcessor(QuantumComputer):
    def __init__(self,
                 ansatz_list: List[QuantumCircuit],
                 numerator_list: List[np.ndarray],
                 backend: Union[Backend, BackendV2],
                 num_shots: int,
                 is_simulator: bool,
                 params_list: List[np.ndarray],
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
            params_list = params_list,
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
            c = ClassicalRegister(num_qubits)
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