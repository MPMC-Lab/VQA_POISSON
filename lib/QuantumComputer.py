from abc import ABC, abstractmethod
from typing import List, Union, Optional, Dict
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import SamplerV2
from qiskit.providers.backend import Backend, BackendV2
import numpy as np
from typing import Optional

class QuantumComputer(ABC):
    def __init__(
        self,
        num_evs: int,
        num_circuits_per_evs: List[int],
        ansatz_list: List[QuantumCircuit],
        backend: Union[Backend, BackendV2],
        num_shots: int,
        is_simulator: bool,
        sampler: Optional[SamplerV2] = None,
    ):
        """
        Abstract base class for QuantumComputer object.

        Args:
            num_evs: Number of expectation values the user wants to compute.
            num_circuits_per_evs: Number of circuits required to compute each evs (list)
            ansatz_list: List of parameterized ansatz (QuantumCircuit)
            backend: Qiskit backend for simulation or hardware.
            num_shots: Number of shots per circuit execution.
            is_simulator: Flag indicating whether the backend is a simulator.
            sampler: Sampler object (only used for IBM hardware).
        """
        self.num_evs: int = num_evs
        self.num_circuits_per_evs: List[int] = num_circuits_per_evs
        self.ansatz_list: List[QuantumCircuit] = ansatz_list
        self.backend: Union[Backend, BackendV2] = backend
        self.num_shots: int = num_shots
        self.is_simulator: bool = is_simulator
        self.sampler: Optional[SamplerV2] = sampler
        
        if not self.is_simulator and self.sampler is None:
            raise ValueError("For IBM hardware execution, sampler must be provided.")

        list_args = {
            "num_circuits_per_evs": self.num_circuits_per_evs,
            "ansatz_list": self.ansatz_list
        }

        for name, lst in list_args.items():
            if len(lst) != self.num_evs:
                raise ValueError(f"Expected {self.num_evs} elements in {name}, but got {len(lst)}.")

    def transpile(self,
                  qc_list: List[QuantumCircuit],
                  params_list: List[np.ndarray],
                  optimization_level: int = 1,
                  ) -> List[QuantumCircuit]:
        """
        Transpile each quantum circuit with assigned parameters for the target backend.

        Args:
            qc_list: List of quantum circuits to transpile.
            params_list: List of parameters to input
            optimization_level: Qiskit transpiler optimization level.

        Returns:
            List of transpiled and parameter-assigned quantum circuits.
        """
        transpiled_list = []
        if len(params_list) != self.num_evs:
            raise ValueError(f"The number of parameter sets does not match the number of expectation values: {len(params_list)} is not equal to {self.num_evs}.")
        qc_idx = 0
        for evs_idx in range (self.num_evs):
            params = params_list[evs_idx]
            num_circuits = self.num_circuits_per_evs[evs_idx]
            
            for circuit_idx in range (num_circuits):
                qc = qc_list[qc_idx + circuit_idx]
                pv = qc.parameters
                
                if len(params) != len(pv):
                    raise ValueError(f"Parameter length mismatch: expected {len(pv)} but got {len(params)} at evs index {evs_idx}.")

                qc_transpiled = transpile(qc, self.backend, optimization_level = optimization_level)
                qc_transpiled = qc_transpiled.assign_parameters({pv[j]: params[j] for j in range (len(pv))})
                transpiled_list.append(qc_transpiled)
            qc_idx += num_circuits
        
        return transpiled_list
        
    def hardware_execute(self,
                         transpiled_list: List[QuantumCircuit],
                        ) -> List[Dict[str, int]]:
        sampler = self.sampler
        job = sampler.run(transpiled_list, shots = self.num_shots)
        print(f"Hardware Job ID: {job.job_id()}")
        
        result_list = job.result()
        count_list = [result_list[i].data.my_creg.get_counts() for i in range (len(transpiled_list))]
        
        return count_list
        
    def simulator_execute(self,
                          transpiled_list: List[QuantumCircuit],
                         ) -> List[Dict[str, int]]:
        backend = self.backend
        count_list = [backend.run(transpiled_list[i], shots = self.num_shots).get_counts() for i in range (len(transpiled_list))]
        
        return count_list
    
    @abstractmethod
    def make_circuits(self):
        """
        Abstract method to construct quantum circuits required to compute the specified expectation values.
        
        Must be implemented in subclasses.
        """
        pass
    
    @abstractmethod
    def make_evs(self, count_list):
        """
        Abstract method to construct the expectation value from the count_list in _execute method.
        
        Must be implemented in subclasses.
        """
        pass
    