import numpy as np

from qiskit.circuit.library import Isometry
from qiskit import transpile, QuantumCircuit, ClassicalRegister

def fpsi_Choi_simulator(params, parameters, psi_param_gate, eta, num_qubits, simulator_backend, num_shots):
    
    param_num = params.shape[0]
    num_qubits1D = np.int64(num_qubits/2)
    
    qc = QuantumCircuit(num_qubits+1)
    qc.h(0)
    qc.append(psi_param_gate, qc.qubits[1:])
    qc.h(num_qubits1D)
    for i in range (num_qubits1D, num_qubits + 1):
        qc.ch(0,i)
    for i in range (1, num_qubits1D):
        qc.ch(0,i)
    qc.cx(0,2)
    qc.cx(0,3)
    qc.x(0)
    qc.cx(1,3)
    qc.cx(1,2)
    qc.x(2)
    qc.x(3)
    qc.ch(0,1)
    qc.x(0)
    qc.ry(-2 * eta, 0)
    qc.measure_all()
    
    qc_transpiled = transpile(qc, backend = simulator_backend)
    qc_transpiled = qc_transpiled.assign_parameters({parameters[i]: params[i] for i in range (param_num)})
    
    counts = simulator_backend.run(qc_transpiled, shots = num_shots).get_counts()
    
    evs = (counts.get('0' * (num_qubits + 1), 0)) / num_shots
    return evs

def fpsi_Choi_hardware(params, parameters, psi_param_gate, eta, num_qubits, hardware_backend, sampler, num_shots):
    param_num = params.shape[0]
    num_qubits1D = np.int64(num_qubits/2)
    
    qc = QuantumCircuit(num_qubits+1)
    c = ClassicalRegister(num_qubits + 1, 'my_creg')
    qc.h(0)
    qc.append(psi_param_gate, qc.qubits[1:])
    qc.h(num_qubits1D)
    for i in range (num_qubits1D, num_qubits + 1):
        qc.ch(0,i)
    for i in range (1, num_qubits1D):
        qc.ch(0,i)
    qc.cx(0,2)
    qc.cx(0,3)
    qc.x(0)
    qc.cx(1,3)
    qc.cx(1,2)
    qc.x(2)
    qc.x(3)
    qc.ch(0,1)
    qc.x(0)
    qc.ry(-2 * eta, 0)
    qc.measure([i for i in range (num_qubits + 1)], c)
    
    qc_transpiled = transpile(qc, backend = hardware_backend, optimization_level = 3)
    qc_transpiled = qc_transpiled.assign_parameters({parameters[i]: params[i] for i in range (param_num)})
    
    counts = sampler.run([qc_transpiled]).result()[0].data.my_creg.get_counts()
    
    evs = (counts.get('0' * (num_qubits + 1), 0)) / num_shots
    return evs

def fpsi_arbitrary_simulator(RHS, params, parameters, psi_param_gate, num_qubits, simulator_backend, num_shots):
    qc = QuantumCircuit(num_qubits)
    qc.append(Isometry(RHS, 0, 0), qc.qubits)
    qc.append(psi_param_gate.inverse(), qc.qubits)
    qc.measure_all()
    
    qc_transpiled = transpile(qc, backend = simulator_backend)
    qc_transpiled = qc_transpiled.assign_parameters({parameters[i]: params[i] for i in range (len(parameters))})
    
    counts = simulator_backend.run(qc_transpiled, shots = num_shots).get_counts()
    
    evs = (counts.get('0' * (num_qubits), 0)) / num_shots
    return evs
    
def fpsi_arbitrary_hardware(RHS, params, parameters, psi_param_gate, num_qubits, hardware_backend, sampler, num_shots):
    qc = QuantumCircuit(num_qubits)
    c = ClassicalRegister(num_qubits, 'my_creg')
    qc.add_register(c)
    
    qc.append(Isometry(RHS, 0, 0), qc.qubits)
    qc.append(psi_param_gate.inverse(), qc.qubits)
    qc.measure([i for i in range (num_qubits)], c)
    
    qc_transpiled = transpile(qc, backend = hardware_backend, optimization_level = 1)
    qc_transpiled = qc_transpiled.assign_parameters({parameters[i]: params[i] for i in range (len(parameters))})
    
    counts = sampler.run([qc_transpiled]).result()[0].data.my_creg.get_counts()
    
    evs = (counts.get('0' * (num_qubits), 0)) / num_shots
    return evs
