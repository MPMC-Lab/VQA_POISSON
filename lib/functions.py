from qiskit.circuit.library import QFT, Isometry
import numpy as np
from qiskit import transpile, QuantumCircuit

import numpy as np
from qiskit import transpile, ClassicalRegister, QuantumRegister, QuantumCircuit
from qiskit.circuit.library import StatePreparation, DiagonalGate, QFT

from scipy.stats import linregress


def create_parameterized_ansatz(num_qubits, ansatz_depth, parameters):
    qc = QuantumCircuit(num_qubits)
    for i in range (num_qubits):
        qc.ry(parameters[i], i)
    for depth in range (ansatz_depth):
        for i in range (0,num_qubits-1, 2):
            qc.cz(i, i+1)
        for i in range (num_qubits):
            qc.ry(parameters[num_qubits + depth * (2 * num_qubits - 2) + i], i)
        for i in range (1,num_qubits-1, 2):
            qc.cz(i, i+1)
        for i in range (1, num_qubits-1):
            qc.ry(parameters[2 * num_qubits + depth * (2 * num_qubits - 2) + (i-1)], qc.qubits[i])
    return qc

def QFT_LNN(qc, num_qubits, qubit_index):
    for idx in range (1, num_qubits):
        qc.rz(np.pi * ((2**(idx)-1)/2**(idx+1)), qc.qubits[num_qubits - 1 - idx + qubit_index])
    for block_number in range (num_qubits-1):
        qc.h(num_qubits - 1 - block_number + qubit_index)
        
        if block_number>=1 and block_number <= num_qubits-2:
            qc.cx(num_qubits - 1 - block_number + qubit_index, num_qubits - 2 - block_number + qubit_index)
        else:
            for idx in range (num_qubits - block_number-1):
                qc.cx(num_qubits - 1 - (num_qubits -2-idx) + qubit_index, num_qubits - 1 - (num_qubits -idx-1) + qubit_index)
        for idx in range (num_qubits - block_number-2):
            qc.cx(num_qubits - 1 - (idx + 1 + block_number) + qubit_index, num_qubits - 1 - (idx + 2 + block_number) + qubit_index)
        for idx in range (1, num_qubits - block_number):
            qc.rz(-np.pi / (2**(idx+1)), qc.qubits[num_qubits - 1 - (idx + block_number) + qubit_index])
        for idx in range (num_qubits - block_number-1):
            qc.cx(num_qubits - 1 - (num_qubits - 2 - idx) + qubit_index, num_qubits - 1 - (num_qubits - idx - 1) + qubit_index)
        if block_number <= num_qubits-3:
            qc.cx(num_qubits - 1 - (1 + block_number) + qubit_index, num_qubits -1 - (2 + block_number) + qubit_index)
        else:
            for idx in range (num_qubits - block_number-2):
                qc.cx(num_qubits - 1 - (idx+1 + block_number) + qubit_index, num_qubits - 1 - (idx+2 + block_number) + qubit_index)
        
    for idx in range (1, num_qubits):
        qc.rz(np.pi * ((2**(idx)-1)/2**(idx+1)), qc.qubits[num_qubits - 1 - (num_qubits - idx - 1) + qubit_index])

    qc.h(qubit_index)
    
    return qc

def fpsi_Choi(params, parameters, psi_param_gate, eta, num_qubits, simulator_backend, num_shots):
    
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
    
    qc_transpiled = transpile(qc, backend = hardware_backend, optimization_level = 3)
    qc_transpiled = qc_transpiled.assign_parameters({parameters[i]: params[i] for i in range (param_num)})
    
    counts = sampler.run([qc_transpiled]).result()[0].data.my_creg.get_counts()
    
    evs = (counts.get('0' * (num_qubits + 1), 0)) / num_shots
    return evs

def fpsi_arbitrary_simulator(RHS, params, parameters, num_qubits, ansatz_depth, simulator_backend, num_shots):
    qc = QuantumCircuit(num_qubits)
    for i in range (num_qubits):
        qc.ry(parameters[i], num_qubits - 1 - i)
    for depth in range (ansatz_depth):
        for i in range (0,num_qubits-1, 2):
            qc.cz(i, i+1)
        for i in range (num_qubits):
            qc.ry(parameters[num_qubits + depth * (2 * num_qubits - 2) + i], num_qubits - 1 - i)
        for i in range (1,num_qubits-1, 2):
            qc.cz(i, i+1)
        for i in range (1, num_qubits-1):
            qc.ry(parameters[2 * num_qubits + depth * (2 * num_qubits - 2) + (i-1)], qc.qubits[num_qubits - 1 - i])
    psi_param_gate = qc.to_gate()
    
    qc = QuantumCircuit(num_qubits)
    qc.append(Isometry(RHS, 0, 0), qc.qubits)
    qc.append(psi_param_gate.inverse(), qc.qubits)
    qc.measure_all()
    
    qc_transpiled = transpile(qc, backend = simulator_backend)
    qc_transpiled = qc_transpiled.assign_parameters({parameters[i]: params[i] for i in range (len(parameters))})
    
    counts = simulator_backend.run(qc_transpiled, shots = num_shots).get_counts()
    
    evs = (counts.get('0' * (num_qubits), 0)) / num_shots
    return evs
    
def bit_reverse_index(i, num_bits):
    rev = 0
    for _ in range(num_bits):
        rev = (rev << 1) | (i & 1)
        i >>= 1
    return rev

def bit_reverse_statevector(state, num_qubits):
    dim = 2 ** num_qubits
    new_state = np.zeros(dim, dtype=complex)
    for i in range(dim):
        new_state[bit_reverse_index(i, num_qubits)] = state[i]
    return new_state

def make_classical_psi(num_qubits, ansatz_depth, params, reverse = False):
    psi_list = []
    classical_CZ = np.array([[1,0,0,0],
                             [0,1,0,0],
                             [0,0,1,0],
                             [0,0,0,-1]], dtype=complex)
    CZ_barrier = classical_CZ
    for i in range (2, num_qubits, 2):
        CZ_barrier = np.kron(CZ_barrier, classical_CZ)
        
    CZ_barrier2 = I
    for i in range (1, num_qubits-1 , 2):
        CZ_barrier2 = np.kron(CZ_barrier2, classical_CZ)
    CZ_barrier2 = np.kron(CZ_barrier2, I)
    
    # Make the first layer of RY
    for qubit in range(0,num_qubits):
        psi = np.array([1,0], dtype = complex)
        psi = classical_RYGate(params[qubit]) @ psi.T
        psi_list.append(psi)
    psi = psi_list[0]
    for psis in psi_list[1:]:
        psi = np.kron(psi, psis)
    
    for depth in range (ansatz_depth):
        # First CZ barrier
        psi = CZ_barrier @ psi
        psi = classical_RYGate_nqubits(params[num_qubits + depth * (2 * num_qubits - 2): 2 * num_qubits + depth * (2 * num_qubits - 2)], num_qubits) @ psi
        psi = CZ_barrier2 @ psi
        psi = np.kron(np.kron(I, classical_RYGate_nqubits(params[2 * num_qubits + depth * (2 * num_qubits - 2): 3 * num_qubits - 2 + depth * (2 * num_qubits - 2)], num_qubits - 2)), I) @ psi
    
    if reverse == False:
        return psi
    else:
        return bit_reverse_statevector(psi, num_qubits)

I = np.array([[1,0],
             [0,1]], dtype = complex)

X = np.array([[0,1],
             [1,0]], dtype = complex)

def classical_RZGate(theta):
    return np.array([[np.exp(-1j * theta/2), 0],
                    [0, np.exp(1j * theta/2)]], dtype = complex)

def classical_RYGate(theta):
    return np.array([[np.cos(theta/2), -np.sin(theta/2)],
                    [np.sin(theta/2), np.cos(theta/2)]], dtype = complex)

def classical_RYGate_nqubits(params, num_qubits):
    RYGate = classical_RYGate(params[0])
    for qubit_num in range (num_qubits-1):
        RYGate = np.kron(RYGate, classical_RYGate(params[qubit_num+1]))
    return RYGate
        
def classical_RZGate_nqubits(params, num_qubits):
    RZGate = classical_RZGate(params[0])
    for qubit_num in range (num_qubits-1):
        RZGate = np.kron(RZGate, classical_RZGate(params[qubit_num+1]))
    return RZGate


def statepreparation(f):
    num_qubits = np.int64(np.log2(len(f)))
    qc = QuantumCircuit(num_qubits)
    qc.append(StatePreparation(f), qc.qubits)
    for i in range (np.int64(num_qubits//2)):
        qc.swap(i, num_qubits - i - 1)
    return qc.to_gate(label = r'$\left| f \right\rangle$')

def E1_matrix(n):
    L = np.zeros((n, n))
    for i in range(n - 1):
        L[i + 1, i] = 1
    L[0, n - 1] = 1
    return L

def E3_matrix(n):
    L = np.zeros((n, n))
    for i in range(n - 1):
        L[i, i+1] = 1
    L[n - 1, 0] = 1
    return L

def off_matrix(n):
    L = np.zeros((n, n))
    for i in range(n - 1):
        L[i + 1, i] = 1
    for i in range(n - 1):
        L[i, i+1] = 1
    return L
def return_count_array(aggregate_counts):
    count_array = []
    count_list = np.array([])
    for num_qubits, dictionary in aggregate_counts.items():
        count_list = np.array([])
        for ops, counts in dictionary.items():
            count_list = np.append(count_list, counts)
        count_array.append(count_list)
    return np.array(count_array)

def loglog_regression(x, y):
    log_x = np.log2(x)
    log_y = np.log10(y)
    slope, intercept, _, _, _ = linregress(log_x, log_y)
    return slope, intercept

def bit_reversal(n, num_bits):
    return int(bin(n)[2:].zfill(num_bits)[::-1], 2)

def bit_reversal_map(f):
    num_qubits_1D = np.int64(np.log2(len(f))/2)
    bit_reversed_f = np.zeros_like(f, dtype = complex)
    one_dim_grid_num = np.int64(np.sqrt(len(f)))
    for i in range (one_dim_grid_num):
        for j in range (one_dim_grid_num):
            bit_reversed_j = bit_reversal(j, num_qubits_1D)
            bit_reversed_f[i * one_dim_grid_num +j] = f[i * one_dim_grid_num + bit_reversed_j]
    return bit_reversed_f

def count_list(counts, array_size, num_shots, num_qubits, reverse = True):
    count_list = np.zeros(array_size)
    if reverse == True:
        for string, count in counts.items():
            # Convert the binary string to an integer
            index = int(string, 2)
            
            # Perform bit reversal
            bit_reversed_index = bit_reversal(index, num_qubits)
            
            # Update the count list
            count_list[bit_reversed_index] = np.sqrt(count / num_shots)
        
        return count_list
    else:
        for string, count in counts.items():
            # Convert the binary string to an integer
            index = int(string, 2)
            # Update the count list
            count_list[index] = np.sqrt(count / num_shots)
        return count_list
    
def laplacian_matrix(n, boundary_condition):
    L = np.zeros((n, n))
    for i in range (n):
        L[i,i] = -2
    for i in range(n - 1):
        L[i + 1, i] = 1
    for i in range(n - 1):
        L[i, i+1] = 1
    if boundary_condition == "Neumann":
        L[0,0] = -1
        L[n-1, n-1] = -1
    elif boundary_condition == "Periodic":        
        L[n - 1, 0] = 1
        L[0, n - 1] = 1
    return L

def one_dim_to_two_dim(f):
    one_dim_grid_num = np.int64(np.sqrt(len(f)))
    f2D = np.zeros((one_dim_grid_num, one_dim_grid_num), dtype = complex)
    for i in range (f2D.shape[0]):
        f2D[i] = f[i*one_dim_grid_num : (i+1) * one_dim_grid_num]
    return f2D

def bit_reversal(n, num_bits):
    return int(bin(n)[2:].zfill(num_bits)[::-1], 2)

def bit_reversal_map(f):
    num_qubits_1D = np.int64(np.log2(len(f))/2)
    bit_reversed_f = np.zeros_like(f, dtype = complex)
    one_dim_grid_num = np.int64(np.sqrt(len(f)))
    for i in range (one_dim_grid_num):
        for j in range (one_dim_grid_num):
            bit_reversed_j = bit_reversal(j, num_qubits_1D)
            bit_reversed_f[i * one_dim_grid_num +j] = f[i * one_dim_grid_num + bit_reversed_j]
    return bit_reversed_f

def count_list(counts, array_size, num_shots, num_qubits, reverse = True):
    count_list = np.zeros(array_size)
    if reverse == True:
        for string, count in counts.items():
            # Convert the binary string to an integer
            index = int(string, 2)
            
            # Perform bit reversal
            bit_reversed_index = bit_reversal(index, num_qubits)
            
            # Update the count list
            count_list[bit_reversed_index] = np.sqrt(count / num_shots)
        
        return count_list
    else:
        for string, count in counts.items():
            # Convert the binary string to an integer
            index = int(string, 2)
            # Update the count list
            count_list[index] = np.sqrt(count / num_shots)
        
        return count_list

def cost_function_classical(params, f, num_qubits1D):
    psi = make_classical_psi(params)
    A = np.kron(laplacian_matrix(2**num_qubits1D), np.eye(2**num_qubits1D)) + np.kron(np.eye(2**num_qubits1D), laplacian_matrix(2**num_qubits1D))
    return np.real(0.5 * ((np.inner(f, psi)**2) / (psi @ A @ psi)))
