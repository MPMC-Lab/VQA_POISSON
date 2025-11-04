import numpy as np
from qiskit import QuantumCircuit

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

def make_LNN_ansatz(num_qubits, ansatz_depth, parameters):
    qc = QuantumCircuit(num_qubits)
    for depth in range (ansatz_depth):
        for i in range(0,num_qubits):
            qc.ry(parameters[i + depth * (num_qubits)],  i)
        for i in range (0, num_qubits-1):
            qc.cx( i,  (i+1))

    return qc

def permutation_gate(qc, n, start_index=0):
    
    ancilla_last_index = (2 * n - 3) - n
    num_qubits = qc.num_qubits
    def map_index(idx):
        if idx >= ancilla_last_index:
            idx = idx + start_index
            return num_qubits - 1 - idx
        
        return num_qubits - 1 - idx

    if n == 2:
        qc.cx(map_index(1), map_index(0))
        qc.x(map_index(1))
        return qc
    elif n == 3:
        qc.ccx(map_index(2), map_index(1), map_index(0))
        qc.cx(map_index(2), map_index(1))
        qc.x(map_index(2))
        return qc
    elif n == 4:
        qc.ccx(map_index(4), map_index(3), map_index(0))
        qc.ccx(map_index(0), map_index(2), map_index(1))
        qc.ccx(map_index(4), map_index(3), map_index(0))
        qc.ccx(map_index(4), map_index(3), map_index(2))
        qc.cx(map_index(4), map_index(3))
        qc.x(map_index(4))
        return qc
    else:
        for j in range(5, n+1):
            offset = (n + 1) - j
            qc.ccx(map_index(2*n - 4), map_index(2*n - 5), map_index(2*n - 5 - (n-1)))

            for i in range(offset):
                qc.ccx(map_index(2*n - 6 - i),
                       map_index(2*n - 5 - i - (n-1)),
                       map_index(2*n - 6 - i - (n-1)))

            qc.ccx(map_index(2*n - 6 - offset),
                   map_index(2*n - 7 - (n-3) - offset),
                   map_index(2*n - 7 - offset))

            for i in range(offset-1, -1, -1):
                qc.ccx(map_index(2*n - 6 - i),
                       map_index(2*n - 5 - i - (n-1)),
                       map_index(2*n - 6 - i - (n-1)))

            qc.ccx(map_index(2*n - 4),
                   map_index(2*n - 5),
                   map_index(2*n - 5 - (n-1)))

        qc.ccx(map_index(2*n - 4), map_index(2*n - 5), map_index(2*n - 5 - (n-1)))
        qc.ccx(map_index(2*n - 6), map_index(2*n - 7 - (n-3)), map_index(2*n - 7))
        qc.ccx(map_index(2*n - 4), map_index(2*n - 5), map_index(2*n - 5 - (n-1)))

        qc.ccx(map_index(2*n - 4), map_index(2*n - 5), map_index(2*n - 6))
        qc.cx(map_index(2*n - 4), map_index(2*n - 5))
        qc.x(map_index(2*n - 4))
        return qc


