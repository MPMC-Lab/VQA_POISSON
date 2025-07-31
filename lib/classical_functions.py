"""
This script provides classical (NumPy-based) emulations of quantum state preparations and gate applications.
Primarily used for validating Qiskit-based quantum algorithms via statevector simulation.
"""

import numpy as np

I = np.array([[1,0],
             [0,1]], dtype = complex)

X = np.array([[0,1],
             [1,0]], dtype = complex)

classical_CNOT = np.array([[1,0,0,0],
                          [0,1,0,0],
                          [0,0,0,1],
                          [0,0,1,0]], dtype=complex)

def laplacian_matrix(n, boundary_condition):
    
    """
    Construct a finite-difference Laplacian matrix with specified boundary condition.

    Args:
        n: Size of the matrix (number of grid points).
        boundary_condition: 'Neumann' or 'Periodic'

    Returns:
        L: (n x n) NumPy array representing Laplacian matrix
    """

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

def CNOT_matrix(num_qubits, control_index):
    
    """
    Construct the full matrix of a CNOT gate acting on (control_index, control_index+1)
    in a multi-qubit system via Kronecker products.

    Note:
        Assumes control and target are adjacent.
    """
    
    if control_index != 0:
        matrix = np.array([[1,0],[0,1]], dtype = complex)
    else:
        matrix = 1
    for i in range (1, control_index):
        matrix = np.kron(matrix, I)
    matrix = np.kron(matrix, classical_CNOT)
    for i in range (control_index+2, num_qubits):
        matrix = np.kron(matrix, I)
    return matrix

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

def bit_reverse_index(i, num_bits):
    
    """
    Compute the bit-reversed version of index i
    """
    
    rev = 0
    for _ in range(num_bits):
        rev = (rev << 1) | (i & 1)
        i >>= 1
    return rev

def bit_reverse_statevector(state, num_qubits):
    
    """
    Reorder a statevector into bit-reversed indexing.
    """
    
    dim = 2 ** num_qubits
    new_state = np.zeros(dim, dtype=complex)
    for i in range(dim):
        new_state[bit_reverse_index(i, num_qubits)] = state[i]
    return new_state


def make_classical_psi(num_qubits, ansatz_depth, params):
    
    """
    Construct a multi-qubit quantum statevector using a layered RY-RZ-CNOT ansatz.

    Args:
        num_qubits: Number of qubits in the ansatz.
        ansatz_depth: Number of alternating RY-RZ-CNOT layers.
        params: Flat parameter list of length (num_qubits × depth)

    Returns:
        psi: Final statevector after applying the ansatz.
    """
    
    # Make the first layer of RY & RZ
    psi_list = []
    for qubit in range(0,num_qubits):
        psi = np.array([1,0], dtype = complex)
        psi = classical_RYGate(params[qubit]) @ psi.T
        psi_list.append(psi)
    psi = psi_list[0]
    for psis in psi_list[1:]:
        psi = np.kron(psi, psis)
    # First CNOT barrier
    for qubit in range (0, num_qubits - 1):
        psi = CNOT_matrix(num_qubits, qubit) @ psi
    
    for depth in range (1, ansatz_depth):
        psi = classical_RYGate_nqubits(params = [params[i] for i in range ((num_qubits)*depth , (num_qubits)*depth + num_qubits)], num_qubits=num_qubits) @ psi
        for qubit in range (0, num_qubits-1):
            psi = CNOT_matrix(num_qubits, qubit) @ psi
    return bit_reverse_statevector(psi, num_qubits)

def classical_cost_function_1D(num_qubits, ansatz_depth, params, f_normalized, dx, boundary_condition):
    
    cpsi = make_classical_psi(num_qubits, ansatz_depth, params)
    num = np.inner(cpsi, f_normalized)**2
    A = laplacian_matrix(f_normalized.size, boundary_condition)
    denom = cpsi @ A @ cpsi
    denom = denom / (dx**2)
    
    return (num / denom) * 0.5