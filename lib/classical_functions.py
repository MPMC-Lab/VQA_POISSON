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

def laplacian_matrix(n, boundary_condition, alpha = 0.0, beta = 0.0, dx = 0.0):
    
    """
    Construct a finite-difference Laplacian matrix with specified boundary condition.

    Args:
        n: Size of the matrix (number of grid points).
        boundary_condition: "P" or "R"

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
    if boundary_condition == "P":        
        L[n - 1, 0] = 1
        L[0, n - 1] = 1
    elif boundary_condition == "R":
        if beta != 0.0:
            L[0,0] += (beta / dx) / ((beta / dx) - alpha)
            L[n - 1, n - 1] += (beta / dx) / ((beta / dx) - alpha)
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

def bit_reversal(n, num_bits):
    """
    Performs bit reversal on the integer `n` with `num_bits` total bits.
    For example, if n=6 (110) and num_bits=3, the result is 3 (011).
    """
    return int(bin(n)[2:].zfill(num_bits)[::-1], 2)

def count_list(counts, array_size, num_shots, num_qubits, reverse = True):
    
    """
    Converts a Qiskit-style `counts` dictionary (e.g., {'001': 10, '010': 12, ...})
    into a normalized numpy array of square-rooted probabilities.

    Parameters:
    -----------
    counts : dict
        Dictionary of bitstrings (str) → counts (int) from measurement results.

    array_size : int
        Size of the output array. Typically 2^n for n qubits.

    num_shots : int
        Total number of measurement shots used in the experiment.

    num_qubits : int
        Number of qubits used in the circuit. Used for bit reversal.

    reverse : bool (default=True)
        Whether to apply bit reversal on indices (useful if qubit order is reversed).

    Returns:
    --------
    count_list : np.ndarray
        A 1D array of size `array_size`, with each entry containing
        sqrt(probability) for the corresponding basis state (after optional bit reversal).
    """
    
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
    
def check_stability(A, num_shots, C = 10.0):
    """
    Assess the numerical stability of a matrix A under sampling-based quantum optimization.
    If the smallest absolute eigenvalue is below a threshold determined by C / sqrt(num_shots),
    a warning is issued indicating potential instability due to sampling noise.
    """

    eigenvalues = np.linalg.eigvalsh(A)
    lambda_min = np.min(np.abs(eigenvalues))

    threshold = C / np.sqrt(num_shots)

    if lambda_min < threshold:
        print(f"[Warning] The system matrix may be ill-conditioned for sampling-based optimization.")
        print(f"          Minimum |<ψ|A|ψ>| = {lambda_min:.3e} < {threshold:.3e} = (C / sqrt({num_shots}))")
        print(f"          This may lead to instability or unreliable convergence due to sampling noise. Check your boundary conditions or increase the number of shots.\n")
        return True
    
    else:
        print(f"  - Matrix is well-conditioned. Minimum |<ψ|A|ψ>| = {lambda_min:.3e}\n")
        return False