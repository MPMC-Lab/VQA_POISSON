#pragma once
#include <cudaq.h>
#include <vector>

// -----------------------------------------------------------------------------
// Variational ansatz utilities for CUDA-Q based Poisson VQA experiments.
// This file provides:
//   * A lightweight LNN-style parameterized circuit (forward + inverse).
//   * A controlled-phase helper expressed via CRZ decomposition.
//   * A QFT implementation constrained to linear-nearest-neighbor connectivity.
//   * A numerator helper that prepares the trial state, applies the adjoint
//     ansatz, and measures in the computational basis.
// The routines are written as __qpu__ kernels to run directly on CUDA-Q targets.
// -----------------------------------------------------------------------------

inline __qpu__ void variational_ansatz(cudaq::qvector<>& qubits,
                                      int ansatz_depth, 
                                      std::vector<double>& parameters, 
                                      int start, int n, 
                                      bool reverse = false) {
  // Apply a layered RY + CX ansatz to a contiguous block of qubits.
  // - start: starting qubit index into the shared register
  // - n: number of qubits in the block
  // - parameters: angles, consumed in-order per layer
  // - reverse: when true, traverse the block from high to low indices
  
  int param_idx = 0;
  
  for (int i = 0; i < n; ++i) {
    h(qubits[start + i]);
  }

  if (!reverse) {
    
    for (int depth = 0; depth < ansatz_depth; ++depth) {
      for (int i = 0; i < n; ++i) {
        ry(parameters[param_idx++], qubits[start + i]);
      }
      for (int i = 0; i < n - 1; i++) {
        cx(qubits[start + i], qubits[start + (i+1)]);
      }
    }
  } else {
    for (int depth = 0; depth < ansatz_depth; ++depth) {
      for (int i = 0; i < n; ++i) {
        ry(parameters[param_idx++], qubits[start + (n - 1 - i)]);
      }
      // Reverse-direction chain of CX gates to respect LNN ordering.
      for (int i = 0; i < n - 1; i++) {
        cx(qubits[start + n - 1 - i], qubits[start + n - 1 - (i+1)]);
      }
    }
  }
}

inline __qpu__ void variational_ansatz_inverse(cudaq::qvector<>& qubits,
                                               int ansatz_depth,
                                               std::vector<double>& parameters,
                                               int start, int n,
                                               bool reverse = false) {
  // Adjoint of the variational ansatz. Applies gates in reverse order with
  // inverted angles to uncompute the state prepared by variational_ansatz.
  int param_idx = ansatz_depth * n;

  if (!reverse) {
    for (int depth = 0; depth < ansatz_depth; ++depth) {
      param_idx -= n;
      // CX gate: apply in reverse order
      for (int i = n - 2; i >= 0; --i) {
        cx(qubits[start + i], qubits[start + (i + 1)]);
      }
      // RY gate: reverse order, and negate angle
      for (int i = n - 1; i >= 0; --i) {
        ry(-parameters[param_idx + i], qubits[start + i]);
      }
    }
    for (int i = 0; i < n; ++i) {
      h(qubits[start + i]);
    }
    
    
  } else {
    for (int depth = ansatz_depth - 1; depth >= 0; --depth) {
      param_idx -= n;
      // Reverse-direction CX ladder for the mirrored traversal.
      for (int i = 0; i < n - 1; ++i) {
        cx(qubits[start + i + 1], qubits[start + i]);
      }
      for (int i = n - 1; i >= 0; --i) {
        ry(-parameters[param_idx + i], qubits[start + n - 1 - i]);
      }
      
    }
    for (int i = 0; i < n; ++i) {
      h(qubits[start + i]);
    }
  }
}


inline __qpu__ void cp(double theta, cudaq::qubit &control, cudaq::qubit &target) {
  // Controlled-phase via two CX and three RZ rotations.
  // Equivalent to CRZ(theta) but written explicitly to match CUDA-Q gate set.
  rz(theta / 2.0, control);

  rz(theta / 2.0, target);   // ┐
  cx(control, target);       // │  CRZ(θ)
  rz(-theta / 2.0, target);  // │
  cx(control, target);       // ┘
}


inline __qpu__ void QFT_LNN(cudaq::qvector<>& qubits, int start, int n) {
  // Quantum Fourier Transform adapted for linear-nearest-neighbor connectivity.
  // The implementation swaps multi-control interactions with CX ladders so the
  // circuit can run on hardware with restricted coupling graphs.

  for (int idx = 1; idx < n; ++idx) {
    double angle = M_PI * ( ( (1 << idx) - 1 ) / static_cast<double>(1 << (idx + 1)) );
    rz(angle, qubits[start + n - 1 - idx]);
  }

  for (int b = 0; b < n - 1; ++b) {

    int ctrl = start + n - 1 - b;
    h(qubits[ctrl]);                         // Hadamard

    if (b >= 1 && b <= n - 2) {
      cx(qubits[ctrl], qubits[ctrl - 1]);
    } else {
      for (int k = 0; k < n - b - 1; ++k) {
        cx(qubits[start + n - 1 - (n - 2 - k)],
           qubits[start + n - 1 - (n - 1 - k)]);
      }
    }

    for (int k= 0; k < n - b - 2; k++) {
      cx(qubits[n - 1 - (k + 1 + b) + start], qubits[n - 1 - (k + 2 + b) + start]);
    }

    for (int k = 1; k < n - b; ++k) {
      double phi = -M_PI / static_cast<double>(1 << (k + 1));
      rz(phi, qubits[start + n - 1 - (k + b)]);
    }

    for (int k = 0; k < n - b - 1; ++k) {
      cx(qubits[start + n - 1 - (n - 2 - k)],
         qubits[start + n - 1 - (n - 1 - k)]);
    }

    if (b <= n - 3) {
      cx(qubits[ctrl - 1], qubits[ctrl - 2]);
    } else {
      for (int k = 0; k < n - b - 2; ++k) {
        cx(qubits[ctrl - 1 - k], qubits[ctrl - 2 - k]);
      }
    }
  }

  for (int idx = 1; idx < n; ++idx) {
    double angle = M_PI * ( ( (1 << idx) - 1 ) / static_cast<double>(1 << (idx + 1)) );
    rz(angle, qubits[start + idx]);
  }
  h(qubits[start]);
}

inline __qpu__ void numerator(int num_qubits,
                       int depth,
                       std::vector<double> params,
                       const std::vector<cudaq::complex> &stateVec) {
  // Prepares a register from a classical state vector, applies the inverse
  // variational ansatz, and measures. Used to evaluate ⟨ψ(θ)|f⟩ overlaps.
  cudaq::qvector q = stateVec;
  variational_ansatz_inverse(q, depth, params,
                     /*start=*/0, /*n=*/num_qubits,
                     /*reverse=*/true);
  mz(q);

}
