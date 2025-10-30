#pragma once
#include <cudaq.h>
#include <vector>

inline __qpu__ void variational_ansatz(cudaq::qvector<>& qubits,
                                      int ansatz_depth, 
                                      std::vector<double>& parameters, 
                                      int start, int n, 
                                      bool reverse = false) {
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
  rz(theta / 2.0, control);

  rz(theta / 2.0, target);   // ┐
  cx(control, target);       // │  CRZ(θ)
  rz(-theta / 2.0, target);  // │
  cx(control, target);       // ┘
}


inline __qpu__ void QFT_LNN(cudaq::qvector<>& qubits, int start, int n) {

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
  cudaq::qvector q = stateVec;
  variational_ansatz_inverse(q, depth, params,
                     /*start=*/0, /*n=*/num_qubits,
                     /*reverse=*/true);
  mz(q);
}