#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <cassert>
#include <nlopt.h>
#include "../VQA.hpp"

std::vector<std::vector<double>> identity(int n) {
    std::vector<std::vector<double>> I(n, std::vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i) I[i][i] = 1.0;
    return I;
}

std::vector<std::vector<double>> laplacian_matrix(int n, const std::string& bc) {
    std::vector<std::vector<double>> L(n, std::vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i) L[i][i] = -2.0;
    for (int i = 0; i < n - 1; ++i) L[i + 1][i] = L[i][i + 1] = 1.0;
    if (bc == "Neumann") {
        L[0][0] = L[n - 1][n - 1] = -1.0;
    } else if (bc == "Periodic") {
        L[n - 1][0] = L[0][n - 1] = 1.0;
    }
    return L;
}

std::vector<std::vector<double>> kronecker(const std::vector<std::vector<double>>& A,
                                           const std::vector<std::vector<double>>& B) {
    int m = A.size(), n = A[0].size();
    int p = B.size(), q = B[0].size();
    std::vector<std::vector<double>> K(m * p, std::vector<double>(n * q));
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            for (int k = 0; k < p; ++k)
                for (int l = 0; l < q; ++l)
                    K[i * p + k][j * q + l] = A[i][j] * B[k][l];
    return K;
}

std::vector<double> apply_matrix(const std::vector<std::vector<double>>& M, const std::vector<double>& v) {
    int n = M.size();
    std::vector<double> res(n, 0.0);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            res[i] += M[i][j] * v[j];
    return res;
}

double dot(const std::vector<double>& a, const std::vector<double>& b) {
    double res = 0.0;
    int grid_num = a.size();
    for (int i = 0; i < grid_num; i++) {
      res = res + a[i] * b[i];
    }
    return res;
}

std::vector<std::vector<double>> CNOT_matrix(int nq, int control_index) {
    assert(control_index >= 0 && control_index < nq - 1);

    // Define classical CNOT gate (4x4)
    std::vector<std::vector<double>> classical_CNOT = {
        {1, 0, 0, 0},
        {0, 1, 0, 0},
        {0, 0, 0, 1},
        {0, 0, 1, 0}
    };

    // Identity gate (2x2)
    std::vector<std::vector<double>> I = {
        {1, 0},
        {0, 1}
    };

    std::vector<std::vector<double>> result;

    // Initial part: I x I x ... x I
    if (control_index != 0)
        result = identity(1);  // 1x1 identity to start Kronecker
    else
        result = classical_CNOT; // special case when control_index = 0

    for (int i = 0; i < control_index; ++i)
        result = kronecker(result, I);

    if (control_index != 0)
        result = kronecker(result, classical_CNOT);

    for (int i = control_index + 2; i < nq; ++i)
        result = kronecker(result, I);

    return result;
}

// RY Gate: 2x2 rotation matrix
std::vector<std::vector<double>> classical_RYGate(double theta) {
    return {
        {std::cos(theta / 2.0), -std::sin(theta / 2.0)},
        {std::sin(theta / 2.0),  std::cos(theta / 2.0)}
    };
}

// Tensor product (Kronecker) of a list of RY gates
std::vector<std::vector<double>> classical_RYGate_nqubits(const std::vector<double>& params, int nq) {
    std::vector<std::vector<double>> RY = classical_RYGate(params[0]);
    for (int i = 1; i < nq; ++i) {
        auto next = classical_RYGate(params[i]);
        RY = kronecker(RY, next);
    }
    return RY;
}

// Function to normalize a vector
void normalize(std::vector<double>& v) {
    double norm = std::sqrt(dot(v, v));
    for (auto& x : v) x /= norm;
}

// Construct the classical state vector from parameters
std::vector<double> make_classical_psi(int dim, std::vector<double>& params, int num_qubits, int ansatz_depth) {
    std::vector<double> psi(dim, 1.0);
    normalize(psi);

    for (int depth = 0; depth < ansatz_depth; ++depth) {
        std::vector<double> params_layer(params.begin() + depth * num_qubits,
                                         params.begin() + (depth + 1) * num_qubits);
        auto RY = classical_RYGate_nqubits(params_layer, num_qubits);
        psi = apply_matrix(RY, psi);

        for (int qubit = 0; qubit < num_qubits - 1; ++qubit) {
            auto CNOT = CNOT_matrix(num_qubits, qubit);
            psi = apply_matrix(CNOT, psi);
        }
    }
    return psi;
}

double amplitude(int x_qubits, int y_qubits, std::vector<double>& psi, std::vector<double>& RHS) {
      int N1 = 1 << x_qubits;
      int N2 = 1 << y_qubits;
      auto A_x = kronecker(identity(N1), laplacian_matrix(N2, "Periodic"));
      auto A_y = kronecker(laplacian_matrix(N1, "Dirichlet"), identity(N2));

      for (int i = 0; i < N1*N2; i++) {
          for (int j = 0; j < N1*N2; j++) {
          A_x[i][j] = A_x[i][j] + A_y[i][j];
          }
      }
      double numerator = 0.0;
      for (int i = 0; i < N1 * N2; i++) {
        numerator += RHS[i] * psi[i];
      }
      std::vector<double> Apsi = apply_matrix(A_x, psi);
      double denominator = dot(psi, Apsi);
      double amp = abs(numerator / denominator);
      return amp;
}

double l2normerror(const std::vector<double>& psi, const std::vector<double>& analytic_solution, double r) {
    double l2norm = 0.0;
    int size = psi.size();
    for (int i = 0; i < size; i++) {
        l2norm += (abs(r * psi[i]) - abs(analytic_solution[i])) 
                * (abs(r * psi[i]) - abs(analytic_solution[i]));
    }
    l2norm = sqrt(l2norm / size);
    return l2norm;
}

double trace_distance(const std::vector<double>& psi, const std::vector<double>& analytic_solution) {
    double inner = 0.0;
    int size = psi.size();

    for (int i = 0; i < size; i++) {
        inner += abs(psi[i]) * abs(analytic_solution[i]);
    }
    inner = inner * inner;
    return sqrt(1 - inner);
}

std::vector<double> gauss_solve(std::vector<std::vector<double>>& A, std::vector<double>& b) {
    int n = A.size();
    for (int i = 0; i < n; ++i) {
        double maxVal = std::fabs(A[i][i]);
        int maxRow = i;
        for (int k = i + 1; k < n; ++k) {
            if (std::fabs(A[k][i]) > maxVal) {
                maxVal = std::fabs(A[k][i]);
                maxRow = k;
            }
        }
        std::swap(A[i], A[maxRow]);
        std::swap(b[i], b[maxRow]);

        for (int k = i + 1; k < n; ++k) {
            double c = A[k][i] / A[i][i];
            for (int j = i; j < n; ++j)
                A[k][j] -= c * A[i][j];
            b[k] -= c * b[i];
        }
    }

    // back-substitution
    std::vector<double> x(n);
    for (int i = n - 1; i >= 0; --i) {
        x[i] = b[i];
        for (int j = i + 1; j < n; ++j)
            x[i] -= A[i][j] * x[j];
        x[i] /= A[i][i];
    }
    return x;
}

