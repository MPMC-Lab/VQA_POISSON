#include <cudaq.h>
#include <vector>
#include <complex>
#include <cmath>
#include <iostream>
#include "../VQA.hpp"

std::vector<cudaq::complex> make_b_state(int num_qubits1D) {
  using cpx = cudaq::complex;

  const int one_dim      = 1 << num_qubits1D;
  const int grid_num     = one_dim * one_dim;
  const int N            = 2 * grid_num;
  const double alpha0    = 0.5;
  const double alpha1    = 1.81e-5;

  // b₀, b₁
  std::vector<double> b0(N, 0.0), b1(N, 1.0);

  for (int i = 0; i < one_dim / 4; ++i) {
    int base = 3 * grid_num / 4 + one_dim - 1;
    b0[base + 2 * one_dim * i]     = 1.0;
    b0[base + 2 * one_dim * i + 1] = 1.0;
  }

  // b = α₀·b₀ + α₁·b₁
  std::vector<double> amp_d(N);
  double norm2 = 0.0;
  for (int k = 0; k < N; ++k) {
    double v = alpha0 * b0[k] + alpha1 * b1[k];
    amp_d[k] = v;
    norm2   += v * v;
  }

  double norm = std::sqrt(norm2);
  std::vector<cudaq::complex> amp(N);
  for (int k = 0; k < N; ++k)
    amp[k] = cpx{ static_cast<float>(amp_d[k] / norm), 0.0f };

  return amp;
}

std::vector<double> make_analytic_solution(int x_qubits, int y_qubits, std::vector<double>& RHS) {
  int N1 = 1 << x_qubits;
  int N2 = 1 << y_qubits;
  auto A_x = kronecker(identity(N1), laplacian_matrix(N2, "Periodic"));
  auto A_y = kronecker(laplacian_matrix(N1, "Dirichlet"), identity(N2));

  for (int i = 0; i < N1*N2; i++) {
    for (int j = 0; j < N1*N2; j++) {
      A_x[i][j] = A_x[i][j] + A_y[i][j];
    }
  }

  auto x = gauss_solve(A_x, RHS);
  return x;

}