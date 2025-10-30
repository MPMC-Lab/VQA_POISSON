#pragma once
#include <cudaq.h>
#include <vector>

// Classical functions
std::vector<cudaq::complex> make_b_state(int num_qubits1D = 4);
std::vector<double> make_analytic_solution(int x_qubits, int y_qubits, std::vector<double>& RHS);
std::vector<std::vector<double>> identity(int n);
std::vector<std::vector<double>> laplacian_matrix(int n, const std::string& bc);
std::vector<std::vector<double>> kronecker(const std::vector<std::vector<double>>& A,const std::vector<std::vector<double>>& B);
std::vector<double> apply_matrix(const std::vector<std::vector<double>>& M, const std::vector<double>& v);
double dot(const std::vector<double>& a, const std::vector<double>& b);
std::vector<std::vector<double>> CNOT_matrix(int nq, int control_index);
std::vector<std::vector<double>> classical_RYGate(double theta);
std::vector<std::vector<double>> classical_RYGate_nqubits(const std::vector<double>& params, int nq);
void normalize(std::vector<double>& v);
std::vector<double> make_classical_psi(int dim, std::vector<double>& params, int num_qubits, int ansatz_depth);
double amplitude(std::vector<double>& psi);
double l2normerror(const std::vector<double>& psi, const std::vector<double>& analytic_solution, double r);
double trace_distance(const std::vector<double>& psi, const std::vector<double>& analytic_solution);
double amplitude(int x_qubits, int y_qubits, std::vector<double>& psi, std::vector<double>& RHS);
std::vector<double> gauss_solve(std::vector<std::vector<double>>& A, std::vector<double>& b);

// IO
void split_tokens(const std::string& line, std::vector<std::string>& out);
std::vector<double> load_params_csv(const std::string& path);