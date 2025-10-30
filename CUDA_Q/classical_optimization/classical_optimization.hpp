#include <iostream>
#include <vector>
#include <fstream>
#include <complex>
#include <cmath>
#include <random>
#include <iomanip>
#include <nlopt.h>

extern int num_qubits;
extern int ansatz_depth;
extern int param_num;
extern int dim;
extern int N1;
extern int grid_size;

extern double alpha_0;
extern double alpha_1;
extern int one_dim_grid_num;
extern int grid_num;

extern double eta;
extern std::vector<std::vector<double>> A;
extern std::vector<double> b_0;
extern std::vector<double> b_1;
extern std::vector<double> b_normalized;

void initialize_b_normalized();
void initialize_A();
double classical_cost_function(const std::vector<double>& params);

std::vector<std::vector<double>> identity(int n);
std::vector<std::vector<double>> laplacian_matrix(int n, const std::string& bc);

std::vector<std::vector<double>> kronecker(const std::vector<std::vector<double>>& A,
                                           const std::vector<std::vector<double>>& B);
std::vector<double> apply_matrix(const std::vector<std::vector<double>>& M, const std::vector<double>& v);
double dot(const std::vector<double>& a, const std::vector<double>& b);
std::vector<std::vector<double>> CNOT_matrix(int nq, int control_index);
std::vector<std::vector<double>> classical_RYGate(double theta);
std::vector<std::vector<double>> classical_RYGate_nqubits(const std::vector<double>& params, int nq);
void normalize(std::vector<double>& v);
std::vector<double> make_classical_psi(const std::vector<double>& params);
double amplitude(int x_qubits, int y_qubits, std::vector<double>& psi, std::vector<double>& RHS);
double l2normerror(const std::vector<double>& psi, const std::vector<double>& analytic_solution, double r);
double trace_distance(const std::vector<double>& psi, const std::vector<double>& analytic_solution);
std::vector<double> gauss_solve(std::vector<std::vector<double>>& A, std::vector<double>& b);
std::vector<double> make_analytic_solution(int x_qubits, int y_qubits, std::vector<double> RHS);