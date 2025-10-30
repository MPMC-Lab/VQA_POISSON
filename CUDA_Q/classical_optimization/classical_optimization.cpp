#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <complex>
#include <cmath>
#include <random>
#include <iomanip>
#include <nlopt.h>
#include "classical_optimization.hpp"

int num_qubits = 7;
int num_qubits1D = num_qubits/2;
int ansatz_depth = 3;
int param_num = num_qubits * ansatz_depth;
int dim = 1 << num_qubits;  // 2^9 = 512
int N1 = 8;
int N2 = 16;
int grid_size = N1 * N2;

// Constants
double alpha_0 = 0.5;
double alpha_1 = 1.81e-5;
int one_dim_grid_num = N1;
int grid_num = N1 * N2;

// Compute etaclassical_optimi
double eta = std::atan(std::pow(std::sqrt(2), one_dim_grid_num + 2) * alpha_1 / alpha_0);

std::vector<std::vector<double>> A;
std::vector<double> b_normalized(grid_size, 0.0);
std::vector<double> b_0(grid_num, 0.0);
std::vector<double> b_1(grid_num, 1.0);
std::vector<double> b(grid_num, 0.0);

void initialize_b_normalized() {
    for (int i = 0; i < one_dim_grid_num / 4; ++i) {
        int first_index_1 = static_cast<int>(3 * one_dim_grid_num * one_dim_grid_num / 4 + one_dim_grid_num - 1);
        int first_index_2 = static_cast<int>(3 * one_dim_grid_num * one_dim_grid_num / 4 + one_dim_grid_num);

        int idx1 = static_cast<int>(first_index_1 + 2 * one_dim_grid_num * i);
        int idx2 = static_cast<int>(first_index_2 + 2 * one_dim_grid_num * i);

        b_0[idx1] = 1.0;
        b_0[idx2] = 1.0;
    }
    
    for (int i = 0; i < grid_num; ++i) {
        b[i] = alpha_0 * b_0[i] + alpha_1 * b_1[i];
    }

    double norm = std::sqrt(dot(b, b));

    for (int i = 0; i < grid_num; ++i) {
        b_normalized[i] = b[i] / norm;
    }
}

// Initialize global stiffness matrix A
void initialize_A() {
    auto A_x = kronecker(identity(N1), laplacian_matrix(N2, "Periodic"));
    auto A_y = kronecker(laplacian_matrix(N1, "Dirichlet"), identity(N2));

    int n = A_x.size();
    A.resize(n, std::vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            A[i][j] = A_x[i][j] + A_y[i][j];
}

// Cost function
double classical_cost_function(const std::vector<double>& params) {
    std::vector<double> psi = make_classical_psi(params);
    double numerator = dot(b_normalized, psi);
    
    numerator = numerator * numerator; // (b^T psi)^2
    std::vector<double> Apsi = apply_matrix(A, psi);
    double denominator = dot(psi, Apsi); // psi^T A psi
    
    double cost = numerator / denominator;
    return cost;
}

struct OptData {
    int dim;
    int nq;
    int num_qubits1D;
    int depth;
    const std::vector<double>* b_state;
    const std::vector<double>* analytic;
    const std::vector<double>* normalized_analytic;
    size_t eval_cnt = 0;
    size_t iter_cnt = 0;
};

double nlopt_cost_function(unsigned n, const double* x, double* grad, void* data) {
    auto* d = static_cast<OptData*>(data);
    std::vector<double> params(x, x + n);
    double cost = classical_cost_function(params);

    // Print every evaluation
    std::cout << "[eval " << std::setw(6) << ++d->eval_cnt
              << "] cost = " << std::setprecision(15) << cost << "\n";
    std::cout << "        params = ";
    for (double v : params)
        std::cout << std::setprecision(12) << v << ' ';
    std::cout << "\n\n";

    // Log to file every (dim + 1) evaluations (COBYLA-like iteration)
    if (d->eval_cnt % (d->dim + 1) == 0) {
        std::ofstream evs("../log/cost_log.txt", std::ios::app);
        std::ofstream par("../log/params_log.txt", std::ios::app);
        std::ofstream tr("../log/trace_log.txt", std::ios::app);
        std::ofstream l2("../log/l2_log.txt", std::ios::app);

        evs << std::setprecision(17) << cost << '\n';
        par << std::setprecision(17);
        for (double v : params) par << v << ' ';
        par << '\n';

        std::cout << "  └─ iter " << std::setw(4) << ++d->iter_cnt
                  << " logged to file.\n\n";

    // L2norm error, Trace distance
    std::vector<double> psi;
    double l2norm = 0.0;
    double trace = 0.0;
    psi = make_classical_psi(params);
    double r = amplitude(d->num_qubits1D, (d->num_qubits1D) + 1, psi, b_normalized);
    l2norm = l2normerror(psi, *(d->analytic), r);
    trace = trace_distance(psi, *(d->normalized_analytic));

    tr << std::setprecision(17) << trace << '\n';
    l2 << std::setprecision(17) << l2norm << '\n';
    }

    return cost;
}

static inline void split_tokens(const std::string& line, std::vector<std::string>& out) {
    std::string token;
    for (char c : line) {
        if (c == ',' || c == ' ' || c == '\t' || c == ';')
            c = ' ';
        token.push_back(c);
    }
    std::istringstream iss(token);
    std::string w;
    while (iss >> w) out.push_back(w);
}

std::vector<double> load_params_csv(const std::string& path) {
    std::ifstream fin(path);
    if (!fin) throw std::runtime_error("cannot open " + path);

    std::vector<std::string> toks;
    std::string line;
    while (std::getline(fin, line)) {
        if (line.empty()) continue;
        split_tokens(line, toks);
    }
    std::vector<double> vals;
    vals.reserve(toks.size());
    for (auto& s : toks) vals.push_back(std::stod(s));
    return vals;
}

int main() {

    std::string filename = "../../init/init_" + std::to_string(num_qubits) +
                           "_" + std::to_string(ansatz_depth) + ".csv";

    std::cout << "Loading file: " << filename << "\n";

    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: could not open file " << filename << "\n";
        return 1;
    }
     // CSV → vector<double>
    std::vector<double> theta;
    std::string line;
    while (std::getline(file, line, ' ')) {
        std::stringstream ss(line);
        double val;
        if (ss >> val) theta.push_back(val);
    }
    file.close();

    initialize_b_normalized();
    std::ofstream("../log/cost_log.txt", std::ios::trunc).close();
    std::ofstream("../log/params_log.txt", std::ios::trunc).close();
    std::ofstream("../log/trace_log.txt", std::ios::trunc).close();
    std::ofstream("../log/l2_log.txt", std::ios::trunc).close();
    initialize_A();

    auto analytic_solution = make_analytic_solution(num_qubits1D, num_qubits1D + 1, b_normalized);
    std::vector<double> normalized_analytic_solution = analytic_solution;
    normalize(normalized_analytic_solution);
    
    OptData data{param_num, num_qubits, num_qubits1D, ansatz_depth, &b_normalized, &analytic_solution, &normalized_analytic_solution};

    nlopt_opt opt = nlopt_create(NLOPT_LN_COBYLA, param_num);
    nlopt_set_min_objective(opt, nlopt_cost_function, &data);
    nlopt_set_maxeval(opt, 2000000);
    nlopt_set_ftol_abs(opt, 1e-4);
    nlopt_set_xtol_rel(opt, 1e-4);
    nlopt_set_xtol_abs1(opt, 1e-4);
    std::vector<double> xtol_abs_vec(dim, 1e-4);
    nlopt_set_xtol_abs(opt, xtol_abs_vec.data());

    double min_cost;
    
    nlopt_result result = nlopt_optimize(opt, theta.data(), &min_cost);

    if (result < 0) {
        std::cerr << "NLOPT failed with error code: " << result << std::endl;
    } else {
        std::cout << "Optimization succeeded. Final cost = " << min_cost << std::endl;
        for (int i = 0; i < param_num; ++i)
            std::cout << "theta[" << i << "] = " << theta[i] << std::endl;
    }
    nlopt_destroy(opt);
    return 0;
}