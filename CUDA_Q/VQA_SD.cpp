#include <cudaq.h>
#include <cudaq/optimizers.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <complex>
#include <cmath>
#include <random>
#include <iomanip>
#include <nlopt.h>
#include <chrono>
#include "VQA_qpu.hpp"
#include "VQA.hpp"

__qpu__ void park_QFT(int num_qubits,
                             int depth,
                             std::vector<double> params) {

  cudaq::qvector<> q(num_qubits + 1);
  int num_qubits1D = 5;
  variational_ansatz(q, depth, params,
                     /*start=*/1, /*n=*/num_qubits,
                     /*reverse=*/true);
  QFT_LNN(q, 1, num_qubits1D);
  h(q[0]);

  for (int idx = 0; idx < num_qubits1D; ++idx) {
    double theta = (2 * M_PI / (1 << num_qubits1D)) * (1 << idx);
    cp(theta, q[0], q[num_qubits1D - idx]);
  }
  h(q[0]);
  mz(q[0]);
}

__qpu__ void laplacian_dirichlet(int num_qubits, int num_qubits1D, int depth, int circuit_num, std::vector<double> params){
  // int num_qubits1D = 4;
  cudaq::qvector<> q(num_qubits);

  variational_ansatz(q, depth, params,
                     /*start=*/0, /*n=*/num_qubits,
                     /*reverse=*/true);

  for (int i = 0; i < num_qubits1D - circuit_num - 1; i++) {
    cx(q[(i + 1) + num_qubits1D + 1], q[i + num_qubits1D + 1]);
  }

  h(q[num_qubits - 1 - circuit_num]);

  for (int index = 0; index < num_qubits1D - circuit_num; index++) {
    int idx = index + num_qubits1D + 1;
    mz(q[idx]);
  }
}

__qpu__ void laplacian_periodic(int num_qubits, int num_qubits1D, int depth, int circuit_num, std::vector<double> params){

  cudaq::qvector<> q(num_qubits);

  variational_ansatz(q, depth, params,
                     /*start=*/0, /*n=*/num_qubits,
                     /*reverse=*/true);

  for (int i = 0; i < num_qubits1D - circuit_num - 1; i++) {
    cx(q[(i + 1)], q[i]);
  }

  h(q[num_qubits1D - 1 - circuit_num]);

  for (int index = 0; index < num_qubits1D - circuit_num; index++) {
    mz(q[index]);
  }
}

struct OptData {
  int shots;
  int nq;
  int num_qubits1D;
  int depth;
  int dim;
  const std::vector<cudaq::complex>* b_state;
  const std::vector<double>* analytic;
  const std::vector<double>* normalized_analytic;
  
  std::string prefix;
  
  std::size_t eval_cnt = 0;
  std::size_t iter_cnt = 0;
};

double objective_cb(unsigned n, const double* x, double* /*grad*/, void* ptr){
  auto* d = static_cast<OptData*>(ptr);

  std::vector<double> theta(x, x + n);

  auto num_res = cudaq::sample(d->shots, numerator,
                               d->nq, d->depth, theta, *(d->b_state));

  const std::string num_key((d->nq), '0');
  double p_zero = num_res.count(num_key) / double(d->shots);

  double denom = 0.0;


  for (int j = 0; j < (d->num_qubits1D); j++) {
    auto den_res_dir = cudaq::sample(d->shots, laplacian_dirichlet, d->nq, d->num_qubits1D, d->depth, j, theta);
    if (j == (d->num_qubits1D) - 1) {
      double p0 = den_res_dir.count("0") / static_cast<double>(d->shots);
      double p1 = den_res_dir.count("1") / static_cast<double>(d->shots);
      denom = denom + p0 - p1;
    } else {
      const std::string tail((d->num_qubits1D) - j - 2, '0');
      const std::string key0 = tail + "1" "0";  // == "01" + tail
      const std::string key1 = tail + "1" "1";  // == "11" + tail
      
      double p0 = den_res_dir.count(key0) / static_cast<double>(d->shots);
      double p1 = den_res_dir.count(key1) / static_cast<double>(d->shots);
      denom = denom + (p0 - p1);
    }
  }
  for (int j = 0; j < (d->num_qubits1D) + 1; j++) {
    auto den_res_per = cudaq::sample(d->shots, laplacian_periodic, d->nq, (d->num_qubits1D) + 1, d->depth, j, theta);
    if (j == (d->num_qubits1D)) {
      double p0 = den_res_per.count("0") / static_cast<double>(d->shots);
      double p1 = den_res_per.count("1") / static_cast<double>(d->shots);
      denom = denom + p0 - p1;
    } else if (j == 0) {
      
      const std::string key_per0((d->num_qubits1D) + 1, '0');
      const std::string key_per1((d->num_qubits1D), '0');
      double per0 = den_res_per.count(key_per0) / static_cast<double>(d->shots);
      double per1 = den_res_per.count(key_per1 + "1") / static_cast<double>(d->shots);

      const std::string tail((d->num_qubits1D) + 1 - j - 2, '0');
      const std::string key0 = tail + "1" "0";  // == "01" + tail
      const std::string key1 = tail + "1" "1";  // == "11" + tail
      
      double p0 = den_res_per.count(key0) / static_cast<double>(d->shots);
      double p1 = den_res_per.count(key1) / static_cast<double>(d->shots);
      denom = denom + (p0 - p1) + (per0 - per1);

    } else{
      const std::string tail((d->num_qubits1D) + 1 - j - 2, '0');
      const std::string key0 = tail + "1" "0";  // == "01" + tail
      const std::string key1 = tail + "1" "1";  // == "11" + tail
      
      double p0 = den_res_per.count(key0) / static_cast<double>(d->shots);
      double p1 = den_res_per.count(key1) / static_cast<double>(d->shots);
      denom = denom + (p0 - p1);

    }
  }
  denom = denom - 4.0;
  double f = (std::abs(denom) < 1e-9) ? 1e6 : p_zero / denom;

  std::cout << "[eval " << std::setw(6) << ++d->eval_cnt
            << "] f = " << std::setprecision(15) << f << "\n";
  std::cout << "        params = ";
  for (double v : theta)
    std::cout << std::setprecision(12) << v << ' ';
  std::cout << "\n\n";

  if (d->eval_cnt % (d->dim + 1) == 0) {
    std::string evs_name = "../log/evs_log_SD_" + d->prefix + ".txt";
    std::string trace_name = "../log/trace_log_SD_" + d->prefix + ".txt";
    std::string l2_name = "../log/l2_log_SD_" + d->prefix + ".txt";

    std::ofstream evs(evs_name, std::ios::app);

    evs << std::setprecision(17) << f << '\n';

    std::cout << "  └─ simplex iter " << std::setw(4) << ++d->iter_cnt
              << "  (logged)\n\n";
    
    // L2norm error, Trace distance
    std::vector<double> psi;
    double l2norm = 0.0;
    double trace = 0.0;
    psi = make_classical_psi(1 << d->nq, theta, d->nq, d->depth);

    // Convert b_state to double
    std::vector<double> ampRHS;
    ampRHS.reserve(d->b_state->size());
    for (const auto &c : *d->b_state) {
        ampRHS.push_back(c.real());
    }

    double r = amplitude(d->num_qubits1D, (d->num_qubits1D) + 1, psi, ampRHS);

    l2norm = l2normerror(psi, *(d->analytic), r);
    trace = trace_distance(psi, *(d->normalized_analytic));
    
    std::ofstream l2(l2_name, std::ios::app);
    std::ofstream tr(trace_name, std::ios::app);

    tr << std::setprecision(17) << trace << '\n';
    l2 << std::setprecision(17) << l2norm << '\n';
  }

  return f;
}

//───────────────────── main ───────────────────────────────────────────────────
int main(int argc, char* argv[]){
  int num_qubits   = 9;
  int depth        = 4;
  int shots        = 524'288;
  int dim          = num_qubits * depth;
  int num_qubits1D = num_qubits / 2;

  // check if user passed a value
  if (argc > 1) {
    num_qubits = std::atoi(argv[1]);
  }
  if (argc > 2) {
    depth = std::atoi(argv[2]);
  }
  if (argc > 3) {
    shots = std::atoi(argv[3]);
  }

  printf("VQA Starts (SD):\n");
  printf("Num_qubits: %8d\n", num_qubits);
  printf("Depth: %8d\n", depth);
  printf("num_shots: %8d\n", shots);

  dim          = num_qubits * depth;
  num_qubits1D = num_qubits / 2;

  std::string filename = "../init/init_" + std::to_string(num_qubits) +
                         "_" + std::to_string(depth) + ".csv";

  std::string prefix = std::to_string(num_qubits) + "_" + std::to_string(shots);

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
  std::cout << "Loaded " << theta.size() << " parameters.\n";
  for (size_t i = 0; i < theta.size(); ++i) {
      std::cout << "theta[" << i << "] = " << theta[i] << "\n";
  }

  auto b_state = make_b_state(num_qubits1D);
  std::vector<double> b_real;
  b_real.reserve(b_state.size());
  for (auto &c : b_state)
      b_real.push_back(c.real());

  auto analytic_solution = make_analytic_solution(num_qubits1D, num_qubits1D + 1, b_real);
  std::vector<double> normalized_analytic_solution = analytic_solution;

  normalize(normalized_analytic_solution);

  
  nlopt_opt opt = nlopt_create(NLOPT_LN_COBYLA, dim);
  nlopt_set_maxeval(opt, 2000000);
  nlopt_set_ftol_abs(opt, 1e-4);

  nlopt_set_xtol_rel(opt, 1e-4);
  nlopt_set_xtol_abs1(opt, 1e-4);
  std::vector<double> xtol_abs_vec(dim, 1e-4);
  nlopt_set_xtol_abs(opt, xtol_abs_vec.data());


  OptData data{shots, num_qubits, num_qubits1D, depth, dim, &b_state, &analytic_solution, &normalized_analytic_solution, prefix};

  nlopt_set_min_objective(opt, objective_cb, &data);

  double bestVal;

  std::ofstream("../log/evs_log_SD_" + data.prefix + ".txt", std::ios::trunc).close();
  std::ofstream("../log/trace_log_SD_" + data.prefix + ".txt", std::ios::trunc).close();
  std::ofstream("../log/l2_log_SD_" + data.prefix + ".txt", std::ios::trunc).close();

  // Time
  auto t0 = std::chrono::steady_clock::now();
  nlopt_result ret = nlopt_optimize(opt, theta.data(), &bestVal);
  auto t1 = std::chrono::steady_clock::now();
  double secs = std::chrono::duration<double>(t1 - t0).count();
  std::cout << "\n⏱️  NLopt elapsed = " << secs << " s\n";

  std::cout << "\nNLopt finished code " << int(ret)
            << ", best value = " << bestVal << "\nBest params:\n";
  std::cout << std::setprecision(17);
  for(auto v: theta) std::cout << v << '\n';


{
    std::ofstream evs("../log/evs_log_SD_" + data.prefix + ".txt", std::ios::app);
    std::ofstream tr("../log/trace_log_SD_" + data.prefix + ".txt", std::ios::app);
    std::ofstream l2("../log/l2_log_SD_" + data.prefix + ".txt", std::ios::app);

    evs << std::setprecision(17)
        << bestVal << ' ' << '\n';

    std::vector<double> psi_opt;
    double l2norm_opt = 0.0;
    double trace_opt = 0.0;
    psi_opt = make_classical_psi(1 << (data.nq), theta, data.nq, data.depth);
    
    // Convert b_state to double
    std::vector<double> ampRHS_opt;
    ampRHS_opt.reserve(data.b_state->size());
    for (const auto &c : *data.b_state) {
        ampRHS_opt.push_back(c.real());
    }
    double r_opt = amplitude(data.num_qubits1D, (data.num_qubits1D) + 1, psi_opt, ampRHS_opt);
    
    l2norm_opt = l2normerror(psi_opt, *(data.analytic), r_opt);
    trace_opt = trace_distance(psi_opt, *(data.normalized_analytic));

    tr << std::setprecision(17) << trace_opt << '\n';
    l2 << std::setprecision(17) << l2norm_opt << '\n';
}

  nlopt_destroy(opt);
  return 0;
}