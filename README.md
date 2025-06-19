# VQA-Poisson

**Variational Quantum Algorithm for Solving the Poisson Equation**

VQA-Poisson is a hybrid quantum-classical solver designed to compute the solution of the Poisson equation using near-term quantum devices (NISQ). The algorithm is based on a variational formulation, where the trial solution is encoded into a quantum circuit, and the energy functional is minimized using a classical optimizer. This method is suitable for quantum computers without the need for fault tolerance, and it leverages the expressive power of parameterized quantum circuits.

This implementation is based on the publication:

> **Jeong, Dong-Yoon.** (2025). *Solving the Poisson Equation Using a Variational Quantum Algorithm*. [arXiv/Journal link if available]

---

## Overview of the Method

We aim to solve the Poisson equation:

\(-\nabla^2 u(x) = f(x), \quad x \in \Omega\)

with Dirichlet boundary conditions, by minimizing the energy functional:

\(\mathcal{L}[u] = \int_\Omega \left[ \frac{1}{2} (\nabla u)^2 - f(x) u(x) \right] dx\)

A parameterized quantum circuit \(|\psi(\theta)\rangle\) is used to represent a trial solution. The cost function is evaluated as a quantum expectation value, and minimized using a classical optimizer.

---

## Features

- Hybrid VQA framework for solving PDEs
- Cost function based on discretized energy functional
- Support for 1D/2D domains with Dirichlet boundary conditions
- Modular structure for ansatz design and Hamiltonian construction

---

## Repository Structure

- `src/` : Source code for the quantum-classical hybrid solver
- `circuit/` : Qiskit circuits defining the variational ansatz
- `data/` : Benchmark solutions and result logs
- `figures/` : Sample visualizations of results
- `notebooks/` : Interactive Jupyter notebooks for demonstration

---

## Installation & Requirements

### Requirements

- Python >= 3.8
- Qiskit
- NumPy
- SciPy
- Matplotlib

### Installation

```bash
git clone https://github.com/[YOUR_USERNAME]/VQA_Poisson.git
cd VQA_Poisson
pip install -r requirements.txt
```

---

## Running the Solver

```bash
python main.py --dim 1 --ansatz default --optimizer COBYLA --shots 8192
```

Arguments:

- `--dim`: Problem dimensionality (1 or 2)
- `--ansatz`: Type of parameterized circuit (default/custom)
- `--optimizer`: Classical optimizer (COBYLA, SLSQP, etc.)
- `--shots`: Number of shots for expectation estimation

---

## Visualization

The solver supports visualization of the learned solution and convergence plot:

```bash
python visualize.py --input result.npy
```

---

## Authors

- **Jeong, Dong-Yoon** ([dyjeong@yonsei.ac.kr](mailto\:dyjeong@yonsei.ac.kr)), Graduate School of Computational Science, Yonsei University

---

## Cite

If you use this work, please cite:

```bibtex
@article{jeong2025vqa,
  title={Solving the Poisson Equation Using a Variational Quantum Algorithm},
  author={Jeong, Dong-Yoon},
  journal={To be submitted},
  year={2025}
}
```

---

## Acknowledgments

This project was supported by Yonsei University and was inspired by foundational work in variational quantum algorithms for scientific computing.

---

## License

MIT License

---

## Contact

For issues, suggestions, or collaborations, please contact: [**dyjeong@yonsei.ac.kr**](mailto\:dyjeong@yonsei.ac.kr)

