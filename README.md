# VQA-Poisson

**Variational Quantum Algorithm for Solving the Poisson Equation**

VQA-Poisson is a hybrid quantum-classical solver designed to compute the solution of the Poisson equation using near-term quantum devices (NISQ). The algorithm is based on a variational formulation, where the trial solution is encoded into a quantum circuit, and the energy functional is minimized using a classical optimizer. This method is suitable for quantum computers without the need for fault tolerance, and it leverages the expressive power of parameterized quantum circuits.

This implementation is based on the publication:

> **Dongyun Ching, Jiyong Choi, Jung-Il Choi .** (2025). *Efficient and Scalable Quantum Library Solving Two-Dimensional Poisson Equatiosn with Mixed Boundary Conditions*.

---

## Overview of the Method

We aim to solve the Poisson equation:

\(-\nabla^2 u(x) = f(x), \quad x \in \Omega\)

with Dirichlet / Neumann / Periodic boundary conditions, by minimizing the energy functional:

\(E_{\boldsymbol{\theta}} = \frac{\text{Re}\left(\left\langle\psi \left(\boldsymbol{\theta}\right)|f\right\rangle\right)^2}{\left\langle\psi\left(\boldsymbol{\theta}\right)|A|\psi\left(\boldsymbol{\theta}\right)\right\rangle}\)

A parameterized quantum circuit \(|\psi(\theta)\rangle\) is used to represent a trial solution. The cost function is evaluated as a quantum expectation value, and minimized using a classical optimizer.

---

## Features

- Hybrid VQA framework for solving PDEs
- Cost function based on discretized energy functional
- Support for 1D/2D domains with Dirichlet / Neumann / Periodic boundary conditions

---

## Repository Structure

- `lib/` : Source code for the quantum-classical hybrid solver
- `lib/Laplacian2D/` : Source code for calculating two-dimensional expectation values
- `lib/example_1D.ipynb` : Interactive Jupyter notebook for 1D demonstration

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
git clone https://github.com/MPMC-Lab/VQA_POISSON.git
```

---


## Authors

- **Dongyun Chung** ([achung3312@yonsei.ac.kr](mailto\:achung3312@yonsei.ac.kr)), School of Mathematics and Computing (Computational Science and Engineering), Yonsei University, Seoul 03722, Republic of Korea

---

## Cite

If you use this work, please cite:

```bibtex
@article{Chung2025vqa,
  title={Efficient and Scalable Quantum Library Solving Two-Dimensional Poisson Equatiosn with Mixed Boundary Conditions},
  author={Dongyun Ching, Jiyong Choi, Jung-Il Choi},
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

For issues, suggestions, or collaborations, please contact: [**achung3312@yonsei.ac.kr**](mailto\:achung3312@yonsei.ac.kr)

