# VQA-Poisson

VQA_POISSON is a hybrid quantum-classical solver designed to compute the solution of the Poisson equation using near-term quantum devices (NISQ). The algorithm is based on a variational formulation, where the trial solution is encoded into a quantum circuit, and the energy functional is minimized using a classical optimizer.

In our implementation, boundary conditions are treated using a modular circuit construction:
- The cost functional follows the structure introduced by *Sato et al. (2021)*, using a Rayleigh quotient-type expression for variational minimization.
- **Dirichlet** and **Neumann** conditions are implemented based on the variational formulations proposed by *Choi & Ryu (2024)*.
- **Periodic** boundary conditions leverage *QFT-based methods* inspired by *Park & Ahn (2023)* and *Liu et al. (2025)*.

This implementation is based on the following publications:

> **Yuki Sato, Ruho Kondo, Satoshi Koide, Hideki Takamatsu, Nobuyuki Imoto** (2021). *Variational quantum algorithm based on the minimum potential energy for solving the Poisson equation*.
> **Byeongyong Park, Doyeol Ahn .** (2023). *Reducing CNOT count in quantum Fourier transform for the linear nearest-neighbor architecture*.
> **Minjin Choi, Hoon Ryu .** (2024). *A variational quantum algorithm for tackling multi-dimensional Poisson equations with inhomogeneous boundary conditions*.
> **Xiaoqi Liu, Yuedi Qu, Ming Li, Shu-Qian Shen .** (2025). *A variational quantum algorithm for the Poisson equation based on the banded Toeplitz systems*.
> **Dongyun Chung, Jiyong Choi, Jung-Il Choi .** (2025). *Efficient and Scalable Quantum Library Solving Two-Dimensional Poisson Equations with Mixed Boundary Conditions*.
---

## Features

- Hybrid VQA framework for solving PDEs
- Cost function based on discretized energy functional
- Support for 1D/2D domains with Dirichlet / Neumann / Periodic boundary conditions

---

## Repository Structure

- `lib/` : Source code for the quantum-classical hybrid solver
- `lib/QuantumComputer.py` :  
  Core framework that orchestrates the variational quantum algorithm (VQA) workflow.  
  This class provides:
  - validation for ansatz/parameter/circuit structure,
  - transpilation for hardware/simulator execution,
  - unified interfaces for building and evaluating quantum expectation values.  
  Serves as a base class to be subclassed for specific physical problems (e.g., Laplacian, inner product).

- `lib/QuantumCalculator.py` :  
  Problem-specific processors that inherit from `QuantumComputer`, implementing:
  - `LaplacianEVProcessor1D` for computing Laplacian expectation values under Dirichlet, Neumann, or Periodic boundary conditions;
  - `InnerProductProcessor` for evaluating numerator overlaps in variational energy formulations.
 
- `lib/quantum_functions.py` : qiskit-based functions to implement QFT LNN (Park et al.) and linear ansatze
- `lib/classical_functions.py` : NumPy-based emulations of quantum state preparatios and gate applications (mainly used for validation in validation_.py)
- `validation.ipynb` : Jupyter notebook to verify whether our expectations and quantum simulations align.
---

## Installation & Requirements

### Requirements

- Python >= 3.8
- qiskit 1.4.3
- qiskit-ionq 0.5.13
- qiskit-ibm-runtime 0.39.0
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
  author={Dongyun Chung, Jiyong Choi, Jung-Il Choi},
  journal={To be submitted},
  year={2025}
}
```

---

## Acknowledgments

This project was supported by Yonsei University and funded by the Ministry of Science and ICT of Korea through the National Research Foundation (Grant No. RS-2023-00282764).
We are also grateful to the Institute of Quantum Information Technology (IQIT) for providing quantum computing facilities and collaborative guidance.

---

## License

MIT License

---

## Contact

For issues, suggestions, or collaborations, please contact: [**achung3312@yonsei.ac.kr**](mailto\:achung3312@yonsei.ac.kr)

