# Numerical Solution of Boundary Value Problems  

## Overview  
This project applies numerical methods to solve **second-order boundary value problems (BVPs)** using the **finite difference method** in Python.  
It demonstrates how to handle both **linear** and **nonlinear** problems, and how to improve accuracy with **Richardson extrapolation**.  

📄 A detailed report with results, error analysis, and convergence plots can be found in [Resultate.pdf](Resultate.pdf).  

---

## Key Skills Highlighted  
- **Numerical Analysis:** Finite differences, Newton’s method, error analysis, convergence studies.  
- **Scientific Programming in Python:** Implemented algorithms for solving linear and nonlinear systems, structured code design.  
- **Optimization of Accuracy:** Applied Richardson extrapolation to systematically improve numerical solutions.  
- **Data Presentation:** Analyzed results with tables and plots, and documented findings clearly.  

---

## Methods Implemented  
- **Finite Difference Method (FDM):** Discretized derivatives on a uniform grid.  
- **Linear Solver:** Used the Thomas algorithm for tridiagonal systems.  
- **Nonlinear Solver:** Applied Newton’s method with a tridiagonal Jacobian.  
- **Richardson Extrapolation:** Improved solution accuracy beyond the base scheme.  

---

## Results  
- Verified **second-order convergence** for linear problems.  
- Achieved **machine-precision accuracy** after Richardson extrapolation.  
- Newton’s method converged rapidly (≈4 iterations) for nonlinear cases.  
- Investigated a challenging problem with singularities near \(x=0\), showing the limitations of uniform grids and pointing to adaptive strategies.  

📄 Full analysis is documented in [Resultate.pdf](Resultate.pdf).  

---

## Repository Structure  
├── Program.py        # Python implementation

├── Resultate.pdf     # Report with results and analysis

└── README.md         # Project description

---

## How to Run  
1. Clone the repository:  
   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name
   python Program.py
