# USCO-Solver

Code for USCO-Solver.

**USCO-Solver: Solving Undetermined Stochastic Combinatorial Optimization Problems, NeurIPS 2021**

Paper: https://arxiv.org/abs/2107.07508

Please download complete data at https://github.com/cdslabamotong/usco_benchmark

The three folders "ssp" "ssc" and "ssp" contain the codes used for experiments for _Stochastic shortest path, _Stochastic set cover__
Stochastic bipartite matching_, and _
Stochastic bipartite matching_, respectively. 

The one-slack cutting plane algorithm is implemented based on Pystruct.

Except "cvxopt" for quadratic programming, other require packages such as numpy and torch are common.

cvxopt can be installed via pip. (Conda seems to do not have cvxopt)

Thank you.
