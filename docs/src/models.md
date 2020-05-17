# Available models

Your help is welcome to help extend these lists. Note the current limitations:

* The models are built and tested assuming `n > p`; if this doesn't hold, tricks should be employed to speed up computations; these have not been implemented yet.
* CV-aware code not implemented yet (code that re-uses computations when fitting over a number of hyper-parameters);  "Meta" functionalities such as One-vs-All or Cross-Validation are left to other packages such as MLJ.
* No support yet for sparse matrices.
* Stochastic solvers have not yet been implemented.
* All computations are assumed to be done in Float64.

## Regression models

| Regressors          | Formulation¹           | Available solvers                 | Comments  |
| :------------------ | :--------------------- | :-------------------------------- | :-------- |
| OLS & Ridge         | L2Loss + 0/L2          | Analytical² or CG³                |           |
| Lasso & Elastic-Net | L2Loss + 0/L2 + L1     | (F)ISTA⁴                          |           |
| Robust 0/L2         | RobustLoss⁵ + 0/L2     | Newton, NewtonCG, LBFGS, IWLS-CG⁶ | no scale⁷ |
| Robust L1/EN        | RobustLoss + 0/L2 + L1 | (F)ISTA                           |           |
| Quantile⁸ + 0/L2    | RobustLoss + 0/L2      | LBFGS, IWLS-CG                    |           |
| Quantile L1/EN      | RobustLoss + 0/L2 + L1 | (F)ISTA                           |           |


1. "0" stands for no penalty
1. Analytical means the solution is computed in "one shot" using the `\` solver,
1. CG = conjugate gradient
1. (Accelerated) Proximal Gradient Descent
1. _Huber_, _Andrews_, _Bisquare_, _Logistic_, _Fair_ and _Talwar_ weighing functions available.
1. Iteratively re-Weighted Least Squares where each system is solved iteratively via CG
1. In other packages such as Scikit-Learn, a scale factor is estimated along with the parameters, this is a bit ad-hoc and corresponds more to a statistical perspective, further it does not work well with penalties; we recommend using cross-validation to set the parameter of the Huber Loss.
1. Includes as special case the _least absolute deviation_ (LAD) regression when `δ=0.5`.

## Classification models

| Classifiers       | Formulation                 | Available solvers        | Comments       |
| :-----------------| :-------------------------- | :----------------------- | :------------- |
| Logistic 0/L2     | LogisticLoss + 0/L2         | Newton, Newton-CG, LBFGS | `yᵢ∈{±1}`      |
| Logistic L1/EN    | LogisticLoss + 0/L2 + L1    | (F)ISTA                  | `yᵢ∈{±1}`      |
| Multinomial 0/L2  | MultinomialLoss + 0/L2      | Newton-CG, LBFGS         | `yᵢ∈{1,...,c}` |
| Multinomial L1/EN | MultinomialLoss + 0/L2 + L1 | ISTA, FISTA              | `yᵢ∈{1,...,c}` |

Unless otherwise specified:

* Newton-like solvers use Hager-Zhang line search (default in [`Optim.jl`]((https://github.com/JuliaNLSolvers/Optim.jl)))
* ISTA, FISTA solvers use backtracking line search and a shrinkage factor of `β=0.8`

**Note**: these models were all tested for correctness whenever a direct comparison with another package was possible, usually by comparing the objective function at the coefficients returned (cf. the tests):
- (_against [scikit-learn](https://scikit-learn.org/)_): Lasso, Elastic-Net, Logistic (L1/L2/EN), Multinomial (L1/L2/EN)
- (_against [quantreg](https://cran.r-project.org/web/packages/quantreg/index.html)_): Quantile (0/L1)

Systematic timing benchmarks have not been run yet but it's planned (see [this issue](https://github.com/alan-turing-institute/MLJLinearModels.jl/issues/14)).
