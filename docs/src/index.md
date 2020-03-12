# MLJLinearModels.jl

This is a convenience package gathering functionalities to solve a number of generalised linear regression/classification problems which, inherently, correspond to an optimisation problem of the form

```math
L(y, X\theta) + P(\theta)
```

where ``L`` is a _loss function_ and ``P`` is a  _penalty function_ (both of those can be scaled or composed).

A well known example is the [Ridge regression](https://en.wikipedia.org/wiki/Tikhonov_regularization) where the problem amounts to minimising

```math
\|y - X\theta\|_2^2 + \lambda\|\theta\|_2^2.
```

## Goals for the package

- make these regressions models "easy to call" and callable in a unified way,
- interface with [`MLJ.jl`](https://github.com/alan-turing-institute/MLJ.jl),
- focus on performance including in "big data" settings exploiting packages such as [`Optim.jl`](https://github.com/JuliaNLSolvers/Optim.jl), and [`IterativeSolvers.jl`](https://github.com/JuliaMath/IterativeSolvers.jl),
- use a "machine learning" perspective, i.e.: focus primarily on prediction, hyper-parameters should be obtained via a data-driven procedure such as cross-validation.

All models allow to fit an intercept and allow the penalty to be optionally applied on the intercept (not applied by default).
All models attempt to be efficient in terms of memory allocation to avoid unnecessary copies of the data.

## Quick start

The package works by

1. specifying the kind of model you want along with its hyper-parameters,
2. calling `fit` with that model and the data: `fit(model, X, y)`.

!!! note

    The convention is that the feature matrix has dimensions ``n \times p`` where ``n`` is the number of records (points) and ``p`` is the number of features (dimensions).

### Lasso regression

The lasso regression corresponds to a l2-loss function with a l1-penalty:

```math
\theta_{\text{Lasso}} = \frac12\|y-X\theta\|_2^2 + \lambda\|\theta\|_1
```

which you can create as follows:

```julia
λ = 0.7
lasso = LassoRegression(0.7)
fit(lasso, X, y)
```

### (Multinomial) logistic classifier

In a classification context, the multinomial logistic regression returns a predicted score per class that can be interpreted as the likelihood of a point belonging to a class given the trained model.
It's given by the multinomial loss plus an optional penalty (typically the l2 penalty).

Here's a way to do this:

```julia
λ = 0.1
mlr = MultinomialRegression(λ) # you can also just use LogisticRegression
fit(mlr, X, y)
```

In a **binary** context, ``y`` is expected to have values ``y_i \in \{\pm 1\}`` whereas in the **multiclass** context, ``y`` is expected to have values ``y_i \in {1, \dots, c}`` where ``c > 2`` is the number of classes.

## Available models

### Regression models (continuous target)

| Regressors          | Formulation¹           | Available solvers                 | Comments  |
| :------------------ | :--------------------- | :-------------------------------- | :-------- |
| OLS & Ridge         | L2Loss + 0/L2          | Analytical² or CG³                |           |
| Lasso & Elastic-Net | L2Loss + 0/L2 + L1     | (F)ISTA⁴                          |           |
| Robust 0/L2         | RobustLoss⁵ + 0/L2     | Newton, NewtonCG, LBFGS, IWLS-CG⁶ | no scale⁷ |
| Robust L1/EN        | RobustLoss + 0/L2 + L1 | (F)ISTA                           |           |
| Quantile⁸ + 0/L2    | RobustLoss + 0/L2      | LBFGS, IWLS-CG                    |           |
| Quantile L1/EN      | RobustLoss + 0/L2 + L1 | (F)ISTA                           |           |

1. "0" stands for no penalty
2. Analytical means the solution is computed in "one shot" using the `\` solver,
3. CG = conjugate gradient
4. (Accelerated) Proximal Gradient Descent
5. _Huber_, _Andrews_, _Bisquare_, _Logistic_, _Fair_ and _Talwar_ weighing functions available.
6. Iteratively re-Weighted Least Squares where each system is solved iteratively via CG
7. In other packages such as Scikit-Learn, a scale factor is estimated along with the parameters, this is a bit ad-hoc and corresponds more to a statistical perspective, further it does not work well with penalties; we recommend using cross-validation to set the parameter of the Huber Loss.
8. Includes as special case the _least absolute deviation_ (LAD) regression when `δ=0.5`.

### Classification models (finite target)

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

## Limitations

Note the current limitations:

* The models are built and tested assuming `n > p`; if this doesn't hold, tricks should be employed to speed up computations; these have not been implemented yet.
* CV-aware code not implemented yet (code that re-uses computations when fitting over a number of hyper-parameters);  "Meta" functionalities such as One-vs-All or Cross-Validation are left to other packages such as MLJ.
* No support yet for sparse matrices.
* Stochastic solvers have not yet been implemented.
* All computations are assumed to be done in Float64.
