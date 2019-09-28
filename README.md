# MLJLinearModels.jl

| [MacOS/Linux] | Coverage | Documentation |
| :------------ | :------- | :------------ |
| [![Build Status](https://travis-ci.org/alan-turing-institute/MLJLinearModels.jl.svg?branch=master)](https://travis-ci.org/alan-turing-institute/MLJLinearModels.jl) | [![codecov.io](http://codecov.io/github/alan-turing-institute/MLJLinearModels.jl/coverage.svg?branch=master)](http://codecov.io/github/alan-turing-institute/MLJLinearModels.jl?branch=master) | TODO |

This is a convenience package gathering functionalities to solve a number of generalised linear regression/classification problems which, inherently, correspond to an optimisation problem of the form

```
L(y, Xθ) + P(θ)
```

where `L` is a loss function and `P`  is a penalty function (both of those can be scaled or composed).
Additional regression/classification methods which do not directly correspond to this formulation may be added in the future.

The core aims of this package are:

- make these regressions models "easy to call" and callable in a unified way,
- interface with [`MLJ.jl`](https://github.com/alan-turing-institute/MLJ.jl),
- focus on performance including in "big data" settings exploiting packages such as [`Optim.jl`](https://github.com/JuliaNLSolvers/Optim.jl), [`IterativeSolvers.jl`](https://github.com/JuliaMath/IterativeSolvers.jl),
- use a "machine learning" perspective, i.e.: focus essentially on prediction, hyper-parameters should be obtained via a data-driven procedure such as cross-validation.

All models allow to fit an intercept and allow the penalty to be applied or not on the intercept (not applied by default).
All models attempt to be efficient in terms of memory allocation to avoid unnecessary copies of the data.

## Implemented

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

### Current limitations

* The models are built and tested assuming `n > p`; if this doesn't hold, tricks should be employed to speed up computations; these have not been implemented yet.
* CV-aware code not implemented yet (code that re-uses computations when fitting over a number of hyper-parameters);  "Meta" functionalities such as One-vs-All or Cross-Validation are left to other packages such as MLJ.
* No support yet for sparse matrices.
* Stochastic solvers have not yet been implemented.
* All computations are assumed to be done in Float64.

### Possible future models

#### Future

| Model                     | Formulation                  | Comments |
| :------------------------ | :--------------------------- | :------- |
| Group Lasso               | L2Loss + ∑L1 over groups     |  ⭒       |
| Adaptive Lasso            | L2Loss + weighted L1         |  ⭒ [A](http://myweb.uiowa.edu/pbreheny/7600/s16/notes/2-29.pdf) |
| SCAD                      | L2Loss + SCAD                |  A, [B](https://arxiv.org/abs/0903.5474), [C](https://orfe.princeton.edu/~jqfan/papers/01/penlike.pdf) |
| MCP                       | L2Loss + MCP                 |  A        |
| OMP                       | L2Loss + L0Loss              |  [D](https://www.cs.technion.ac.il/~ronrubin/Publications/KSVD-OMP-v2.pdf) |
| SGD Classifiers           | *Loss + No/L2/L1  and OVA    | [SkL](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html) |

* (⭒) should be added soon


#### Other regression models

There are a number of other regression models that may be included in this package in the longer term but may not directly correspond to the paradigm `Loss+Penalty` introduced earlier.

In some cases it will make more sense to just use [GLM.jl](https://github.com/JuliaStats/GLM.jl).

Sklearn's list: https://scikit-learn.org/stable/supervised_learning.html#supervised-learning

| Model                       | Note        | Link(s)                                            |
| :-------------------------- | :---------- | :------------------------------------------------- |
| LARS                        | --          |                                                    |
| Quantile Regression         | --          | [Yang et al, 2013](https://www.stat.berkeley.edu/~mmahoney/pubs/quantile-icml13.pdf), [QuantileRegression.jl](https://github.com/pkofod/QuantileRegression.jl) |
| L∞ approx (Logsumexp)       | --          | [slides](https://www.cs.ubc.ca/~schmidtm/Courses/340-F15/L15.pdf)|
| Passive Agressive           | --          | [Crammer et al, 2006](http://jmlr.csail.mit.edu/papers/volume7/crammer06a/crammer06a.pdf) [SkL](https://scikit-learn.org/stable/modules/linear_model.html#passive-aggressive-algorithms) |
| Orthogonal Matching Pursuit | --          | [SkL](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.OrthogonalMatchingPursuit.html#sklearn.linear_model.OrthogonalMatchingPursuit) |
| Least Median of Squares     | --          | [Rousseeuw, 1984](http://web.ipac.caltech.edu/staff/fmasci/home/astro_refs/LeastMedianOfSquares.pdf) |
| RANSAC, Theil-Sen           | Robust reg  | [Overview RANSAC](http://www.cse.yorku.ca/~kosta/CompVis_Notes/ransac.pdf), [SkL](https://scikit-learn.org/stable/modules/linear_model.html#robustness-regression-outliers-and-modeling-errors), [SkL](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.TheilSenRegressor.html#sklearn.linear_model.TheilSenRegressor), [More Ransac](http://www.cs.tau.ac.il/~turkel/imagepapers/RANSAC4Dummies.pdf) |
| Ordinal regression          | _need to figure out how they work_ | [E](https://cran.r-project.org/web/packages/pscl/vignettes/countreg.pdf)|
| Count regression            | _need to figure out how they work_ | [R](https://cran.r-project.org/web/packages/pscl/vignettes/countreg.pdf) |
| Robust M estimators         | --          | [F](https://arxiv.org/pdf/1508.01967.pdf) |
| Perceptron, MIRA classifier | Sklearn just does OVA with binary in SGDClassif      | [H](https://cl.lingfil.uu.se/~nivre/master/ml7-18.pdf) |
| Robust PTS and LTS | -- | [PTS](https://arxiv.org/pdf/0901.0876.pdf) [LTS](https://arxiv.org/pdf/1304.4773.pdf) |


## What about other packages

While the functionalities in this package overlap with a number of existing packages, the hope is that this package will offer a general entry point for all of them in a way that won't require too much thinking from an end user (similar to how someone would use the tools from `sklearn.linear_model`).
If you're looking for specific functionalities/algorithms, it's probably a good idea to look at one of the packages below:

- [SparseRegression.jl](https://github.com/joshday/SparseRegression.jl)
- [Lasso.jl](https://github.com/JuliaStats/Lasso.jl)
- [QuantileRegression.jl](https://github.com/pkofod/QuantileRegression.jl)
- (unmaintained) [Regression.jl](https://github.com/lindahua/Regression.jl)
- (unmaintained) [LARS.jl](https://github.com/simonster/LARS.jl)
- (unmaintained) [FISTA.jl](https://github.com/klkeys/FISTA.jl)
- (unmaintained) [RobustLeastSquares.jl](https://github.com/FugroRoames/RobustLeastSquares.jl)

There's also [GLM.jl](https://github.com/JuliaStats/GLM.jl) which is more geared towards statistical analysis for reasonably-sized datasets and does (as far as I'm aware) lack a few key functionalities for ML such as penalised regressions or multinomial regression.

## References

* **Minka**, [Algorithms for Maximum Likelihood Regression](https://tminka.github.io/papers/logreg/minka-logreg.pdf), 2003. For a review of numerical methods for the binary Logistic Regression.
* **Beck** and **Teboulle**, [A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems](https://tinyurl.com/beck-teboulle-fista), 2009. For the ISTA and FISTA algorithms.
* **Raman** et al, [DS-MLR: Exploiting Double Separability for Scaling up DistributedMultinomial Logistic Regression](https://arxiv.org/pdf/1604.04706.pdf), 2018. For a discussion of multinomial regression.
* _Robust regression_
    * **Mastronardi**, [Fast Robust Regression Algorithms for Problems with Toeplitz Structure](https://pdfs.semanticscholar.org/5d54/df9fc59b26027ede8599af850cd46cdf2255.pdf), 2007. For a discussion on algorithms for robust regression.
    * **Fox** and **Weisberg**, [Robust Regression](http://users.stat.umn.edu/~sandy/courses/8053/handouts/robust.pdf), 2013. For a discussion on robust regression and the IWLS algorithm.
    * _Statsmodels_, [M Estimators for Robust Linear Modeling](https://www.statsmodels.org/dev/examples/notebooks/generated/robust_models_1.html). For a list of weight functions beyond Huber's.
    * **O'Leary**, [Robust Regression Computation using Iteratively Reweighted Least Squares](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.306.8839&rep=rep1&type=pdf), 1990. Discussion of a few common robust regressions and implementation with IWLS.

## Dev notes

* Probit Loss --> via StatsFuns // Φ(x) (normcdf); ϕ(x) (normpdf); -xϕ(x)
* Newton, LBFGS take linesearches, seems NewtonCG doesn't
* several ways of doing backtracking (e.g. https://archive.siam.org/books/mo25/mo25_ch10.pdf); for FISTA many though see http://www.seas.ucla.edu/~vandenbe/236C/lectures/fista.pdf; probably best to have "decent safe defaults"; also this for FISTA http://150.162.46.34:8080/icassp2017/pdfs/0004521.pdf ; https://github.com/tiepvupsu/FISTA#in-case-lf-is-hard-to-find ; https://hal.archives-ouvertes.fr/hal-01596103/document; not so great https://github.com/klkeys/FISTA.jl/blob/master/src/lasso.jl ;
* https://www.ljll.math.upmc.fr/~plc/prox.pdf
* proximal QN http://www.stat.cmu.edu/~ryantibs/convexopt-S15/lectures/24-prox-newton.pdf; https://www.cs.utexas.edu/~inderjit/public_papers/Prox-QN_nips2014.pdf; https://github.com/yuekai/PNOPT; https://arxiv.org/pdf/1206.1623.pdf
* group lasso http://myweb.uiowa.edu/pbreheny/7600/s16/notes/4-27.pdf
