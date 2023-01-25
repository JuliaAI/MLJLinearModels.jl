# MLJLinearModels.jl

This is a convenience package gathering functionalities to solve a number of generalised linear regression/classification problems which, inherently, correspond to an optimisation problem of the form

```math
L(y, X\theta) + P(\theta)
```

where:

* ``y`` is the **target** or **response**, a vector of length ``n`` either of real values (_regression_) or integers (_classification_),
* ``X`` is the **design** or **feature** matrix, a matrix of real values of size ``n \times p`` where ``p`` is the number of _features_ or _dimensions_,\
* ``\theta`` is a vector of ``p`` real valued coefficients to determine,
* ``L`` is a **loss function**, a pre-determined function of ``\mathbb R^n \times \mathbb R^n`` to ``\mathbb R^+`` penalising the amplitude of the _residuals_ in a specific way,
* ``P`` is a **penalty function**, a pre-determined function of ``\mathbb R^n`` to ``\mathbb R^+`` penalising the amplitude of the  _coefficients_ in a specific way.

A well known example is the [Ridge regression](https://en.wikipedia.org/wiki/Tikhonov_regularization) where the objective is to minimise:

```math
\|y - X\theta\|_2^2 + \lambda\|\theta\|_2^2.
```

Head to the [Quick Start](/quickstart/) page to get an idea of how this package works.

## What this package aims to do

- make these regressions models "easy to call" and callable in a unified way,
- seamless interface with [MLJ.jl](https://github.com/alan-turing-institute/MLJ.jl),
- focus on performance including in "big data" settings exploiting packages such as [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl), and [IterativeSolvers.jl](https://github.com/JuliaMath/IterativeSolvers.jl),

All models allow to fit an intercept and allow the penalty to be optionally applied on the intercept.
All models attempt to be efficient in terms of memory allocation to avoid unnecessary copies of the data.

## What this package does not aim to do

This package deliberately does not offer the following features

- facilities for data pre-processing (use MLJ for that),
- facilities for hyperparameter tuning (use MLJ for that)
- "significance" statistics (consider [GLM](https://github.com/JuliaStats/GLM.jl) for that)

The last point is important, the package assumes that the user has some principled way of picking an appropriate loss function / penalty. The package makes no assumption of normality etc which befalls the realm of statistics.

You can still build data-driven uncertainty estimates around your parameters if you so desire by using [Bootstrap.jl](https://github.com/juliangehring/Bootstrap.jl).
