using MLJLinearModels
using BenchmarkTools, Random, LinearAlgebra
DO_COMPARISONS = false; include("../testutils.jl")

n, p = 50_000, 500
((X, y, θ), (X1, y1, θ1)) = generate_binary(n, p; seed=525)

λ = 5.0
lr = LogisticRegression(λ)

# XXX need to check whether this is doing the right thing or not... might
# be worth comparing with your own implementation using a simple backtracking
# line search or your own implementation of Hanger-Zhang

# newtoncg = NewtonCG()
# θ_ncg = fit(lr, X, y1, solver=newtoncg)

lbfgs = LBFGS()

# by far the fastest
@btime θ_lbfgs = fit($lr, $X, $y1, solver=$lbfgs)

# <old>       1.421 s (2866 allocations: 142.33 MiB)
# with cache  1.357 s (1912 allocations:  30.15 MiB)
