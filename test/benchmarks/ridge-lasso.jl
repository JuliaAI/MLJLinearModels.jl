# WIP WIP !

using MLJLinearModels
using BenchmarkTools, Random, LinearAlgebra
DO_COMPARISONS = false; include("../testutils.jl")

n, p = 50_000, 500
((X, _, _), (X_, y1, θ1)) = generate_continuous(n, p;  seed=512, sparse=0.1)

λ  = 50
lr = LassoRegression(λ)
J  = objective(lr, X, y1)

fista = FISTA()
ista = ISTA()

@btime θ_fista = fit($lr, $X, $y1, solver=$fista)

# <old> initial: ~650ms ; 1725 allocations;  104.24 MiB
# using cache: ; ~650ms ; 1666 allocations;   95.02 MiB

@btime θ_ista  = fit($lr, $X, $y1, solver=$ista)

# using cache ;  ~590ms ; 1584 allocations;   91.36 MiB

rr = RidgeRegression(λ)
θ_ridge = fit(rr, X, y1)
cg = CG()

@btime θ_ridge_cg = fit($rr, $X, $y1, solver=$cg)

# <old> initial: ~ 220ms; 100 allocations; 3.5 MiB
# using cache:   ~ 216ms;  76 allocations; 458.5KiB
