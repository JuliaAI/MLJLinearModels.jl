using MLJLinearModels, StableRNGs
using BenchmarkTools, Random, LinearAlgebra
DO_COMPARISONS = false; include("../testutils.jl")

n, p = 50_000, 500
((X, y, θ), (X1, y1, θ1)) = generate_continuous(n, p;  seed=512, sparse=0.5)

# =============== #
# RIDGE FUNCTIONS #
#
# Hv!           ✅
# =============== #

# No fit_intercept
s = R.scratch(X; i=false)
λ = 0.5
ridge = RidgeRegression(λ; fit_intercept=false)
Hv! = R.Hv!(ridge, X, y, s)
v   = randn(p)
Hv  = similar(v)

# Sept 13, 2019 :: 15.300 ms (0 allocations: 0 bytes)
# May 12, 2020 :: 15.785 ms (22 allocations: 432 bytes)
#              :: 15.332 ms (22 allocations: 432 bytes)
#              :: 15.198 ms (22 allocations: 432 bytes)
@btime Hv!($Hv, $v)

# With fit_intercept
s = R.scratch(X; i=true)
ridge = RidgeRegression(λ)
Hv! = R.Hv!(ridge, X, y, s)
v  = randn(p+1)
Hv = similar(v)

# Sept 13, 2019 :: 26.246 ms (5 allocations: 4.22 KiB)
# (decrease in perf due to views, and need a sum on an array)
# May 12, 2020 :: 120.892 ms (34 allocations: 4.77 KiB) NOTE != rng
#              :: 118.999 ms (35 allocations: 4.81 KiB) NOTE != rng
#              :: 114.046 ms (34 allocations: 768 bytes)
@btime Hv!($Hv, $v)

# =============== #
# LASSO FUNCTIONS #
#
# smooth_fg!
# =============== #

lasso = LassoRegression(0.5, fit_intercept=false)
s = R.scratch(X; i=false)
smooth_fg! = R.smooth_fg!(lasso, X, y, s)
v = randn(p)
g = similar(v)

# Sept 13, 2019 :: 15.508 ms (3 allocations: 390.72 KiB)
# May 12, 2020 :: 16.659 ms (32 allocations: 391.34 KiB)
#              :: 15.868 ms (32 allocations: 391.34 KiB)
@btime smooth_fg!($g, $v);

((X, _, _), (X1, y1, θ1)) = generate_continuous(n, p;  seed=512, sparse=0.1)

λ  = 50
lr = LassoRegression(λ)
J  = objective(lr, X, y1)

fista = FISTA()
ista = ISTA()

# <old> initial: ~650ms ; 1725 allocations;  104.24 MiB
# using cache: ; ~650ms ; 1666 allocations;   95.02 MiB
# May 12, 2020 (note RNG is different)
# -- 780.603 ms (4482 allocations: 97.43 MiB)
# -- 772.273 ms (4488 allocations: 98.20 MiB)
@btime θ_fista = fit($lr, $X, $y1, solver=$fista)

# using cache ;  ~590ms ; 1584 allocations;   91.36 MiB
# May 12, 2020 (note RNG is different)
# -- 655.937 ms (4119 allocations: 91.63 MiB)
# -- 644.725 ms (4125 allocations: 92.40 MiB)
@btime θ_ista  = fit($lr, $X, $y1, solver=$ista)

rr = RidgeRegression(λ)
θ_ridge = fit(rr, X, y1)
cg = CG()

# <old> initial: ~ 220ms; 100 allocations; 3.5 MiB
# using cache:   ~ 216ms;  76 allocations; 458.5KiB
# May 12, 2020 :: 982.551 ms (496 allocations: 467.69 KiB)
#              :: 1.005 s (509 allocations: 1.22 MiB)
@btime θ_ridge_cg = fit($rr, $X, $y1, solver=$cg)
