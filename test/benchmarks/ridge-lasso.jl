using MLJLinearModels
using BenchmarkTools, Random, LinearAlgebra
DO_COMPARISONS = false; include("../testutils.jl")

n, p = 50_000, 500
((X, y, θ), (X_, y1, θ1)) = generate_continuous(n, p;  seed=512, sparse=0.5)

# =============== #
# RIDGE FUNCTIONS #
#
# Hv!           ✅
# =============== #

# No fit_intercept
R.allocate(n, p)
ridge = RidgeRegression(0.5; fit_intercept=false)
Hv! = R.Hv!(ridge, X, y)
v   = randn(p)
Hv  = similar(v)

# Sept 13, 2019 :: 15.300 ms (0 allocations: 0 bytes)
@btime Hv!($Hv, $v)

# With fit_intercept
R.allocate(n, p+1)
ridge = RidgeRegression(λ)
Hv! = R.Hv!(ridge, X, y)
v  = randn(p+1)
Hv = similar(v)

# Sept 13, 2019 :: 26.246 ms (5 allocations: 4.22 KiB)
# (decrease in perf due to views, and need a sum on an array)
@btime Hv!($Hv, $v)

# =============== #
# LASSO FUNCTIONS #
#
# smooth_fg!
# =============== #

# No fit_intercept
R.allocate(n, p)

lasso = LassoRegression(0.5, fit_intercept=false)
smooth_fg! = R.smooth_fg!(lasso, X, y)
v = randn(p)
g = similar(v)

# Sept 13, 2019 :: 15.508 ms (3 allocations: 390.72 KiB)
@btime smooth_fg!($g, $v);

@btime R.get_residuals!($X, $θ, $y)

r = R.get_residuals!(X, θ, y)

@btime R.apply_Xt!(g, X, r)

@btime (g .+= 0.5 .* θ)





# =======
=======
((X, _, _), (X_, y1, θ1)) = generate_continuous(n, p;  seed=512, sparse=0.1)

λ  = 50
lr = LassoRegression(λ)
J  = objective(lr, X, y1)
>>>>>>> master

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
