export Analytical, CG,
        Newton, NewtonCG,
        LBFGS,
        ProxGrad, FISTA, ISTA,
        IWLSCG,
        ADMM, FADMM # NOTE: these do not work currently

# =====
# TODO
# * all - pick linesearch
# * NewtonCG number of inner iter
# * FISTA field to enforce descent
# ====

abstract type Solver end

# ===================== analytical.jl

"""
$SIGNATURES

Analytical solver (Cholesky). If the `iterative` parameter is set to `true`
then a CG solver is used. The CG solver is matrix-free and should be preferred
in "large scale" cases (when the hat matrix `X'X` is "big").

## Parameters

* `iterative` (Bool): whether to use CG (iterative) or not
* `max_inner` (Int): in the iterative mode, how many inner iterations to do.
"""
@with_kw struct Analytical <: Solver
    iterative::Bool = false
    max_inner::Int  = 200
end
CG() = Analytical(; iterative=true)

# ===================== newton.jl

"""
$SIGNATURES

Newton solver. This is a full Hessian solver and should be avoided for
"large scale" cases.
"""
struct Newton <: Solver end

"""
$SIGNATURES

Newton CG solver. This is the same as the Newton solver except that instead
of solving systems of the form `H\\b` where `H` is the full Hessian, it uses
a matrix-free conjugate gradient approach to solving that system. This should
generally be preferred for larger scale cases.
"""
struct NewtonCG <: Solver end

"""
$SIGNATURES

LBFGS quasi-Newton solver. See [the wikipedia entry](https://en.wikipedia.org/wiki/Limited-memory_BFGS).
"""
struct LBFGS <: Solver end

# struct BFGS <: Solver end

# ===================== pgrad.jl

"""
$SIGNATURES

Proximal Gradient solver for non-smooth objective functions.

## Parameters

* `accel` (Bool): whether to use Nesterov-style acceleration
* `max_iter` (Int): number of overall iterations
* `tol` (Float64): tolerance for the relative change θ ie `norm(θ-θ_)/norm(θ)`
* `max_inner`: number of inner steps when searching for a stepsize in the
               backtracking step
* `beta`: rate of shrinkage in the backtracking step (between 0 and 1)
"""
@with_kw struct ProxGrad <: Solver
    accel::Bool    = false # use Nesterov style acceleration (see also FISTA)
    max_iter::Int  = 1000  # max number of overall iterations
    tol::Float64   = 1e-4  # tol relative change of θ i.e. norm(θ-θ_)/norm(θ)
    max_inner::Int = 100   # β^max_inner should be > 1e-10
    beta::Float64  = 0.8   # in (0, 1); shrinkage in the backtracking step
end

FISTA(; kwa...) = ProxGrad(;accel = true, kwa...)
ISTA(; kwa...)  = ProxGrad(;accel = false, kwa...)

# ===================== iwls.jl

"""
$SIGNATURES

Iteratively Reweighted Least Square with Conjugate Gradient. This is the
standard (expensive) IWLS but with more efficient solves to avoid full matrix
computations.

## Parameters

* `max_iter` (Int): number of max iterations (outer)
* `max_inner` (Int): number of iterations for the CG solves
* `tol` (Float64): tolerance for the relative change θ ie `norm(θ-θ_)/norm(θ)`
* `damping` (Float64): how much to trust iterates (1=full trust)
* `threshold` (Float64): threshold for the residuals
"""
@with_kw struct IWLSCG <: Solver
    max_iter::Int      = 100
    max_inner::Int     = 200
    tol::Float64       = 1e-4
    damping::Float64   = 1.0   # should be between 0 and 1, 1 = trust iterates
    threshold::Float64 = 1e-6  # thresh for residuals; used eg in quantile reg
end

# ===================== admm.jl

# @with_kw struct ADMM <: Solver
#     max_iter::Int  = 100
#     tol::Float64   = 1e-4
#     alpha::Float64 = 1.5  # over-relaxation (recommended between 1.5 and 1.8)
#     rho::Float64   = 1.0  # Lagrangian parameter (should be decreased if poor condition)
# end
#
# @with_kw struct FADMM <: Solver
#     max_iter::Int  = 100
#     tol::Float64   = 1e-4
#     eta::Float64   = 0.999 # restart parameter
#     rho::Float64   = 1.0   # Lagrangian parameter (should be decreased if poor condition)
#     tau::Float64   = 2.0   # Increase / decrease of ρ should be > 1
#     mu::Float64    = 10.0  #
# end
