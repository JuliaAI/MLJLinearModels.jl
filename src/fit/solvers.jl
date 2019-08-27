export Analytical, CG,
        Newton, NewtonCG,
        LBFGS,
        ProxGrad, FISTA, ISTA,
        IWLSCG

# =====
# TODO
# * all - pick linesearch
# * NewtonCG number of inner iter
# * FISTA field to enforce descent
# ====

abstract type Solver end

# ===================== analytical.jl

@with_kw struct Analytical <: Solver
    iterative::Bool = false
    max_inner::Int  = 200
end

CG() = Analytical(; iterative=true)

# ===================== newton.jl

struct Newton <: Solver end

struct NewtonCG <: Solver end

struct LBFGS <: Solver end

# struct BFGS <: Solver end

# ===================== pgrad.jl

@with_kw struct ProxGrad <: Solver
    accel::Bool    = false # use Nesterov style acceleration (see also FISTA)
    max_iter::Int  = 1000  # max number of overall iterations
    tol::Float64   = 1e-4  # tolerance over relative change of θ i.e. norm(θ-θ_)/norm(θ)
    max_inner::Int = 100   # β^max_inner should be > 1e-10
    β::Float64     = 0.8   # in (0, 1); shrinkage in the backtracking step
end

FISTA(; kwa...) = ProxGrad(;accel = true, kwa...)
ISTA(; kwa...)  = ProxGrad(;accel = false, kwa...)


# ===================== iwls.jl

@with_kw struct IWLSCG <: Solver
    max_iter::Int    = 100
    max_inner::Int   = 200
    tol::Float64     = 1e-4
    damping::Float64 = 1.0   # should be between 0 and 1, 1 = trust iterates
end
