# Newton and quasi Newton solvers

## LOGISTIC + 0/L2 and Huber ==============

"""
$SIGNATURES

Fit a GLR using Newton's method.

## Complexity

Assuming `n` dominates `p`, O(κnp²), dominated by the construction of the
Hessian at each step with κ the number of Newton steps.
"""
function _fit(::Type{T}, glr::GLR{<:Union{LogisticLoss,RobustLoss},<:L2R},
    solver::Newton, X, y, scratch) where {T<:Real}
    _,p,_ = npc(scratch)
    θ₀    = zeros(T, p)
    _fgh! = fgh!(T, glr, X, y, scratch)
    opt   = Optim.only_fgh!(_fgh!)
    res   = Optim.optimize(opt, θ₀, Optim.Newton(; solver.newton_options...),
                           solver.optim_options)
    return Optim.minimizer(res)
end

function _fit(glr::GLR{<:Union{LogisticLoss,RobustLoss},<:L2R},
    solver::Newton, X, y, scratch)
    return _fit(eltype(X), glr, solver, X, y, scratch)
end

"""
$SIGNATURES

Fit a GLR using Newton's method combined with an iterative solver  (conjugate
gradient) to solve the Newton steps (∇²f)⁻¹∇f.

## Complexity

Assuming `n` dominates `p`, O(κ₁κ₂np), dominated by the application of the
Hessian at each step where κ₁ is the number of Newton steps and κ₂ is the
average number of CG steps per Newton step (which is at most p).
"""
function _fit(::Type{T}, glr::GLR{<:Union{LogisticLoss,RobustLoss},<:L2R},
    solver::NewtonCG, X, y, scratch) where {T<:Real}
    _,p,_ = npc(scratch)
    θ₀    = zeros(T, p)
    _f    = objective(glr, X, y)
    _fg!  = (g, θ) -> fgh!(T, glr, X, y, scratch)(0.0, g, nothing, θ) # Optim#738
    _Hv!  = Hv!(T, glr, X, y, scratch)
    opt   = Optim.TwiceDifferentiableHV(_f, _fg!, _Hv!, θ₀)
    res   = Optim.optimize(opt, θ₀, Optim.KrylovTrustRegion(; solver.newtoncg_options...),
                           solver.optim_options)
    return Optim.minimizer(res)
end

"""
$SIGNATURES

Fit a GLR using LBFGS.

## Complexity

Assuming `n` dominates `p`, O(κnp), dominated by the computation of the
gradient at each step with κ the number of LBFGS steps.
"""
function _fit(::Type{T}, glr::GLR{<:Union{LogisticLoss,RobustLoss},<:L2R},
    solver::LBFGS, X, y, scratch) where {T<:Real}
    _,p,_ = npc(scratch)
    θ₀    = zeros(T, p)
    _fg!  = (f, g, θ) -> fgh!(T, glr, X, y, scratch)(f, g, nothing, θ)
    opt   = Optim.only_fg!(_fg!)
    res   = Optim.optimize(opt, θ₀, Optim.LBFGS(; solver.lbfgs_options...),
                           solver.optim_options)
    return Optim.minimizer(res)
end


## MULTINOMIAL + 0/L2 ==============

"""
$SIGNATURES

Fit a multiclass GLR using Newton's method with an iterative solver (conjugate
gradient).

## Complexity

Assuming `n` dominates `p`, O(κ₁κ₂npc), where `c` is the number of classes. The
computations are dominated by the application of the Hessian at each step with
κ₁ the number of Newton steps and κ₂ the average number of CG steps per Newton
step.
"""
function _fit(::Type{T}, glr::GLR{<:MultinomialLoss,<:L2R}, solver::NewtonCG,
    X, y, scratch) where {T<:Real}
    _,p,c = npc(scratch)
    θ₀    = zeros(T, p * c)
    _f    = objective(glr, X, y; c=c)
    _fg!  = (g, θ) -> fg!(T, glr, X, y, scratch)(T(0.0), g, θ) # XXX: Optim.jl/738
    _Hv!  = Hv!(T, glr, X, y, scratch)
    opt   = Optim.TwiceDifferentiableHV(_f, _fg!, _Hv!, θ₀)
    res   = Optim.optimize(opt, θ₀, Optim.KrylovTrustRegion(; solver.newtoncg_options...),
                           solver.optim_options)
    return Optim.minimizer(res)
end

function _fit(glr::GLR{<:MultinomialLoss,<:L2R}, solver::NewtonCG, X, y, scratch)
    return _fit(eltype(X), glr, solver, X, y, scratch)
end

"""
$SIGNATURES

Fit a multiclass GLR using LBFGS.

## Complexity

Assuming `n` dominates `p`, O(κnpc), with `c` the number of classes, dominated
by the computation of the gradient at each step with κ the number of LBFGS
steps.
"""
function _fit(::Type{T}, glr::GLR{<:MultinomialLoss,<:L2R}, solver::LBFGS,
    X, y, scratch) where {T<:Real}
    _,p,c = npc(scratch)
    θ₀    = zeros(T, p * c)
    _fg!  = fg!(T, glr, X, y, scratch)
    opt   = Optim.only_fg!(_fg!)
    res   = Optim.optimize(opt, θ₀, Optim.LBFGS(; solver.lbfgs_options...),
                           solver.optim_options)
    return Optim.minimizer(res)
end

function _fit(glr::GLR{<:MultinomialLoss,<:L2R}, solver::LBFGS, X, y, scratch)
    return _fit(eltype(X), glr, solver, X, y, scratch)
end
