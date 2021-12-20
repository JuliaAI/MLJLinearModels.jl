#= ======================
   LINEAR REGRESSOR (OLS)
   ====================== =#

"""
$SIGNATURES

Standard linear regression model.

## Parameters

* `fit_intercept` (Bool): whether to fit the intercept or not.
* `solver`: type of solver to use (if `nothing` the default is used). The
            solver is Cholesky by default but can be Conjugate-Gradient as
            well. See `?Analytical` for more information.

"""
@with_kw_noshow mutable struct LinearRegressor <: MMI.Deterministic
    fit_intercept::Bool    = true
    solver::Option{Solver} = nothing
end

glr(m::LinearRegressor) = LinearRegression(fit_intercept=m.fit_intercept)

descr(::Type{LinearRegressor}) = "Regression with objective function ``|Xθ - y|₂²/2``."

#= ===============
   RIDGE REGRESSOR
   =============== =#

"""
$SIGNATURES

Ridge regression model with objective function

``|Xθ - y|₂²/2 + n⋅λ|θ|₂²/2``


## Parameters

* `lambda` (Real): strength of the L2 regularisation.
* `fit_intercept` (Bool): whether to fit the intercept or not.
* `penalize_intercept` (Bool): whether to penalize the intercept.
* `solver`: type of solver to use (if `nothing` the default is used). The
            solver is Cholesky by default but can be Conjugate-Gradient as
            well. See `?Analytical` for more information.
"""
@with_kw_noshow mutable struct RidgeRegressor <: MMI.Deterministic
    lambda::Real             = 1.0
    fit_intercept::Bool      = true
    penalize_intercept::Bool = false
    scale_penalty_with_samples::Bool = true
    solver::Option{Solver}   = nothing
end

glr(m::RidgeRegressor) =
    RidgeRegression(m.lambda,
                    fit_intercept=m.fit_intercept,
                    penalize_intercept=m.penalize_intercept,
                    scale_penalty_with_samples=m.scale_penalty_with_samples)

descr(::Type{RidgeRegressor}) = "Regression with objective function ``|Xθ - y|₂²/2 + λ|θ|₂²/2``."

#= ===============
   LASSO REGRESSOR
   =============== =#

"""
$SIGNATURES

Lasso regression model with objective function

``|Xθ - y|₂²/2 + λ|θ|₁``

## Parameters

* `lambda` (Real): strength of the L1 regularisation.
* `fit_intercept` (Bool): whether to fit the intercept or not.
* `penalize_intercept` (Bool): whether to penalize the intercept.
* `solver`: type of solver to use (if `nothing` the default is used). Either
            `FISTA` or `ISTA` can be used (proximal methods, with/without
            acceleration).
"""
@with_kw_noshow mutable struct LassoRegressor <: MMI.Deterministic
    lambda::Real             = 1.0
    fit_intercept::Bool      = true
    penalize_intercept::Bool = false
    scale_penalty_with_samples::Bool = true
    solver::Option{Solver}   = nothing
end

glr(m::LassoRegressor) =
    LassoRegression(m.lambda,
                    fit_intercept=m.fit_intercept,
                    penalize_intercept=m.penalize_intercept,
                    scale_penalty_with_samples=m.scale_penalty_with_samples)

descr(::Type{LassoRegressor}) = "Regression with objective function ``|Xθ - y|₂²/2 + λ|θ|₁``."

#= =====================
   ELASTIC NET REGRESSOR
   ===================== =#

"""
$SIGNATURES

Elastic net regression model with objective function

``|Xθ - y|₂²/2 + λ|θ|₂²/2 + γ|θ|₁``

## Parameters

* `lambda` (Real): strength of the L2 regularisation.
* `gamma` (Real): strength of the L1 regularisation.
* `fit_intercept` (Bool): whether to fit the intercept or not.
* `penalize_intercept` (Bool): whether to penalize the intercept.
* `solver`: type of solver to use (if `nothing` the default is used). Either
            `FISTA` or `ISTA` can be used (proximal methods, with/without
            acceleration).
"""
@with_kw_noshow mutable struct ElasticNetRegressor <: MMI.Deterministic
    lambda::Real             = 1.0
    gamma::Real              = 0.0
    fit_intercept::Bool      = true
    penalize_intercept::Bool = false
    scale_penalty_with_samples::Bool = true
    solver::Option{Solver}   = nothing
end

glr(m::ElasticNetRegressor) =
    ElasticNetRegression(m.lambda, m.gamma,
                         fit_intercept=m.fit_intercept,
                         penalize_intercept=m.penalize_intercept,
                         scale_penalty_with_samples=m.scale_penalty_with_samples)

descr(::Type{ElasticNetRegressor}) = "Regression with objective function ``|Xθ - y|₂²/2 + λ|θ|₂²/2 + γ|θ|₁``."

#= ==========================
   ROBUST REGRESSOR (General)
   ========================== =#

"""
$SIGNATURES

Robust regression model with objective function

``∑ρ(Xθ - y) + λ|θ|₂² + γ|θ|₁``

where `ρ` is a robust loss function (e.g. the Huber function).

## Parameters

* `rho` (RobustRho): the type of robust loss to use (see `HuberRho`,
                     `TalwarRho`, ...)
* `penalty` (Symbol or String): the penalty to use, either `:l2`, `:l1`, `:en`
                                (elastic net) or `:none`. (Default: `:l2`)
* `lambda` (Real): strength of the regulariser if `penalty` is `:l2` or `:l1`.
                   Strength of the L2 regulariser if `penalty` is `:en`.
* `gamma` (Real): strength of the L1 regulariser if `penalty` is `:en`.
* `fit_intercept` (Bool): whether to fit an intercept (Default: `true`)
* `penalize_intercept` (Bool): whether to penalize intercept (Default: `false`)
* `solver` (Solver): type of solver to use, default if `nothing`.
"""
@with_kw_noshow mutable struct RobustRegressor <: MMI.Deterministic
    rho::RobustRho           = HuberRho(0.1)
    lambda::Real             = 1.0
    gamma::Real              = 0.0
    penalty::SymStr          = :l2
    fit_intercept::Bool      = true
    penalize_intercept::Bool = false
    scale_penalty_with_samples::Bool = true
    solver::Option{Solver}   = nothing
end

glr(m::RobustRegressor) =
    RobustRegression(m.rho, m.lambda, m.gamma;
                     penalty=Symbol(m.penalty),
                     fit_intercept=m.fit_intercept,
                     penalize_intercept=m.penalize_intercept,
                     scale_penalty_with_samples=m.scale_penalty_with_samples)

descr(::Type{RobustRegressor}) = "Robust regression with objective ``∑ρ(Xθ - y) + λ|θ|₂² + γ|θ|₁`` for a given robust `ρ`."

#= ===============
   HUBER REGRESSOR
   =============== =#

"""
$SIGNATURES

Huber Regression, see `RobustRegressor`, it's the same but with the robust loss
set to `HuberRho`.  The parameters are the same apart from `delta` which
parametrises the `HuberRho` function (radius of the ball within which the loss
is a quadratic loss).
"""
@with_kw_noshow mutable struct HuberRegressor <: MMI.Deterministic
    delta::Real              = 0.5
    lambda::Real             = 1.0
    gamma::Real              = 0.0
    penalty::SymStr          = :l2
    fit_intercept::Bool      = true
    penalize_intercept::Bool = false
    scale_penalty_with_samples::Bool = true
    solver::Option{Solver}   = nothing
end

glr(m::HuberRegressor) =
    HuberRegression(m.delta, m.lambda, m.gamma;
                    penalty=Symbol(m.penalty),
                    fit_intercept=m.fit_intercept,
                    penalize_intercept=m.penalize_intercept,
                    scale_penalty_with_samples=m.scale_penalty_with_samples)

descr(::Type{HuberRegressor}) = "Robust regression with objective ``∑ρ(Xθ - y) + λ|θ|₂² + γ|θ|₁`` where `ρ` is the Huber Loss."

#= ==================
   QUANTILE REGRESSOR
   ================== =#

"""
$SIGNATURES

Quantile Regression, see `RobustRegressor`, it's the same but with the robust
loss set to `QuantileRho`.  The parameters are the same apart from `delta`
which parametrises the `QuantileRho` function (indicating the  quantile to use
with default `0.5` for the median regression).
"""
@with_kw_noshow mutable struct QuantileRegressor <: MMI.Deterministic
    delta::Real              = 0.5
    lambda::Real             = 1.0
    gamma::Real              = 0.0
    penalty::SymStr          = :l2
    fit_intercept::Bool      = true
    penalize_intercept::Bool = false
    scale_penalty_with_samples::Bool = true
    solver::Option{Solver}   = nothing
end

glr(m::QuantileRegressor) =
    QuantileRegression(m.delta, m.lambda, m.gamma;
                       penalty=Symbol(m.penalty),
                       fit_intercept=m.fit_intercept,
                       penalize_intercept=m.penalize_intercept,
                       scale_penalty_with_samples=m.scale_penalty_with_samples)

descr(::Type{QuantileRegressor}) = "Robust regression with objective ``∑ρ(Xθ - y) + λ|θ|₂² + γ|θ|₁`` where `ρ` is the Quantile Loss."

#= ==================================
   LEAST ABSOLUTE DEVIATION REGRESSOR
   ================================== =#

"""
$SIGNATURES

Least Absolute Deviation regression with with objective function

``∑ρ(Xθ - y) + λ|θ|₂² + γ|θ|₁``

where `ρ` is the absolute loss.

See also `RobustRegressor`.
"""
@with_kw_noshow mutable struct LADRegressor <: MMI.Deterministic
    lambda::Real             = 1.0
    gamma::Real              = 0.0
    penalty::SymStr          = :l2
    fit_intercept::Bool      = true
    penalize_intercept::Bool = false
    scale_penalty_with_samples::Bool = true
    solver::Option{Solver}   = nothing
end

glr(m::LADRegressor) =
    LADRegression(m.lambda, m.gamma;
                  penalty=Symbol(m.penalty),
                  fit_intercept=m.fit_intercept,
                  penalize_intercept=m.penalize_intercept,
                  scale_penalty_with_samples=m.scale_penalty_with_samples)

descr(::Type{LADRegressor}) = "Robust regression with objective ``∑ρ(Xθ - y) + λ|θ|₂² + γ|θ|₁`` where `ρ` is the Absolute Loss."
