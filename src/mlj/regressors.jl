#= ======================
   LINEAR REGRESSOR (OLS)
   ====================== =#

"""
Standard linear regression model with objective function

``|Xθ - y|₂²/2``

## Parameters

$TYPEDFIELDS

$(example_docstring("LinearRegressor"))
"""
@with_kw_noshow mutable struct LinearRegressor <: MMI.Deterministic
    "whether to fit the intercept or not."
    fit_intercept::Bool    = true
    "type of solver to use (if `nothing` the default is used). The solver is
    Cholesky by default but can be Conjugate-Gradient as well. See `?Analytical`
    for more information."
    solver::Option{Solver} = nothing
end

glr(m::LinearRegressor) = LinearRegression(fit_intercept=m.fit_intercept)

descr(::Type{LinearRegressor}) = "Regression with objective function ``|Xθ - y|₂²/2``."

#= ===============
   RIDGE REGRESSOR
   =============== =#

"""
Ridge regression model with objective function

``|Xθ - y|₂²/2 + n⋅λ|θ|₂²/2``

where ``n`` is the number of samples `size(X, 1)`.
With `scale_penalty_with_samples = false` the objective function is
``|Xθ - y|₂²/2 + λ|θ|₂²/2``.

## Parameters

$TYPEDFIELDS

$(example_docstring("RidgeRegressor"))
"""
@with_kw_noshow mutable struct RidgeRegressor <: MMI.Deterministic
    "strength of the L2 regularisation."
    lambda::Real             = 1.0
    "whether to fit the intercept or not."
    fit_intercept::Bool      = true
    "whether to penalize the intercept."
    penalize_intercept::Bool = false
    "whether to scale the penalty with the number of samples."
    scale_penalty_with_samples::Bool = true
    "type of solver to use (if `nothing` the default is used). The
     solver is Cholesky by default but can be Conjugate-Gradient as
     well. See `?Analytical` for more information."
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
Lasso regression model with objective function

``|Xθ - y|₂²/2 + n⋅λ|θ|₁``

where ``n`` is the number of samples `size(X, 1)`.
With `scale_penalty_with_samples = false` the objective function is
``|Xθ - y|₂²/2 + λ|θ|₁``

## Parameters

$TYPEDFIELDS

$(example_docstring("LassoRegressor"))
"""
@with_kw_noshow mutable struct LassoRegressor <: MMI.Deterministic
    "strength of the L1 regularisation."
    lambda::Real             = 1.0
    "whether to fit the intercept or not."
    fit_intercept::Bool      = true
    "whether to penalize the intercept."
    penalize_intercept::Bool = false
    "whether to scale the penalty with the number of samples."
    scale_penalty_with_samples::Bool = true
    "type of solver to use (if `nothing` the default is used). Either `FISTA` or
    `ISTA` can be used (proximal methods, with/without acceleration)."
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
Elastic net regression model with objective function

``|Xθ - y|₂²/2 + n⋅λ|θ|₂²/2 + n⋅γ|θ|₁``

where ``n`` is the number of samples `size(X, 1)`.
With `scale_penalty_with_samples = false` the objective function is
``|Xθ - y|₂²/2 + λ|θ|₂²/2 + γ|θ|₁``

## Parameters

$TYPEDFIELDS

$(example_docstring("ElasticNetRegressor"))
"""
@with_kw_noshow mutable struct ElasticNetRegressor <: MMI.Deterministic
    "strength of the L2 regularisation."
    lambda::Real             = 1.0
    "strength of the L1 regularisation."
    gamma::Real              = 0.0
    "whether to fit the intercept or not."
    fit_intercept::Bool      = true
    "whether to penalize the intercept."
    penalize_intercept::Bool = false
    "whether to scale the penalty with the number of samples."
    scale_penalty_with_samples::Bool = true
    "type of solver to use (if `nothing` the default is used). Either `FISTA` or
    `ISTA` can be used (proximal methods, with/without acceleration)."
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
Robust regression model with objective function

``∑ρ(Xθ - y) + n⋅λ|θ|₂² + n⋅γ|θ|₁``

where ``ρ`` is a robust loss function (e.g. the Huber function) and
``n`` is the number of samples `size(X, 1)`.
With `scale_penalty_with_samples = false` the objective function is
``∑ρ(Xθ - y) + λ|θ|₂² + γ|θ|₁``.

## Parameters

$TYPEDFIELDS

$(example_docstring("RobustRegressor"))
"""
@with_kw_noshow mutable struct RobustRegressor <: MMI.Deterministic
    "the type of robust loss to use (see `HuberRho`, `TalwarRho`, ...)"
    rho::RobustRho           = HuberRho(0.1)
    "strength of the regulariser if `penalty` is `:l2` or `:l1`.
    Strength of the L2 regulariser if `penalty` is `:en`."
    lambda::Real             = 1.0
    "strength of the L1 regulariser if `penalty` is `:en`."
    gamma::Real              = 0.0
    "the penalty to use, either `:l2`, `:l1`, `:en` (elastic net) or `:none`."
    penalty::SymStr          = :l2
    "whether to fit the intercept or not."
    fit_intercept::Bool      = true
    "whether to penalize the intercept."
    penalize_intercept::Bool = false
    "whether to scale the penalty with the number of samples."
    scale_penalty_with_samples::Bool = true
    "type of solver to use, default if `nothing`."
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
Huber Regression is the same as `RobustRegressor` but with the robust loss
set to `HuberRho`.

## Parameters

$TYPEDFIELDS

$(example_docstring("HuberRegressor"))
"""
@with_kw_noshow mutable struct HuberRegressor <: MMI.Deterministic
    "parametrises the `HuberRho` function (radius of the ball within which the loss
is a quadratic loss)"
    delta::Real              = 0.5
    "strength of the regulariser if `penalty` is `:l2` or `:l1`.
    Strength of the L2 regulariser if `penalty` is `:en`."
    lambda::Real             = 1.0
    "strength of the L1 regulariser if `penalty` is `:en`."
    gamma::Real              = 0.0
    "the penalty to use, either `:l2`, `:l1`, `:en` (elastic net) or `:none`."
    penalty::SymStr          = :l2
    "whether to fit the intercept or not."
    fit_intercept::Bool      = true
    "whether to penalize the intercept."
    penalize_intercept::Bool = false
    "whether to scale the penalty with the number of samples."
    scale_penalty_with_samples::Bool = true
    "type of solver to use, default if `nothing`."
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
Quantile Regression is the same as `RobustRegressor` but with the robust
loss set to `QuantileRho`.

## Parameters

$TYPEDFIELDS

$(example_docstring("QuantileRegressor"))
"""
@with_kw_noshow mutable struct QuantileRegressor <: MMI.Deterministic
    "parametrises the `QuantileRho` function (indicating the  quantile to use
with default `0.5` for the median regression)"
    delta::Real              = 0.5
    "strength of the regulariser if `penalty` is `:l2` or `:l1`.
    Strength of the L2 regulariser if `penalty` is `:en`."
    lambda::Real             = 1.0
    "strength of the L1 regulariser if `penalty` is `:en`."
    gamma::Real              = 0.0
    "the penalty to use, either `:l2`, `:l1`, `:en` (elastic net) or `:none`."
    penalty::SymStr          = :l2
    "whether to fit the intercept or not."
    fit_intercept::Bool      = true
    "whether to penalize the intercept."
    penalize_intercept::Bool = false
    "whether to scale the penalty with the number of samples."
    scale_penalty_with_samples::Bool = true
    "type of solver to use, default if `nothing`."
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
Least Absolute Deviation regression with with objective function

``∑ρ(Xθ - y) + n⋅λ|θ|₂² + n⋅γ|θ|₁``

where ``ρ`` is the absolute loss and
``n`` is the number of samples `size(X, 1)`.
With `scale_penalty_with_samples = false` the objective function is
``∑ρ(Xθ - y) + λ|θ|₂² + γ|θ|₁``


See also `RobustRegressor`.

## Parameters

$TYPEDFIELDS

$(example_docstring("LADRegressor"))
"""
@with_kw_noshow mutable struct LADRegressor <: MMI.Deterministic
    "strength of the regulariser if `penalty` is `:l2` or `:l1`.
    Strength of the L2 regulariser if `penalty` is `:en`."
    lambda::Real             = 1.0
    "strength of the L1 regulariser if `penalty` is `:en`."
    gamma::Real              = 0.0
    "the penalty to use, either `:l2`, `:l1`, `:en` (elastic net) or `:none`."
    penalty::SymStr          = :l2
    "whether to fit the intercept or not."
    fit_intercept::Bool      = true
    "whether to penalize the intercept."
    penalize_intercept::Bool = false
    "whether to scale the penalty with the number of samples."
    scale_penalty_with_samples::Bool = true
    "type of solver to use, default if `nothing`."
    solver::Option{Solver}   = nothing
end

glr(m::LADRegressor) =
    LADRegression(m.lambda, m.gamma;
                  penalty=Symbol(m.penalty),
                  fit_intercept=m.fit_intercept,
                  penalize_intercept=m.penalize_intercept,
                  scale_penalty_with_samples=m.scale_penalty_with_samples)

descr(::Type{LADRegressor}) = "Robust regression with objective ``∑ρ(Xθ - y) + λ|θ|₂² + γ|θ|₁`` where `ρ` is the Absolute Loss."
