#= ======================
   LINEAR REGRESSOR (OLS)
   ====================== =#

"""
$(doc_header(LinearRegressor))

This model provides standard linear regression with objective function

``|Xθ - y|₂²/2``

$DOC_SOLVERS

# Training data

In MLJ or MLJBase, bind an instance `model` to data with

    mach = machine(model, X, y)

where:

- `X` is any table of input features (eg, a `DataFrame`) whose columns
  have `Continuous` scitype; check column scitypes with `schema(X)`

- `y` is the target, which can be any `AbstractVector` whose element scitype is
  `Continuous`; check the scitype with `scitype(y)`


Train the machine using `fit!(mach, rows=...)`.


# Hyperparameters

$TYPEDFIELDS

$(example_docstring("LinearRegressor"))
"""
@with_kw_noshow mutable struct LinearRegressor <: MMI.Deterministic
    "whether to fit the intercept or not."
    fit_intercept::Bool    = true
    """"any instance of `MLJLinearModels.Analytical`. Use `Analytical()`
    for Cholesky and `CG()=Analytical(iterative=true)` for conjugate-gradient.

    If `solver = nothing` (default) then `Analytical()` is used. """
    solver::Option{Solver} = nothing
end

glr(m::LinearRegressor) = LinearRegression(fit_intercept=m.fit_intercept)


#= ===============
   RIDGE REGRESSOR
   =============== =#

"""
$(doc_header(RidgeRegressor))

Ridge regression is a linear model with objective function

``|Xθ - y|₂²/2 + n⋅λ|θ|₂²/2``

where ``n`` is the number of observations.

If `scale_penalty_with_samples = false` then the objective function is instead

``|Xθ - y|₂²/2 + λ|θ|₂²/2``.

$DOC_SOLVERS

# Training data

In MLJ or MLJBase, bind an instance `model` to data with

    mach = machine(model, X, y)

where:

- `X` is any table of input features (eg, a `DataFrame`) whose columns
  have `Continuous` scitype; check column scitypes with `schema(X)`

- `y` is the target, which can be any `AbstractVector` whose element scitype is
  `Continuous`; check the scitype with `scitype(y)`


Train the machine using `fit!(mach, rows=...)`.


# Hyperparameters

$TYPEDFIELDS

$(example_docstring("RidgeRegressor"))

See also [`ElasticNetRegressor`](@ref).

"""
@with_kw_noshow mutable struct RidgeRegressor <: MMI.Deterministic
    "strength of the L2 regularization."
    lambda::Real             = 1.0
    "whether to fit the intercept or not."
    fit_intercept::Bool      = true
    "whether to penalize the intercept."
    penalize_intercept::Bool = false
    "whether to scale the penalty with the number of observations."
    scale_penalty_with_samples::Bool = true
    """any instance of `MLJLinearModels.Analytical`. Use `Analytical()` for
    Cholesky and `CG()=Analytical(iteration=true)` for conjugate-gradient.
    If `solver = nothing` (default) then `Analytical()` is used. """
    solver::Option{Solver}   = nothing
end

glr(m::RidgeRegressor) =
    RidgeRegression(m.lambda,
                    fit_intercept=m.fit_intercept,
                    penalize_intercept=m.penalize_intercept,
                    scale_penalty_with_samples=m.scale_penalty_with_samples)

#= ===============
   LASSO REGRESSOR
   =============== =#

"""
$(doc_header(LassoRegressor))

Lasso regression is a linear model with objective function

``|Xθ - y|₂²/2 + n⋅λ|θ|₁``

where ``n`` is the number of observations.

If `scale_penalty_with_samples = false` the objective function is

``|Xθ - y|₂²/2 + λ|θ|₁``.

$DOC_SOLVERS

# Training data

In MLJ or MLJBase, bind an instance `model` to data with

    mach = machine(model, X, y)

where:

- `X` is any table of input features (eg, a `DataFrame`) whose columns
  have `Continuous` scitype; check column scitypes with `schema(X)`

- `y` is the target, which can be any `AbstractVector` whose element scitype is
  `Continuous`; check the scitype with `scitype(y)`


Train the machine using `fit!(mach, rows=...)`.


# Hyperparameters

$TYPEDFIELDS

$(example_docstring("LassoRegressor"))

See also [`ElasticNetRegressor`](@ref).

"""
@with_kw_noshow mutable struct LassoRegressor <: MMI.Deterministic
    "strength of the L1 regularization."
    lambda::Real             = 1.0
    "whether to fit the intercept or not."
    fit_intercept::Bool      = true
    "whether to penalize the intercept."
    penalize_intercept::Bool = false
    "whether to scale the penalty with the number of observations."
    scale_penalty_with_samples::Bool = true
    """any instance of `MLJLinearModels.ProxGrad`.
    If `solver=nothing` (default) then `ProxGrad(accel=true)` (FISTA) is used.
    Solver aliases: `FISTA(; kwargs...) = ProxGrad(accel=true, kwargs...)`,
    `ISTA(; kwargs...) = ProxGrad(accel=false, kwargs...)`. """
    solver::Option{Solver}   = nothing
end

glr(m::LassoRegressor) =
    LassoRegression(m.lambda,
                    fit_intercept=m.fit_intercept,
                    penalize_intercept=m.penalize_intercept,
                    scale_penalty_with_samples=m.scale_penalty_with_samples)


#= =====================
   ELASTIC NET REGRESSOR
   ===================== =#

"""
$(doc_header(ElasticNetRegressor))

Elastic net is a linear model with objective function

``|Xθ - y|₂²/2 + n⋅λ|θ|₂²/2 + n⋅γ|θ|₁``

where ``n`` is the number of observations.

If  `scale_penalty_with_samples = false` the objective function is instead

``|Xθ - y|₂²/2 + λ|θ|₂²/2 + γ|θ|₁``.

$DOC_SOLVERS

# Training data

In MLJ or MLJBase, bind an instance `model` to data with

    mach = machine(model, X, y)

where:

- `X` is any table of input features (eg, a `DataFrame`) whose columns
  have `Continuous` scitype; check column scitypes with `schema(X)`

- `y` is the target, which can be any `AbstractVector` whose element scitype is
  `Continuous`; check the scitype with `scitype(y)`


Train the machine using `fit!(mach, rows=...)`.


# Hyperparameters

$TYPEDFIELDS

$(example_docstring("ElasticNetRegressor"))

See also [`LassoRegressor`](@ref).

"""
@with_kw_noshow mutable struct ElasticNetRegressor <: MMI.Deterministic
    "strength of the L2 regularization."
    lambda::Real             = 1.0
    "strength of the L1 regularization."
    gamma::Real              = 0.0
    "whether to fit the intercept or not."
    fit_intercept::Bool      = true
    "whether to penalize the intercept."
    penalize_intercept::Bool = false
    "whether to scale the penalty with the number of observations."
    scale_penalty_with_samples::Bool = true
    """any instance of `MLJLinearModels.ProxGrad`.

    If `solver=nothing` (default) then `ProxGrad(accel=true)` (FISTA) is used.

    Solver aliases: `FISTA(; kwargs...) = ProxGrad(accel=true, kwargs...)`,
    `ISTA(; kwargs...) = ProxGrad(accel=false, kwargs...)`. """
    solver::Option{Solver}   = nothing
end

glr(m::ElasticNetRegressor) =
    ElasticNetRegression(m.lambda, m.gamma,
                         fit_intercept=m.fit_intercept,
                         penalize_intercept=m.penalize_intercept,
                         scale_penalty_with_samples=m.scale_penalty_with_samples)


#= ==========================
   ROBUST REGRESSOR (General)
   ========================== =#

"""
$(doc_header(RobustRegressor))

Robust regression is a linear model with objective function

``∑ρ(Xθ - y) + n⋅λ|θ|₂² + n⋅γ|θ|₁``

where ``ρ`` is a robust loss function (e.g. the Huber function) and
``n`` is the number of observations.

If `scale_penalty_with_samples = false` the objective function is instead

``∑ρ(Xθ - y) + λ|θ|₂² + γ|θ|₁``.

$DOC_SOLVERS

# Training data

In MLJ or MLJBase, bind an instance `model` to data with

    mach = machine(model, X, y)

where:

- `X` is any table of input features (eg, a `DataFrame`) whose columns
  have `Continuous` scitype; check column scitypes with `schema(X)`

- `y` is the target, which can be any `AbstractVector` whose element scitype is
  `Continuous`; check the scitype with `scitype(y)`

Train the machine using `fit!(mach, rows=...)`.


# Hyperparameters

$TYPEDFIELDS

$(example_docstring("RobustRegressor"))

See also [`HuberRegressor`](@ref), [`QuantileRegressor`](@ref).

"""
@with_kw_noshow mutable struct RobustRegressor <: MMI.Deterministic
    "the type of robust loss, which can be any instance of
    `MLJLinearModels.L` where `L` is one of: `AndrewsRho`,
    `BisquareRho`, `FairRho`, `HuberRho`, `LogisticRho`,
    `QuantileRho`, `TalwarRho`, `HuberRho`, `TalwarRho`. "
    rho::RobustRho           = HuberRho(0.1)
    "strength of the regularizer if `penalty` is `:l2` or `:l1`.
    Strength of the L2 regularizer if `penalty` is `:en`."
    lambda::Real             = 1.0
    "strength of the L1 regularizer if `penalty` is `:en`."
    gamma::Real              = 0.0
    "the penalty to use, either `:l2`, `:l1`, `:en` (elastic net) or `:none`."
    penalty::SymStr          = :l2
    "whether to fit the intercept or not."
    fit_intercept::Bool      = true
    "whether to penalize the intercept."
    penalize_intercept::Bool = false
    "whether to scale the penalty with the number of observations."
    scale_penalty_with_samples::Bool = true
    """some instance of `MLJLinearModels.S` where `S` is one of: `LBFGS`, `IWLSCG`,
    `Newton`, `NewtonCG`, if `penalty = :l2`, and `ProxGrad` otherwise.

    If `solver = nothing` (default) then `LBFGS()` is used, if `penalty = :l2`, and
    otherwise `ProxGrad(accel=true)` (FISTA) is used.

    Solver aliases: `FISTA(; kwargs...) = ProxGrad(accel=true, kwargs...)`,
    `ISTA(; kwargs...) = ProxGrad(accel=false, kwargs...)`"""
    solver::Option{Solver}   = nothing
end

glr(m::RobustRegressor) =
    RobustRegression(m.rho, m.lambda, m.gamma;
                     penalty=Symbol(m.penalty),
                     fit_intercept=m.fit_intercept,
                     penalize_intercept=m.penalize_intercept,
                     scale_penalty_with_samples=m.scale_penalty_with_samples)


#= ===============
   HUBER REGRESSOR
   =============== =#

"""
$(doc_header(HuberRegressor))

This model coincides with [`RobustRegressor`](@ref), with the exception that the robust
loss, `rho`, is fixed to `HuberRho(delta)`, where `delta` is a new hyperparameter.

$DOC_SOLVERS

# Training data

In MLJ or MLJBase, bind an instance `model` to data with

    mach = machine(model, X, y)

where:

- `X` is any table of input features (eg, a `DataFrame`) whose columns
  have `Continuous` scitype; check column scitypes with `schema(X)`

- `y` is the target, which can be any `AbstractVector` whose element scitype is
  `Continuous`; check the scitype with `scitype(y)`

Train the machine using `fit!(mach, rows=...)`.


# Hyperparameters

$TYPEDFIELDS

$(example_docstring("HuberRegressor"))

See also [`RobustRegressor`](@ref), [`QuantileRegressor`](@ref).

"""
@with_kw_noshow mutable struct HuberRegressor <: MMI.Deterministic
    "parameterizes the `HuberRho` function (radius of the ball within which the loss
    is a quadratic loss)"
    delta::Real              = 0.5
    "strength of the regularizer if `penalty` is `:l2` or `:l1`.
    Strength of the L2 regularizer if `penalty` is `:en`."
    lambda::Real             = 1.0
    "strength of the L1 regularizer if `penalty` is `:en`."
    gamma::Real              = 0.0
    "the penalty to use, either `:l2`, `:l1`, `:en` (elastic net) or `:none`."
    penalty::SymStr          = :l2
    "whether to fit the intercept or not."
    fit_intercept::Bool      = true
    "whether to penalize the intercept."
    penalize_intercept::Bool = false
    "whether to scale the penalty with the number of observations."
    scale_penalty_with_samples::Bool = true
    """some instance of `MLJLinearModels.S` where `S` is one of: `LBFGS`, `IWLSCG`,
    `Newton`, `NewtonCG`, if `penalty = :l2`, and `ProxGrad` otherwise.

    If `solver = nothing` (default) then `LBFGS()` is used, if `penalty = :l2`, and
    otherwise `ProxGrad(accel=true)` (FISTA) is used.

    Solver aliases: `FISTA(; kwargs...) = ProxGrad(accel=true, kwargs...)`,
    `ISTA(; kwargs...) = ProxGrad(accel=false, kwargs...)`"""
    solver::Option{Solver}   = nothing
end

glr(m::HuberRegressor) =
    HuberRegression(m.delta, m.lambda, m.gamma;
                    penalty=Symbol(m.penalty),
                    fit_intercept=m.fit_intercept,
                    penalize_intercept=m.penalize_intercept,
                    scale_penalty_with_samples=m.scale_penalty_with_samples)


#= ==================
   QUANTILE REGRESSOR
   ================== =#

"""
$(doc_header(QuantileRegressor))

This model coincides with [`RobustRegressor`](@ref), with the exception that the robust
loss, `rho`, is fixed to `QuantileRho(delta)`, where `delta` is a new hyperparameter.

$DOC_SOLVERS

# Training data

In MLJ or MLJBase, bind an instance `model` to data with

    mach = machine(model, X, y)

where:

- `X` is any table of input features (eg, a `DataFrame`) whose columns
  have `Continuous` scitype; check column scitypes with `schema(X)`

- `y` is the target, which can be any `AbstractVector` whose element scitype is
  `Continuous`; check the scitype with `scitype(y)`

Train the machine using `fit!(mach, rows=...)`.


# Hyperparameters

$TYPEDFIELDS

$(example_docstring("QuantileRegressor"))

See also [`RobustRegressor`](@ref), [`HuberRegressor`](@ref).

"""
@with_kw_noshow mutable struct QuantileRegressor <: MMI.Deterministic
    "parameterizes the `QuantileRho` function (indicating the quantile to use
    with default `0.5` for the median regression)"
    delta::Real              = 0.5
    "strength of the regularizer if `penalty` is `:l2` or `:l1`.
    Strength of the L2 regularizer if `penalty` is `:en`."
    lambda::Real             = 1.0
    "strength of the L1 regularizer if `penalty` is `:en`."
    gamma::Real              = 0.0
    "the penalty to use, either `:l2`, `:l1`, `:en` (elastic net) or `:none`."
    penalty::SymStr          = :l2
    "whether to fit the intercept or not."
    fit_intercept::Bool      = true
    "whether to penalize the intercept."
    penalize_intercept::Bool = false
    "whether to scale the penalty with the number of observations."
    scale_penalty_with_samples::Bool = true
    """some instance of `MLJLinearModels.S` where `S` is one of: `LBFGS`, `IWLSCG`,
    if `penalty = :l2`, and `ProxGrad` otherwise.

    If `solver = nothing` (default) then `LBFGS()` is used, if `penalty = :l2`, and
    otherwise `ProxGrad(accel=true)` (FISTA) is used.

    Solver aliases: `FISTA(; kwargs...) = ProxGrad(accel=true, kwargs...)`,
    `ISTA(; kwargs...) = ProxGrad(accel=false, kwargs...)`"""
    solver::Option{Solver}   = nothing
end

glr(m::QuantileRegressor) =
    QuantileRegression(m.delta, m.lambda, m.gamma;
                       penalty=Symbol(m.penalty),
                       fit_intercept=m.fit_intercept,
                       penalize_intercept=m.penalize_intercept,
                       scale_penalty_with_samples=m.scale_penalty_with_samples)


#= ==================================
   LEAST ABSOLUTE DEVIATION REGRESSOR
   ================================== =#

"""
$(doc_header(LADRegressor))

Least absolute deviation regression is a linear model with objective function

``∑ρ(Xθ - y) + n⋅λ|θ|₂² + n⋅γ|θ|₁``

where ``ρ`` is the absolute loss and ``n`` is the number of observations.

If `scale_penalty_with_samples = false` the objective function is instead

``∑ρ(Xθ - y) + λ|θ|₂² + γ|θ|₁``.

$DOC_SOLVERS

# Training data

In MLJ or MLJBase, bind an instance `model` to data with

    mach = machine(model, X, y)

where:

- `X` is any table of input features (eg, a `DataFrame`) whose columns
  have `Continuous` scitype; check column scitypes with `schema(X)`

- `y` is the target, which can be any `AbstractVector` whose element scitype is
  `Continuous`; check the scitype with `scitype(y)`

Train the machine using `fit!(mach, rows=...)`.


# Hyperparameters

See also `RobustRegressor`.

## Parameters

$TYPEDFIELDS

$(example_docstring("LADRegressor"))
"""
@with_kw_noshow mutable struct LADRegressor <: MMI.Deterministic
    "strength of the regularizer if `penalty` is `:l2` or `:l1`.
    Strength of the L2 regularizer if `penalty` is `:en`."
    lambda::Real             = 1.0
    "strength of the L1 regularizer if `penalty` is `:en`."
    gamma::Real              = 0.0
    "the penalty to use, either `:l2`, `:l1`, `:en` (elastic net) or `:none`."
    penalty::SymStr          = :l2
    "whether to fit the intercept or not."
    fit_intercept::Bool      = true
    "whether to penalize the intercept."
    penalize_intercept::Bool = false
    "whether to scale the penalty with the number of observations."
    scale_penalty_with_samples::Bool = true
    """some instance of `MLJLinearModels.S` where `S` is one of: `LBFGS`, `IWLSCG`,
    if `penalty = :l2`, and `ProxGrad` otherwise.

    If `solver = nothing` (default) then `LBFGS()` is used, if `penalty = :l2`, and
    otherwise `ProxGrad(accel=true)` (FISTA) is used.

    Solver aliases: `FISTA(; kwargs...) = ProxGrad(accel=true, kwargs...)`,
    `ISTA(; kwargs...) = ProxGrad(accel=false, kwargs...)`"""
    solver::Option{Solver}   = nothing
end

glr(m::LADRegressor) =
    LADRegression(m.lambda, m.gamma;
                  penalty=Symbol(m.penalty),
                  fit_intercept=m.fit_intercept,
                  penalize_intercept=m.penalize_intercept,
                  scale_penalty_with_samples=m.scale_penalty_with_samples)
