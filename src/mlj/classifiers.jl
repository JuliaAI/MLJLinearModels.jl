#= ===================
   LOGISTIC CLASSIFIER
=================== =#

"""
$(doc_header(LogisticClassifier))

This model is more commonly known as "logistic regression". It is a standard classifier
for both binary and multiclass classification.  The objective function applies either a
logistic loss (binary target) or multinomial (softmax) loss, and has a mixed L1/L2
penalty:

``L(y, Xθ) + n⋅λ|θ|₂²/2 + n⋅γ|θ|₁``.

Here ``L`` is either `MLJLinearModels.LogisticLoss` or `MLJLinearModels.MultiClassLoss`,
``λ`` and ``γ`` indicate
the strength of the L2 (resp. L1) regularization components and
``n`` is the number of training observations.

With `scale_penalty_with_samples = false` the objective function is instead

``L(y, Xθ) + λ|θ|₂²/2 + γ|θ|₁``.

# Training data

In MLJ or MLJBase, bind an instance `model` to data with

    mach = machine(model, X, y)

where:

- `X` is any table of input features (eg, a `DataFrame`) whose columns
  have `Continuous` scitype; check column scitypes with `schema(X)`

- `y` is the target, which can be any `AbstractVector` whose element
  scitype is `<:OrderedFactor` or `<:Multiclass`; check the scitype
  with `scitype(y)`

Train the machine using `fit!(mach, rows=...)`.


# Hyperparameters

$TYPEDFIELDS

$(example_docstring("LogisticClassifier", nclasses = 2))

See also [`MultinomialClassifier`](@ref).

"""
@with_kw_noshow mutable struct LogisticClassifier <: MMI.Probabilistic
    "strength of the regularizer if `penalty` is `:l2` or `:l1` and strength of the L2
    regularizer if `penalty` is `:en`."
    lambda::Real             = eps()
    "strength of the L1 regularizer if `penalty` is `:en`."
    gamma::Real              = 0.0
    "the penalty to use, either `:l2`, `:l1`, `:en` (elastic net) or `:none`."
    penalty::SymStr          = :l2
    "whether to fit the intercept or not."
    fit_intercept::Bool      = true
    "whether to penalize the intercept."
    penalize_intercept::Bool = false
    "whether to scale the penalty with the number of samples."
    scale_penalty_with_samples::Bool = true
    """some instance of `MLJLinearModels.S` where `S` is one of: `LBFGS`, `Newton`,
    `NewtonCG`, `ProxGrad`; but subject to the following restrictions:

    - If `penalty = :l2`, `ProxGrad` is disallowed. Otherwise, `ProxGrad` is the only
      option.

    - Unless `scitype(y) <: Finite{2}` (binary target) `Newton` is disallowed.

    If `solver = nothing` (default) then `ProxGrad(accel=true)` (FISTA) is used,
    unless `gamma = 0`, in which case `LBFGS()` is used.

    Solver aliases: `FISTA(; kwargs...) = ProxGrad(accel=true, kwargs...)`,
    `ISTA(; kwargs...) = ProxGrad(accel=false, kwargs...)`"""
    solver::Option{Solver}   = nothing
end

glr(m::LogisticClassifier, nclasses::Integer) =
    LogisticRegression(m.lambda, m.gamma;
                       penalty=Symbol(m.penalty),
                       multi_class=(nclasses > 2),
                       fit_intercept=m.fit_intercept,
                       penalize_intercept=m.penalize_intercept,
                       scale_penalty_with_samples=m.scale_penalty_with_samples,
                       nclasses=nclasses)

#= ======================
   MULTINOMIAL CLASSIFIER
   ====================== =#

"""
$(doc_header(MultinomialClassifier))

This model coincides with [`LogisticClassifier`](@ref), except certain optimizations
possible in the special binary case will not be applied. Its hyperparameters are
identical.

# Training data

In MLJ or MLJBase, bind an instance `model` to data with

    mach = machine(model, X, y)

where:

- `X` is any table of input features (eg, a `DataFrame`) whose columns
  have `Continuous` scitype; check column scitypes with `schema(X)`

- `y` is the target, which can be any `AbstractVector` whose element
  scitype is `<:OrderedFactor` or `<:Multiclass`; check the scitype
  with `scitype(y)`

Train the machine using `fit!(mach, rows=...)`.


# Hyperparameters

$TYPEDFIELDS

$(example_docstring("MultinomialClassifier", nclasses = 3))

See also [`LogisticClassifier`](@ref).

"""
@with_kw_noshow mutable struct MultinomialClassifier <: MMI.Probabilistic
    "strength of the regularizer if `penalty` is `:l2` or `:l1`.
    Strength of the L2 regularizer if `penalty` is `:en`."
    lambda::Real             = eps()
    "strength of the L1 regularizer if `penalty` is `:en`."
    gamma::Real              = 0.0
    "the penalty to use, either `:l2`, `:l1`, `:en` (elastic net) or `:none`."
    penalty::SymStr          = :l2
    "whether to fit the intercept or not."
    fit_intercept::Bool      = true
    "whether to penalize the intercept."
    penalize_intercept::Bool = false
    "whether to scale the penalty with the number of samples."
    scale_penalty_with_samples::Bool = true
    """some instance of `MLJLinearModels.S` where `S` is one of: `LBFGS`,
    `NewtonCG`, `ProxGrad`; but subject to the following restrictions:

    - If `penalty = :l2`, `ProxGrad` is disallowed. Otherwise, `ProxGrad` is the only
      option.

    - Unless `scitype(y) <: Finite{2}` (binary target) `Newton` is disallowed.

    If `solver = nothing` (default) then `ProxGrad(accel=true)` (FISTA) is used,
    unless `gamma = 0`, in which case `LBFGS()` is used.

    Solver aliases: `FISTA(; kwargs...) = ProxGrad(accel=true, kwargs...)`,
    `ISTA(; kwargs...) = ProxGrad(accel=false, kwargs...)`"""
    solver::Option{Solver}   = nothing
end

glr(m::MultinomialClassifier, nclasses::Integer) =
    MultinomialRegression(m.lambda, m.gamma;
                          penalty=Symbol(m.penalty),
                          fit_intercept=m.fit_intercept,
                          penalize_intercept=m.penalize_intercept,
                          scale_penalty_with_samples=m.scale_penalty_with_samples,
                          nclasses=nclasses)
