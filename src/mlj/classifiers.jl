#= ===================
   LOGISTIC CLASSIFIER
   =================== =#

"""
Logistic Classifier (typically called "Logistic Regression"). This model is
a standard classifier for both binary and multiclass classification.
In the binary case it corresponds to the LogisticLoss, in the multiclass to the
Multinomial (softmax) loss. An elastic net penalty can be applied with
overall objective function

``L(y, Xθ) + n⋅λ|θ|₂²/2 + n⋅γ|θ|₁``

where ``L`` is either the logistic or multinomial loss and ``λ`` and ``γ`` indicate
the strength of the L2 (resp. L1) regularisation components and
``n`` is the number of samples `size(X, 1)`.
With `scale_penalty_with_samples = false` the objective function is
``L(y, Xθ) + λ|θ|₂²/2 + γ|θ|₁``

## Parameters

$TYPEDFIELDS

$(example_docstring("LogisticClassifier", nclasses = 2))
"""
@with_kw_noshow mutable struct LogisticClassifier <: MMI.Probabilistic
    "strength of the regulariser if `penalty` is `:l2` or `:l1` and strength of the L2
    regulariser if `penalty` is `:en`."
    lambda::Real             = eps()
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

glr(m::LogisticClassifier, nclasses::Integer) =
    LogisticRegression(m.lambda, m.gamma;
                       penalty=Symbol(m.penalty),
                       multi_class=(nclasses > 2),
                       fit_intercept=m.fit_intercept,
                       penalize_intercept=m.penalize_intercept,
                       scale_penalty_with_samples=m.scale_penalty_with_samples,
                       nclasses=nclasses)

descr(::Type{LogisticClassifier}) = "Classifier corresponding to the loss function ``L(y, Xθ) + λ|θ|₂²/2 + γ|θ|₁`` where `L` is the logistic loss."

#= ======================
   MULTINOMIAL CLASSIFIER
   ====================== =#

"""
See `LogisticClassifier`, it's the same except that multiple classes are assumed
by default. The other parameters are the same.

## Parameters

$TYPEDFIELDS

$(example_docstring("LogisticClassifier", nclasses = 3))
"""
@with_kw_noshow mutable struct MultinomialClassifier <: MMI.Probabilistic
    "strength of the regulariser if `penalty` is `:l2` or `:l1`.
    Strength of the L2 regulariser if `penalty` is `:en`."
    lambda::Real             = eps()
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

glr(m::MultinomialClassifier, nclasses::Integer) =
    MultinomialRegression(m.lambda, m.gamma;
                          penalty=Symbol(m.penalty),
                          fit_intercept=m.fit_intercept,
                          penalize_intercept=m.penalize_intercept,
                          scale_penalty_with_samples=m.scale_penalty_with_samples,
                          nclasses=nclasses)

descr(::Type{MultinomialClassifier}) =
    "Classifier corresponding to the loss function " *
    "``L(y, Xθ) + λ|θ|₂²/2 + γ|θ|₁`` where `L` is the multinomial loss."
