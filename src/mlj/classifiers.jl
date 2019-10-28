#=  ===================
    LOGISTIC CLASSIFIER
    =================== =#

@with_kw_noshow mutable struct LogisticClassifier <: MLJBase.Probabilistic
    lambda::Real             = 1.0
    gamma::Real              = 0.0
    penalty::Symbol          = :l2
    fit_intercept::Bool      = true
    penalize_intercept::Bool = false
    solver::Option{Solver}   = nothing
    multi_class::Bool        = false
end

glr(m::LogisticClassifier) = LogisticRegression(m.lambda, m.gamma; penalty=m.penalty,
                                                multi_class=m.multi_class,
                                                fit_intercept=m.fit_intercept,
                                                penalize_intercept=m.penalize_intercept)

descr(::Type{LogisticClassifier}) = "Classifier corresponding to the loss function ``L(y, Xθ) + λ|θ|₂²/2 + γ|θ|₁`` where `L` is the logistic loss."

#=  ======================
    MULTINOMIAL CLASSIFIER
    ====================== =#

@with_kw_noshow mutable struct MultinomialClassifier <: MLJBase.Probabilistic
    lambda::Real             = 1.0
    gamma::Real              = 0.0
    penalty::Symbol          = :l2
    fit_intercept::Bool      = true
    penalize_intercept::Bool = false
    solver::Option{Solver}   = nothing
end

glr(m::MultinomialClassifier) = MultinomialRegression(m.lambda, m.gamma; penalty=m.penalty,
                                                      fit_intercept=m.fit_intercept,
                                                      penalize_intercept=m.penalize_intercept)

descr(::Type{MultinomialClassifier}) = "Classifier corresponding to the loss function ``L(y, Xθ) + λ|θ|₂²/2 + γ|θ|₁`` where `L` is the multinomial loss."
