export GeneralizedLinearRegression, GLR,
        LinearRegression, RidgeRegression,
        LassoRegression, ElasticNetRegression,
        LogisticRegression, MultinomialRegression,
        RobustRegression, HuberRegression

"""
GeneralizedLinearRegression{L<:Loss, P<:Penalty}

Generalized Linear Regression (GLR) model with objective function:

``L(y, Xθ) + P(θ)``

where `L` is a loss function, `P` a penalty, `y` is the vector of observed response, `X` is
the feature matrix and `θ` the vector of parameters.

Special cases include:

* **OLS regression**:      L2 loss, no penalty.
* **Ridge regression**:    L2 loss, L2 penalty.
* **Lasso regression**:    L2 loss, L1 penalty.
* **Logistic regression**: Logit loss, [no,L1,L2] penalty.
"""
@with_kw mutable struct GeneralizedLinearRegression{L<:Loss, P<:Penalty}
    # Parameters that can be tuned
    loss::L             = L2Loss()    # L(y, ŷ=Xθ)
    penalty::P          = NoPenalty() # P(θ)
    fit_intercept::Bool = true        # add intercept ? def=true
end

const GLR = GeneralizedLinearRegression


"""
$SIGNATURES

Objective function: ``|y-Xθ|₂²/2``.
"""
LinearRegression(; fit_intercept::Bool=true) = GLR(fit_intercept=fit_intercept)


"""
$SIGNATURES

Objective function: ``|y-Xθ|₂²/2 + λ|θ|₂²/2``.
"""
function RidgeRegression(λ::Real=1.0; lambda::Real=λ, fit_intercept::Bool=true)
    check_pos(lambda)
    GLR(fit_intercept=fit_intercept, penalty=lambda*L2Penalty())
end


"""
$SIGNATURES

Objective function: ``|y - Xθ|₂²/2 + λ|θ|₁``
"""
function LassoRegression(λ::Real=1.0; lambda::Real=λ, fit_intercept::Bool=true)
    check_pos(lambda)
    GLR(fit_intercept=fit_intercept, penalty=lambda*L1Penalty())
end


"""
$SIGNATURES

Objective function: ``|y - Xθ|₂²/2 + λ|θ|₂²/2 + γ|θ|₁``
"""
function ElasticNetRegression(λ::Real=1.0, γ::Real=1.0; lambda::Real=λ, gamma::Real=γ,
                             fit_intercept::Bool=true)
    check_pos.((lambda, gamma))
    GLR(fit_intercept=fit_intercept, penalty=lambda*L2Penalty()+gamma*L1Penalty())
end


"""
$SIGNATURES

Objective function: ``L(y, Xθ) + λ|θ|₂²/2 + γ|θ|₁`` where `L` is either the logistic loss in the
binary case or the multinomial loss otherwise.
"""
function LogisticRegression(λ::Real=1.0, γ::Real=0.0; lambda::Real=λ, gamma::Real=γ,
                            penalty::Symbol=iszero(gamma) ? :l2 : :en,
                            multi_class::Bool=false,
                            fit_intercept::Bool=true)
    check_pos.((lambda, gamma))
    penalty ∈ (:l1, :l2, :en, :none) ||
        throw(ArgumentError("Unrecognised penalty for a logistic regression: '$penalty' " *
                            "(expected none/l1/l2/en)"))

    penalty = if penalty == :none
       NoPenalty()
    elseif penalty == :l1
        lambda * L1Penalty()
    elseif penalty == :l2
        lambda * L2Penalty()
    else
        lambda * L2Penalty() + gamma * L1Penalty()
    end
    loss = multi_class ? MultinomialLoss() : LogisticLoss()
    GeneralizedLinearRegression(loss=loss, penalty=penalty, fit_intercept=fit_intercept)
end

MultinomialRegression(a...; kwa...) = LogisticRegression(a...; multi_class=true, kwa...)


# ========

"""
$SIGNATURES

Objective function: ``∑ρ(y - Xθ) + λ|θ|₂²`` where ρ is a given function on the residuals and
δ a positive tuning parameter for the function in question (e.g. for Huber it corresponds to the
radius of the ball in which residuals are weighed quadratically).
"""
function RobustRegression(ρ::RobustRho=HuberRho(0.1), λ::Real=1.0; rho::RobustRho=ρ,
                          lambda::Real=λ, fit_intercept::Bool=true)
    check_pos.(lambda)
    GLR(fit_intercept=fit_intercept, loss=RobustLoss(rho), penalty=lambda*L2Penalty())
end

function HuberRegression(δ::Real=0.5, λ::Real=1.0; delta::Real=δ, lambda::Real=λ,
                         fit_intercept::Bool=true)
    return RobustRegression(HuberRho(delta), lambda; fit_intercept=fit_intercept)
end
