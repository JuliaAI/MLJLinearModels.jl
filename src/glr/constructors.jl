export GeneralizedLinearRegression, GLR,
        LinearRegression, RidgeRegression,
        LassoRegression, ElasticNetRegression,
        LADRegression, MADRegression,
        LogisticRegression, MultinomialRegression,
        RobustRegression, HuberRegression, QuantileRegression

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
    loss::L                  = L2Loss()    # L(y, ŷ=Xθ)
    penalty::P               = NoPenalty() # P(θ)
    fit_intercept::Bool      = true        # add intercept ? def=true
    penalize_intercept::Bool = false
end

const GLR = GeneralizedLinearRegression


"""
$SIGNATURES

Objective function: ``|Xθ - y|₂²/2``.
"""
LinearRegression(; fit_intercept::Bool=true) = GLR(fit_intercept=fit_intercept)


"""
$SIGNATURES

Objective function: ``|Xθ - y|₂²/2 + λ|θ|₂²/2``.
"""
function RidgeRegression(λ::Real=1.0; lambda::Real=λ, fit_intercept::Bool=true,
                         penalize_intercept::Bool=false)
    check_pos(lambda)
    GLR(penalty=lambda*L2Penalty(),
        fit_intercept=fit_intercept,
        penalize_intercept=penalize_intercept)
end


"""
$SIGNATURES

Objective function: ``|Xθ - y|₂²/2 + λ|θ|₁``
"""
function LassoRegression(λ::Real=1.0; lambda::Real=λ, fit_intercept::Bool=true,
                         penalize_intercept::Bool=false)
    check_pos(lambda)
    GLR(penalty=lambda*L1Penalty(),
        fit_intercept=fit_intercept,
        penalize_intercept=penalize_intercept)
end


"""
$SIGNATURES

Objective function: ``|Xθ - y|₂²/2 + λ|θ|₂²/2 + γ|θ|₁``
"""
function ElasticNetRegression(λ::Real=1.0, γ::Real=1.0; lambda::Real=λ, gamma::Real=γ,
                              fit_intercept::Bool=true, penalize_intercept::Bool=false)
    check_pos.((lambda, gamma))
    GLR(penalty=lambda*L2Penalty()+gamma*L1Penalty(),
        fit_intercept=fit_intercept,
        penalize_intercept=penalize_intercept)
end


"""
$SIGNATURES

Helper function for objectives which can have 0/L1/L2 penalties.
"""
function _l1l2en(lambda, gamma, penalty, r)
    check_pos.((lambda, gamma))
    penalty ∈ (:l1, :l2, :en, :none) ||
        throw(ArgumentError("Unrecognised penalty for the $r: '$penalty' " *
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
    return penalty
end

"""
$SIGNATURES

Objective function: ``L(y, Xθ) + λ|θ|₂²/2 + γ|θ|₁`` where `L` is either the logistic loss in the
binary case or the multinomial loss otherwise.
"""
function LogisticRegression(λ::Real=1.0, γ::Real=0.0; lambda::Real=λ, gamma::Real=γ,
                            penalty::Symbol=iszero(gamma) ? :l2 : :en,
                            multi_class::Bool=false, fit_intercept::Bool=true,
                            penalize_intercept::Bool=false)
    penalty = _l1l2en(lambda, gamma, penalty, "Logistic regression")
    loss = multi_class ? MultinomialLoss() : LogisticLoss()
    GLR(loss=loss,
        penalty=penalty,
        fit_intercept=fit_intercept,
        penalize_intercept=penalize_intercept)
end

MultinomialRegression(a...; kwa...) = LogisticRegression(a...; multi_class=true, kwa...)


# ========

"""
$SIGNATURES

Objective function: ``∑ρ(Xθ - y) + λ|θ|₂²`` where ρ is a given function on the residuals and
δ a positive tuning parameter for the function in question (e.g. for Huber it corresponds to the
radius of the ball in which residuals are weighed quadratically).
"""
function RobustRegression(ρ::RobustRho=HuberRho(0.1), λ::Real=1.0, γ::Real=0.0;
                          rho::RobustRho=ρ, lambda::Real=λ, gamma::Real=γ,
                          penalty::Symbol=iszero(gamma) ? :l2 : :en, fit_intercept::Bool=true,
                          penalize_intercept::Bool=false)
    penalty = _l1l2en(lambda, gamma, penalty, "Robust regression")
    GLR(loss=RobustLoss(rho),
        penalty=penalty,
        fit_intercept=fit_intercept,
        penalize_intercept=penalize_intercept)
end

"""
$SIGNATURES

Huber Regression with objective:

``∑ρ(Xθ - y) + λ|θ|₂²/2 + γ|θ|``

Where `ρ` is the Huber function `ρ(r) = r²/2``  if `|r|≤δ` and `ρ(r)=δ(|r|-δ/2)` otherwise.
"""
function HuberRegression(δ::Real=0.5, λ::Real=1.0, γ::Real=0.0;
                         delta::Real=δ, lambda::Real=λ, gamma::Real=γ,
                         penalty::Symbol=iszero(gamma) ? :l2 : :en,
                         fit_intercept::Bool=true, penalize_intercept::Bool=false)
    return RobustRegression(HuberRho(delta), lambda, gamma;
                            penalty=penalty, fit_intercept=fit_intercept,
                            penalize_intercept=penalize_intercept)
end

"""
$SIGNATURES

Quantile Regression with objective:

``∑ρ(Xθ - y) + λ|θ|₂²/2 + γ|θ|``

Where `ρ` is the check function `ρ(r) = r(δ - 1(r < 0))`.
"""
function QuantileRegression(δ::Real=0.5, λ::Real=1.0, γ::Real=0.0;
                            delta::Real=δ, lambda::Real=λ, gamma::Real=γ,
                            penalty::Symbol=iszero(gamma) ? :l2 : :en,
                            fit_intercept::Bool=true, penalize_intercept::Bool=false)
    return RobustRegression(QuantileRho(delta), lambda, gamma;
                            penalty=penalty, fit_intercept=fit_intercept,
                            penalize_intercept=penalize_intercept)
end

"""
$SIGNATURES

Least Absolute Deviation regression with objective:

``|Xθ - y|₁ + λ|θ|₂²/2 + γ|θ|``

This is a specific type of Quantile Regression with `δ=0.5` (median).
"""
function LADRegression(λ::Real=1.0, γ::Real=0.0;
                       lambda::Real=λ, gamma::Real=γ,
                       penalty::Symbol=iszero(gamma) ? :l2 : :en,
                       fit_intercept::Bool=true, penalize_intercept::Bool=false)
    return QuantileRegression(0.5, lambda, gamma;
                              penalty=penalty, fit_intercept=fit_intercept,
                              penalize_intercept=penalize_intercept)
end
