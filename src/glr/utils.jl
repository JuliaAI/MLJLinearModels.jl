export objective, smooth_objective

"""
$SIGNATURES

Return the objective function (sum of loss + penalty) of a Generalized Linear Model.
"""
objective(glr::GLR) = glr.loss + glr.penalty

"""
$SIGNATURES

Return a function computing the objective at a given point `θ`.
"""
objective(glr::GLR, X, y; c::Int=1) = θ -> objective(glr)(y, apply_X(X, θ, c), θ)


"""
$SIGNATURES

Return a function computing the smooth part of the objective at a given point `θ`.
"""
smooth_objective(glr::GLR, X, y; c::Int=1) = θ -> smooth_objective(glr)(y, apply_X(X, θ, c), θ)

const SMOOTH_LOSS = Union{L2Loss, LogisticLoss, MultinomialLoss}

"""
$SIGNATURES

Return the smooth part of the objective function of a GLR.
"""
smooth_objective(glr::GLR{<:SMOOTH_LOSS,<:ENR}) = glr.loss + get_l2(glr.penalty)
smooth_objective(::GLR) = @error "Case not implemented yet."

"""
$SIGNATURES

Return a model corresponding to the smooth part of the objective.
"""
get_smooth(glr::GLR) = (o = smooth_objective(glr); GLR(o.loss, o.penalty, glr.fit_intercept))
