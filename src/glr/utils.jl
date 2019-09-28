export objective, smooth_objective

# NOTE: RobustLoss are not always everywhere  smooth but "smooth-enough".
const SMOOTH_LOSS = Union{L2Loss, LogisticLoss, MultinomialLoss, RobustLoss}

"""
$SIGNATURES

Return the objective function (sum of loss + penalty) of a Generalized Linear Model.
"""
objective(glr::GLR) = glr.loss + glr.penalty


"""
$SIGNATURES

Return a function computing the objective at a given point `θ`.
Note that the [`apply_X`](@ref) takes care of a potential intercept.
"""
objective(glr::GLR, X, y; c::Int=1) =
    θ -> objective(glr)(y, apply_X(X, θ, c), view_θ(glr, θ))


"""
$SIGNATURES

Return a function computing the smooth part of the objective at a given point `θ`.
"""
smooth_objective(glr::GLR, X, y; c::Int=1) =
    θ -> smooth_objective(glr)(y, apply_X(X, θ, c), view_θ(glr, θ))

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
get_smooth(glr::GLR) = (
    o = smooth_objective(glr);
    GLR(o.loss, o.penalty, glr.fit_intercept, glr.penalize_intercept))


"""
$SIGNATURES

Helper function to compute the residuals.
"""
function get_residuals!(r, X, θ, y)
    apply_X!(r, X, θ)
    r .-= y
end
