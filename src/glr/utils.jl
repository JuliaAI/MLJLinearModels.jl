export objective, smooth_objective

# NOTE: RobustLoss are not always everywhere  smooth but "smooth-enough".
const SmoothLoss = Union{L2Loss, LogisticLoss, MultinomialLoss, RobustLoss}

"""
$SIGNATURES

Return the objective function (sum of loss + penalty) of a Generalized Linear Model.
"""
objective(glr::GLR, n) = glr.loss + glr.penalty * ifelse(glr.scale_penalty_with_samples, n, 1.)


"""
$SIGNATURES

Return a function computing the objective at a given point `θ`.
Note that the [`apply_X`](@ref) takes care of a potential intercept.
"""
objective(glr::GLR, X, y; c::Int=0) =
    θ -> objective(glr, size(X, 1))(y, apply_X(X, θ, c), view_θ(glr, θ))


"""
$SIGNATURES

Return a function computing the smooth part of the objective at a given
evaluation point `θ`.
"""
smooth_objective(glr::GLR, X, y; c::Int=0) =
    θ -> smooth_objective(glr, size(X, 1))(y, apply_X(X, θ, c), view_θ(glr, θ))

smooth_objective(glr::GLR, XX::T, Xy::U; c::Int=0) where {T <: LinearMap, U <: LinearMap} =
    θ -> θ'*XX*θ - only(2*(θ'*Xy))

"""
$SIGNATURES

Return the smooth part of the objective function of a GLR.
"""
smooth_objective(glr::GLR{<:SmoothLoss,<:ENR}, n) = glr.loss + get_l2(glr.penalty) * ifelse(glr.scale_penalty_with_samples, n, 1.)

smooth_objective(::GLR) = @error "Case not implemented yet."

"""
$SIGNATURES

Return a model corresponding to the smooth part of the objective.
"""
get_smooth(glr::GLR) =
    GLR(glr.loss, get_l2(glr.penalty),
        glr.fit_intercept, glr.penalize_intercept, glr.scale_penalty_with_samples)


"""
$SIGNATURES

Helper function to compute the residuals.
"""
function get_residuals!(r, X, θ, y)
    apply_X!(r, X, θ)
    r .-= y
end
