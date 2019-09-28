# Proximal operators
#   prox_{r}(z) = argmin_x {r(x) + |x-z|₂²/2}
#
# NOTE: usually `r` is of the form `γr̄` and, further, algorithms
# require prox_{αr} so there will typically be a product `α * γ`.
#
# (p, z, α) corresponding to p .= prox_{αλ}(z)
#
# ---------------------------------------------------------------

# ------------- #
# -- L1-Norm -- #
# ------------- #
# r(θ) = λ|θ|₁
# prox_{αr}(z) = sign(z)(abs(z) - αλ)₊
# ------------------------------------

function prox!(glr::GLR{<:Loss,<:Union{L1R,CompositePenalty}})
    γ = getscale_l1(glr.penalty)
    (p, α, z) -> begin
        p .= soft_thresh.(z, α * γ)
        glr.fit_intercept && (glr.penalize_intercept || (p[end] = z[end]))
    end
end
