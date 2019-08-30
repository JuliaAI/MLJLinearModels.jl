# fg! -- objective function and gradient (avoiding recomputations)
# Hv! -- application of the Hessian
# smooth_fg! -- objective associated with the smooth part of the objective

# ----------------------- #
#  -- Ridge Regression -- #
# ----------------------- #
# ->  f(θ)  = |Xθ - y|₂²/2 + λ|θ|₂²
# -> ∇f(θ)  = X'(Xθ - y) + λθ
# -> ∇²f(θ) = X'X + λI
# NOTE:
# * Hv! used in iterative solution
# ---------------------------------------------------------

function Hv!(glr::GLR{L2Loss,<:L2R}, X, y)
    n, p = size(X)
    λ    = getscale(glr.penalty)
    if glr.fit_intercept
        # H = [X 1]'[X 1] + λ I
        # rows a 1:p = [X'X + λI | X'1]
        # row  e end = [1'X      | n+λ]
        (Hv, v) -> begin
            # view on the first p rows
            a   = 1:p
            Hvₐ = view(Hv, a)
            vₐ  = view(v,  a)
            Xt1 = vec(sum(X, dims=1))
            vₑ  = v[end]
            # update for the first p rows -- (X'X + λI)v[1:p] + (X'1)v[end]
            mul!(Hvₐ, X', X * vₐ)
            Hvₐ .+= λ .* vₐ .+ Xt1 .* vₑ
            # update for the last row -- (X'1)'v + n v[end]
            Hv[end] = dot(Xt1, vₐ) + (n+λ) * vₑ
        end
    else
        (Hv, v) -> (mul!(Hv, X', X * v); Hv .+= λ .* v)
    end
end

# ----------------------------- #
#  -- Lasso/Elnet Regression -- #
# ----------------------------- #
# ->  J(θ)  = f(θ) + r(θ)
# ->  f(θ)  = |Xθ - y|₂²/2 + λ|θ|₂²  // smooth
# ->  r(θ)  = γ|θ|₁                  // non-smooth with prox
# -> ∇f(θ)  = X'(Xθ - y) + λθ
# -> ∇²f(θ) = X'X + λI
# -> prox_r = soft-thresh
# ---------------------------------------------------------

function smooth_fg!(glr::GLR{L2Loss,<:ENR}, X, y)
    λ = getscale_l2(glr.penalty)
    p = size(X, 2)
    (g, θ) -> begin
        r = apply_X(X, θ) .- y
        apply_Xt!(g, X, r)
        g .+= λ .* θ
        return glr.loss(r) + get_l2(glr.penalty)(θ)
    end
end
