# fg! -- objective function and gradient (avoiding recomputations)
# Hv! -- application of the Hessian
# smooth_fg! -- objective associated with the smooth part of the objective

# ----------------------- #
#  -- Ridge Regression -- #
# ----------------------- #
# ->  f(θ)  = |Xθ - y|₂²/2 + λ|θ|₂²/2
# -> ∇f(θ)  = X'(Xθ - y) + λθ
# -> ∇²f(θ) = X'X + λI
# NOTE:
# * Hv! used in iterative solution
# ---------------------------------------------------------

function Hv!(glr::GLR{L2Loss,<:L2R}, X, y, scratch)
    n, p = size(X)
    λ    = get_penalty_scale(glr, n)
    if glr.fit_intercept
        # H = [X 1]'[X 1] + λ I
        # rows a 1:p = [X'X + λI | X'1]
        # row  e end = [1'X      | n+λι] where ι is 1 if glr.penalize_intercept
        ι = float(glr.penalize_intercept)
        (Hv, v) -> begin
            # view on the first p rows
            a     = 1:p
            Hvₐ   = view(Hv, a)
            vₐ    = view(v,  a)
            Xt1   = view(scratch.p, a)
            Xt1 .*= 0
            @inbounds for i in a, j in 1:n
                Xt1[i] += X[j, i]           # -- X'1
            end
            vₑ  = v[end]
            # update for the first p rows   -- (X'X + λI)v[1:p] + (X'1)v[end]
            Xvₐ = scratch.n
            mul!(Xvₐ, X, vₐ)
            mul!(Hvₐ, X', Xvₐ)
            Hvₐ .+= λ .* vₐ .+ Xt1 .* vₑ
            # update for the last row       -- (X'1)'v + n v[end]
            Hv[end] = dot(Xt1, vₐ) + (n + λ_if_penalize_intercept(glr, λ)) * vₑ
        end
    else
        (Hv, v) -> begin
            Xv = scratch.n
            mul!(Xv, X, v)       # -- Xv
            mul!(Hv, X', Xv)     # -- X'Xv
            Hv .+= λ .* v        # -- X'Xv + λv
        end
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

function smooth_fg!(glr::GLR{L2Loss,<:ENR}, X, y, scratch)
    λ = get_penalty_scale_l2(glr, length(y))
    (g, θ) -> begin
        # cache contains the residuals (Xθ-y)
        r = scratch.n
        get_residuals!(r, X, θ, y) # -- r = Xθ-y
        apply_Xt!(g, X, r)
        g .+= λ .* θ
        glr.fit_intercept && (glr.penalize_intercept || (g[end] -= λ * θ[end]))
        return glr.loss(r) + get_l2(glr.penalty)(view_θ(glr, θ))
    end
end
