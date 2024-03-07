# ------------------------ #
#  -- Robust Regression -- #
# ------------------------ #
# ->     r  = Xθ - y
# ->   ρ(r) is the robust penalty assoc with residual
# ->   ψ(r) = ρ'(r) (first deriv)
# ->   ϕ(r) = ψ'(r) (second deriv; may be discont)
# ->   Λ(r) = diag(ϕ(r))
# ->   f(θ) = ∑ρ.(r) + λ|θ|₂²
# ->  ∇f(θ) = X'ψ.(r) + λθ
# -> ∇²f(θ) = X'Λ(r)X + λI
# ---------------------------------------------------------

function fgh!(::Type{T}, glr::GLR{RobustLoss{ρ},<:L2R},
    X, y, scratch) where {T<:Real, ρ<:RobustRho1P{δ}} where δ
    n, p = size(X)
    λ    = get_penalty_scale(glr, n)
    ψ_   = ψ(ρ)
    ϕ_   = ϕ(ρ)
    if glr.fit_intercept
        (f, g, H, θ) -> begin
            r  = scratch.n
            get_residuals!(r, X, θ, y)
            w  = scratch.n2
            w .= convert.(Float64, abs.(r) .<= δ)
            # gradient via ψ function
            g === nothing || begin
                ψr  = scratch.n3
                ψr .= ψ_.(r, w)
                apply_Xt!(g, X, ψr)
                g .+= λ .* θ
                glr.penalize_intercept || (g[end] -= λ * θ[end])
            end
            # Hessian via ϕ function
            H === nothing || begin
                # NOTE: Hessian allocates a ton anyway so use of scratch is a
                # bit pointless
                ϕr = ϕ_.(r, w)
                ΛX = ϕr .* X
                mul!(view(H, 1:p, 1:p), X', ΛX)
                ΛXt1 = sum(ΛX, dims=1)
                @inbounds for i in 1:p
                    H[i, end] = H[end, i] = ΛXt1[i]
                end
                H[end, end] = sum(ϕr)
                add_λI!(H, λ, glr.penalize_intercept)
            end
            # function value
            f === nothing || return glr.loss(r) + glr.penalty(view_θ(glr, θ))
        end
    else
        (f, g, H, θ) -> begin
            r = scratch.n
            get_residuals!(r, X, θ, y)
            w = scratch.n2
            w .= convert.(T, abs.(r) .<= δ)
            # gradient via ψ function
            g === nothing || begin
                ψr  = scratch.n3
                ψr .= ψ_.(r, w)
                apply_Xt!(g, X, ψr)
                g .+= λ .* θ
            end
            # Hessian via ϕ function
            H === nothing || (mul!(H, X', ϕ_.(r, w) .* X); add_λI!(H, λ))
            f === nothing || return glr.loss(r) + glr.penalty(θ)
        end
    end
end

function fgh!(glr::GLR{RobustLoss{ρ},<:L2R},
    X, y, scratch) where {ρ<:RobustRho1P{δ}} where δ
    return fgh!(eltype(X), glr, X, y, scratch)
end

function Hv!(::Type{T}, glr::GLR{RobustLoss{ρ},<:L2R},
    X, y, scratch) where {T<:Real, ρ<:RobustRho1P{δ}} where δ
    n, p = size(X)
    λ    = get_penalty_scale(glr, n)
    ϕ_   = ϕ(ρ)
    # see d_logistic.jl for more comments on this (similar procedure)
    if glr.fit_intercept
        (Hv, θ, v) -> begin
            r  = scratch.n
            get_residuals!(r, X, θ, y)
            w  = scratch.n2
            w .= convert.(T, abs.(r) .<= δ)
            w .= ϕ_.(r, w)
            # views on first p rows (intercept row treated after)
            a    = 1:p
            Hvₐ  = view(Hv, a)
            vₐ   = view(v, a)
            XtΛ1 = view(scratch.p, a)     # we can recycle
            apply_Xt!(XtΛ1, X, w)
            vₑ   = v[end]
            # update for first p rows
            t    = scratch.n3
            apply_X!(t, X, vₐ)
            t  .*= w
            apply_Xt!(Hvₐ, X, t)
            Hvₐ .+= λ .* vₐ .+ XtΛ1 .* vₑ
            # update for the last row (intercept)
            Hv[end] = dot(XtΛ1, vₐ) + (sum(w) + λ_if_penalize_intercept(glr, λ)) * vₑ
        end
    else
        (Hv, θ, v) -> begin
            r  = scratch.n
            get_residuals!(r, X, θ, y)
            w  = scratch.n2
            w .= convert.(T, abs.(r) .<= δ)
            w .= ϕ_.(r, w)
            t  = scratch.n3
            apply_X!(t, X, v)
            t .*= w
            apply_Xt!(Hv, X, t)
            Hv .+= λ .* v
        end
    end
end

function Hv!(glr::GLR{RobustLoss{ρ},<:L2R},
    X, y, scratch) where {ρ<:RobustRho1P{δ}} where δ
    return Hv!(eltype(X), glr, X, y, scratch)
end

# For IWLS
function Mv!(::Type{T}, glr::GLR{RobustLoss{ρ},<:L2R}, X, y, scratch;
    threshold=T(1e-6)) where {T<:Real, ρ<:RobustRho1P{δ}} where δ
    n, p = size(X)
    λ    = get_penalty_scale(glr, n)
    ω_   = ω(ρ, threshold)
    # For one θ, we get one system of equation to solve
    # which we solve via an iterative method so, one θ
    # gives one way of applying the relevant matrix (X'ΛX+λI)
    (ωr, θ) -> begin
        r   = scratch.n
        get_residuals!(r, X, θ, y)
        w   = scratch.n2
        w  .= convert.(T, abs.(r) .<= δ)
        # ω = ψ(r)/r ; weighing factor for IWLS
        ωr .= ω_.(r, w)
        # function defining the application of (X'ΛX + λI)
        if glr.fit_intercept
            (Mv, v) -> begin
                a    = 1:p
                vₐ   = view(v, a)
                Mvₐ  = view(Mv, a)
                XtW1 = view(scratch.p, a)
                @inbounds for j in a
                    XtW1[j] = dot(ωr, view(X, :, j))
                end
                vₑ = v[end]
                t  = scratch.n
                apply_X!(t, X, vₐ)
                t .*= ωr
                mul!(Mvₐ, X', t)
                Mvₐ .+= λ .* vₐ .+ XtW1 .* vₑ
                Mv[end] = dot(XtW1, vₐ) +
                            (sum(ωr) + λ_if_penalize_intercept(glr, λ)) * vₑ
            end
        else
            (Mv, v) -> begin
                t  = scratch.n
                apply_X!(t, X, v)
                t .*= ωr
                mul!(Mv, X', t)
                Mv .+= λ .* v
            end
        end
    end
end

function Mv!(glr::GLR{RobustLoss{ρ},<:L2R}, X, y, scratch;
    threshold=1e-6) where {ρ<:RobustRho1P{δ}} where δ
    T = eltype(X)
    return Mv!(T, glr, X, y, scratch; threshold=T(threshold))
end


# this is a bit of an abuse in that in some cases the ρ is not everywhere
# differentiable
function smooth_fg!(::Type{T}, glr::GLR{RobustLoss{ρ},<:ENR},
    X, y, scratch) where {T<:Real, ρ<:RobustRho1P{δ}} where δ
    n, p = size(X)
    λ    = get_penalty_scale_l2(glr, n)
    ψ_   = ψ(ρ)
    (g, θ) -> begin
        r   = scratch.n
        get_residuals!(r, X, θ, y)
        w   = scratch.n2
        w  .= convert.(T, abs.(r) .<= δ)
        ψr  = scratch.n3
        ψr .= ψ_.(r, w)
        apply_Xt!(g, X, ψr)
        g .+= λ .* θ
        glr.fit_intercept && (glr.penalize_intercept || (g[end] -= λ * θ[end]))
        return glr.loss(r) + get_l2(glr.penalty)(view_θ(glr, θ))
    end
end

function smooth_fg!(glr::GLR{RobustLoss{ρ},<:ENR},
    X, y, scratch) where {ρ<:RobustRho1P{δ}} where δ
    return smooth_fg!(eltype(X), glr, X, y, scratch)
end
