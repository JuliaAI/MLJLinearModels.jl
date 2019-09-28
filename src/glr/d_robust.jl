function _get_residuals(X, θ, y)
    Xθ  = SCRATCH_N[]
    apply_X!(Xθ, X, θ)
    r   = Xθ
    r .-= y
    return r
end

function _get_w(r, δ)
    w  = SCRATCH_N2[]
    w .= convert.(Float64, abs.(r) .<= δ)
    return w
end

function _get_ψr(r, w, ψ)
    ψr  = SCRATCH_N3[]
    ψr .= ψ.(r, w)
    return ψr
end

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

function fgh!(glr::GLR{RobustLoss{ρ},<:L2R}, X, y) where ρ <: RobustRho1P{δ} where δ
    p  = size(X, 2)
    λ  = getscale(glr.penalty)
    ψ_ = ψ(ρ)
    ϕ_ = ϕ(ρ)
    if glr.fit_intercept
        (f, g, H, θ) -> begin
            r  = SCRATCH_N[]
            get_residuals!(r, X, θ, y)
            w  = SCRATCH_N2[]
            w .= convert.(Float64, abs.(r) .<= δ)
            # gradient via ψ function
            g === nothing || begin
                ψr  = SCRATCH_N3[]
                ψr .= ψ_.(r, w)
                apply_Xt!(g, X, ψr)
                g .+= λ .* θ
                glr.penalize_intercept || (g[end] -= λ * θ[end])
            end
            # Hessian via ϕ functiono
            H === nothing || begin
                # NOTE: Hessian allocates a ton anyway so use of scratch is a bit pointless
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
            r = SCRATCH_N[]
            get_residuals!(r, X, θ, y)
            w = SCRATCH_N2[]
            w .= convert.(Float64, abs.(r) .<= δ)
            # gradient via ψ function
            g === nothing || begin
                ψr  = SCRATCH_N3[]
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


function Hv!(glr::GLR{RobustLoss{ρ},<:L2R}, X, y) where ρ <: RobustRho1P{δ} where δ
    p  = size(X, 2)
    λ  = getscale(glr.penalty)
    ϕ_ = ϕ(ρ)
    # see d_logistic.jl for more comments on this (similar procedure)
    if glr.fit_intercept
        (Hv, θ, v) -> begin
            r  = SCRATCH_N[]
            get_residuals!(r, X, θ, y)
            w  = SCRATCH_N2[]
            w .= convert.(Float64, abs.(r) .<= δ)
            w .= ϕ_.(r, w)
            # views on first p rows (intercept row treated after)
            a    = 1:p
            Hvₐ  = view(Hv, a)
            vₐ   = view(v, a)
            XtΛ1 = view(SCRATCH_P[], a)       # we can recycle as we don't need r anymore
            apply_Xt!(XtΛ1, X, w)
            vₑ   = v[end]
            # update for first p rows
            t    = SCRATCH_N3[]
            apply_X!(t, X, vₐ)
            t  .*= w
            apply_Xt!(Hvₐ, X, t)
            Hvₐ .+= λ .* vₐ .+ XtΛ1 .* vₑ
            # update for the last row (intercept)
            Hv[end] = dot(XtΛ1, vₐ) + (sum(w) + λ_if_penalize_intercept(glr, λ)) * vₑ
        end
    else
        (Hv, θ, v) -> begin
            r  = SCRATCH_N[]
            get_residuals!(r, X, θ, y)
            w  = SCRATCH_N2[]
            w .= convert.(Float64, abs.(r) .<= δ)
            w .= ϕ_.(r, w)
            t  = SCRATCH_N3[]
            apply_X!(t, X, v)
            t .*= w
            apply_Xt!(Hv, X, t)
            Hv .+= λ .* v
        end
    end
end


# For IWLS
function Mv!(glr::GLR{RobustLoss{ρ},<:L2R}, X, y;
             threshold=1e-6) where ρ <: RobustRho1P{δ} where δ
    p  = size(X, 2)
    λ  = getscale(glr.penalty)
    ω_ = ω(ρ, threshold)
    # For one θ, we get one system of equation to solve
    # which we solve via an iterative method so, one θ
    # gives one way of applying the relevant matrix (X'ΛX+λI)
    (ωr, θ) -> begin
        r   = SCRATCH_N[]
        get_residuals!(r, X, θ, y)
        w   = SCRATCH_N2[]
        w  .= convert.(Float64, abs.(r) .<= δ)
        # ω = ψ(r)/r ; weighing factor for IWLS
        ωr .= ω_.(r, w)
        # function defining the application of (X'ΛX + λI)
        if glr.fit_intercept
            (Mv, v) -> begin
                a     = 1:p
                vₐ    = view(v, a)
                Mvₐ   = view(Mv, a)
                XtW1  = view(SCRATCH_P[], a)
                @inbounds for j in a
                    XtW1[j] = dot(ωr, view(X, :, j))
                end
                vₑ = v[end]
                t  = SCRATCH_N[]
                apply_X!(t, X, vₐ)
                t .*= ωr
                mul!(Mvₐ, X', t)
                Mvₐ .+= λ .* vₐ .+ XtW1 .* vₑ
                Mv[end] = dot(XtW1, vₐ) + (sum(ωr) + λ_if_penalize_intercept(glr, λ)) * vₑ
            end
        else
            (Mv, v) -> begin
                t  = SCRATCH_N[]
                apply_X!(t, X, v)
                t .*= ωr
                mul!(Mv, X', t)
                Mv .+= λ .* v
            end
        end
    end
end


# this is a bit of an abuse in that in some cases the ρ is not everywhere differentiable
function smooth_fg!(glr::GLR{RobustLoss{ρ},<:ENR}, X, y) where ρ <: RobustRho1P{δ} where δ
    λ  = getscale_l2(glr.penalty)
    p  = size(X, 2)
    ψ_ = ψ(ρ)
    (g, θ) -> begin
        r   = SCRATCH_N[]
        get_residuals!(r, X, θ, y)
        w   = SCRATCH_N2[]
        w  .= convert.(Float64, abs.(r) .<= δ)
        ψr  = SCRATCH_N3[]
        ψr .= ψ_.(r, w)
        apply_Xt!(g, X, ψr)
        g .+= λ .* θ
        glr.fit_intercept && (glr.penalize_intercept || (g[end] -= λ * θ[end]))
        return glr.loss(r) + get_l2(glr.penalty)(view_θ(glr, θ))
    end
end
