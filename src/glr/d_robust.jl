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
            r = apply_X(X, θ) .- y
            w = convert.(Float64, abs.(r) .<= δ)
            g === nothing || begin
                ψr = ψ_(r, w)
                mul!(view(g, 1:p), X', ψr)
                g[end] = sum(ψr)
                g .+= λ .* θ
            end
            H === nothing || begin
                ϕr = ϕ_(r, w)
                ΛX = ϕr .* X
                mul!(view(H, 1:p, 1:p), X', ΛX)
                ΛXt1 = sum(ΛX, dims=1)
                @inbounds for i in 1:p
                    H[i, end] = H[end, i] = ΛXt1[i]
                end
                H[end, end] = sum(ϕr)
                add_λI!(H, λ)
            end
            f === nothing || return glr.loss(r) + glr.penalty(θ)
        end
    else
        (f, g, H, θ) -> begin
            r = apply_X(X, θ) .- y
            w = convert.(Float64, abs.(r) .<= δ)
            g === nothing || begin
                ψr = ψ_(r, w)
                mul!(g, X', ψr)
                g .+= λ .* θ
            end
            H === nothing || (mul!(H, X', ϕ_(r, w) .* X); add_λI!(H, λ))
            f === nothing || return glr.loss(r) + glr.penalty(θ)
        end
    end
end


function Hv!(glr::GLR{RobustLoss{ρ},<:L2R}, X, y) where ρ <: RobustRho1P{δ} where δ
    p  = size(X, 2)
    λ  = getscale(glr.penalty)
    # see d_logistic.jl for more comments on this (similar procedure)
    if glr.fit_intercept
        (Hv, θ, v) -> begin
            r    = apply_X(X, θ) .- y
            w    = convert.(Float64, abs.(r) .<= δ)
            a    = 1:p
            Hvₐ  = view(Hv, a)
            vₐ   = view(v, a)
            XtΛ1 = X' * w
            vₑ   = v[end]
            # update for first p rows
            mul!(Hvₐ, X', w .* (X * vₐ))
            Hvₐ .+= λ .* vₐ .+ XtΛ1 .* vₑ
            # update for the last row
            Hv[end] = dot(XtΛ1, vₐ) + (sum(w)+λ) * vₑ
        end
    else
        (Hv, θ, v) -> begin
            r = apply_X(X, θ) .- y
            w = convert.(Float64, abs.(r) .<= δ)
            mul!(Hv, X', w .* (X * v))
            Hv .+= λ .* v
        end
    end
end

# For IWLS
function Mv!(glr::GLR{RobustLoss{ρ},<:L2R}, X, y) where ρ <: RobustRho1P{δ} where δ
    p  = size(X, 2)
    λ  = getscale(glr.penalty)
    ω_ = ω(ρ)
    # For one θ, we get one system of equation to solve
    # which we solve via an iterative method so, one θ
    # gives one way of applying the relevant matrix (X'ΛX+λI)
    (ωr, θ) -> begin
        r  = apply_X(X, θ) .- y
        w  = convert.(Float64, abs.(r) .<= δ)
        # ω = ψ(r)/r ; weighing factor for IWLS
        ωr .= ω_(r, w)
        # function defining the application of (X'ΛX + λI)
        if glr.fit_intercept
            (Mv, v) -> begin
                a    = 1:p
                vₐ   = view(v, a)
                Mvₐ  = view(Mv, a)
                XtW1 = vec(sum(ωr .* X, dims=1))
                vₑ   = v[end]
                mul!(Mvₐ, X', ωr .* (X * vₐ))
                Mvₐ .+= λ .* vₐ .+ XtW1 .* vₑ
                Mv[end] = dot(XtW1, vₐ) + (sum(ωr)+λ) * vₑ
            end
        else
            (Mv, v) -> (mul!(Mv, X', ωr .* (X * v));  Mv .+= λ .* v)
        end
    end
end
