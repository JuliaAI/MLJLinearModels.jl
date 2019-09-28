# ------------------------------- #
#  -- Logistic Regression (L2) -- #
# ------------------------------- #
# ->  f(θ)  = -∑logσ(yXθ) + λ|θ|₂²
# -> ∇f(θ)  = -X'(yσ(-yXθ)) + λθ
# -> ∇²f(θ) = X'(σ(yXθ))X + λI
# NOTE:
# * yᵢ ∈ {±1} so that y² = 1
# * -σ(-x) ==(σ(x)-1)
# ---------------------------------------------------------

function fgh!(glr::GLR{LogisticLoss,<:L2R}, X, y)
    J = objective(glr) # GLR objective (loss+penalty)
    p = size(X, 2)
    λ = getscale(glr.penalty)
    if glr.fit_intercept
        (f, g, H, θ) -> begin
            Xθ = SCRATCH_N[]
            apply_X!(Xθ, X, θ)                       # -- Xθ = apply_X(X, θ)
            # precompute σ(yXθ) use -σ(-x) = (σ(x)-1)
            w  = SCRATCH_N2[]
            w .= σ.(Xθ .* y)                         # -- w  = σ.(Xθ .* y)
            g === nothing || begin
                t  = SCRATCH_N3[]
                t .= y .* (w .- 1.0)                 # -- t = y .* (w .- 1.0)
                apply_Xt!(g, X, t)                   # -- g = X't
                g .+= λ .* θ
                glr.penalize_intercept || (g[end] -= λ * θ[end])
            end
            H === nothing || begin
                # NOTE: we could try to be clever to reduce the allocations for
                # ΛX but computing the full hessian allocates a lot anyway so
                # probably not really worth it
                ΛX = w .* X                           # !! big allocs
                mul!(view(H, 1:p, 1:p), X', ΛX)       # -- H[1:p,1:p] = X'ΛX
                ΛXt1 = view(SCRATCH_P[], 1:p)
                copyto!(ΛXt1, sum(ΛX, dims=1))        # -- (ΛX)'1
                @inbounds for i = 1:p
                    H[i, end] = H[end, i] = ΛXt1[i]   # -- H[:,p+1] = H[p+1,:] = (ΛX)'1
                end
                H[end, end] = sum(w)                  # -- 1'Λ1'
                add_λI!(H, λ, glr.penalize_intercept) # -- H = X'ΛX + λI
            end
            f === nothing || return J(y, Xθ, view_θ(glr, θ))
        end
    else
        # see comments above, same computations just no additional things for
        # fit_intercept
        (f, g, H, θ) -> begin
            Xθ = SCRATCH_N[]
            apply_X!(Xθ, X, θ)
            w  = SCRATCH_N2[]
            w .= σ.(y .* Xθ)
            g === nothing || begin
                t  = SCRATCH_N3[]
                t .= y .* (w .- 1.0)
                apply_Xt!(g, X, t)
                g .+= λ .* θ
            end
            H === nothing || begin
                mul!(H, X', w .* X)
                add_λI!(H, λ)
            end
            f === nothing || return J(y, Xθ, θ)
        end
    end
end

function Hv!(glr::GLR{LogisticLoss,<:L2R}, X, y)
    p = size(X, 2)
    λ = getscale(glr.penalty)
    if glr.fit_intercept
        # H = [X 1]'Λ[X 1] + λ I
        # rows a 1:p = [X'ΛX + λI | X'Λ1]
        # row  e end = [1'ΛX      | sum(a)+λ]
        (Hv, θ, v) -> begin
            Xθ = SCRATCH_N[]
            apply_X!(Xθ, X, θ)                       # -- Xθ = apply_X(X, θ)
            w  = SCRATCH_N2[]
            w .= σ.(Xθ .* y)                         # -- w  = σ.(Xθ .* y)
            # view on the first p rows
            a    = 1:p
            Hvₐ  = view(Hv, a)
            vₐ   = view(v,  a)
            XtΛ1 = view(SCRATCH_P[], 1:p)
            mul!(XtΛ1, X', w)                        # -- X'Λ1; O(np)
            vₑ   = v[end]
            # update for the first p rows -- (X'X + λI)v[1:p] + (X'1)v[end]
            Xvₐ  = SCRATCH_N[]
            mul!(Xvₐ, X, vₐ)
            Xvₐ .*=  w                               # --  ΛXvₐ
            mul!(Hvₐ, X', Xvₐ)                       # -- (X'ΛX)vₐ
            Hvₐ .+= λ .* vₐ .+ XtΛ1 .* vₑ
            # update for the last row -- (X'1)'v + n v[end]
            Hv[end] = dot(XtΛ1, vₐ) + (sum(w) + λ_if_penalize_intercept(glr, λ)) * vₑ
        end
    else
        (Hv, θ, v) -> begin
            Xθ = SCRATCH_N[]
            apply_X!(Xθ, X, θ)
            w  = SCRATCH_N2[]
            w .= σ.(Xθ .* y)                # -- σ(yXθ)
            Xv = SCRATCH_N3[]
            mul!(Xv, X, v)
            Xv .*= SCRATCH_N2[]                # -- ΛXv
            mul!(Hv, X', Xv)                # -- X'ΛXv
            Hv .+= λ .* v
        end
    end
end

# ----------------------------------- #
#  -- L1/Elnet Logistic Regression -- #
# ----------------------------------- #
# ->  J(θ)  = f(θ) + r(θ)
# ->  f(θ)  = LL + λ|θ|₂²  // smooth (LL = LogisticLoss)
# ->  r(θ)  = γ|θ|₁        // non-smooth with prox
# -> ∇f(θ)  = ∇LL + λθ
# -> ∇²f(θ) = ∇²LL + λI
# -> prox_r = soft-thresh
# ---------------------------------------------------------

function smooth_fg!(glr::GLR{LogisticLoss,<:ENR}, X, y)
    smooth = get_smooth(glr)
    (g, θ) -> fgh!(smooth, X, y)(0.0, g, nothing, θ)
end

# ---------------------------------- #
#  -- Multinomial Regression (L2) -- #
# ---------------------------------- #
# ->  c is the number of classes, θ has dims p * c
# ->  P = X * θ
# -> Zᵢ = ∑ exp(Pᵢ)
# -> Λ  = Diagonal(-Z)
# ->  f(θ)   = ∑(log Zᵢ - P[i, y[i]]) +  λ|θ|₂²
# -> ∇f(θ)   = reshape(X'ΛM, c * p)
# -> ∇²f(θ)v = via R operator
# NOTE:
# * yᵢ ∈ {1, 2, ..., c}
# ---------------------------------------------------------

function fg!(glr::GLR{MultinomialLoss,<:L2R}, X, y)
    n, p = size(X)
    c    = length(unique(y))
    λ    = getscale(glr.penalty)
    (f, g, θ) -> begin
        P  = SCRATCH_NC[]
        apply_X!(P, X, θ, c)                                 # O(npc) store n * c
        M  = SCRATCH_NC2[]
        M .= exp.(P)                                         # O(npc) store n * c
        g === nothing || begin
            ΛM  = SCRATCH_NC3[]
            ΛM .= M ./ sum(M, dims=2)                        # O(nc)  store n * c
            Q   = SCRATCH_NC4[]
            @inbounds for i = 1:n, j=1:c
                Q[i, j] = ifelse(y[i] == j, 1.0, 0.0)
            end
            ∑ΛM = sum(ΛM, dims=1)
            ∑Q  = sum(Q, dims=1)
            R   = ΛM
            R .-= Q
            G   = SCRATCH_PC[]
            if glr.fit_intercept
                mul!(view(G, 1:p, :), X', R)
                @inbounds for k in 1:c
                    G[end, k] = ∑ΛM[k] - ∑Q[k]
                end
            else
                mul!(G, X', R)
            end
            g  .= reshape(G, (p + Int(glr.fit_intercept)) * c)
            g .+= λ .* θ
            glr.fit_intercept && (glr.penalize_intercept || (g[end] -= λ * θ[end]))
        end
        f === nothing || begin
            # we re-use pre-computations here, see also MultinomialLoss
            # ms = maximum(P, dims=2)
            # ss = sum(M ./ exp.(ms), dims=2)
            ms   = maximum(P, dims=2)
            ems  = SCRATCH_N[]
            @inbounds for i in 1:n
                ems[i] = exp(ms[i])
            end
            ΛM  = SCRATCH_NC2[] # note that _NC is already linked to P
            ΛM .= M ./ ems
            ss  = sum(ΛM, dims=2)
            t   = 0.0
            @inbounds for i in eachindex(y)
                t += log(ss[i]) + ms[i] - P[i, y[i]]
            end
            return sum(t) + glr.penalty(view_θ(glr, θ))
        end
    end
end

function Hv!(glr::GLR{MultinomialLoss,<:L2R}, X, y)
    p = size(X, 2)
    λ = getscale(glr.penalty)
    c = length(unique(y))
    # NOTE:
    # * ideally P and Q should be recuperated from gradient computations (fghv!)
    # * assumption that c is small so that storing matrices of size n * c is not too bad; if c
    # is large and allocations should be minimized, all these computations can be done per class
    # with views over (c-1)p+1:cp; it will allocate less but is likely slower; maybe in the future
    # we could have a keyword indicating which one the user wants to use.
    (Hv, θ, v) -> begin
        P  = apply_X(X, θ, c)    # P_ik = <x_i, θ_k>                  // dims n * c; O(npc)
        Q  = apply_X(X, v, c)    # Q_ik = <x_i, v_k>                  // dims n * c; O(npc)
        M  = exp.(P)             # M_ik = exp<x_i, w_k>               // dims n * c;
        MQ = M .* Q              #                                    // dims n * c; O(nc)
        ρ  = 1 ./ sum(M, dims=2) # ρ_i = 1/Z_i = 1/∑_k exp<x_i, w_k>
        κ  = sum(MQ, dims=2)     # κ_i  = ∑_k exp<x_i, w_k><x_i, v_k>
        γ  = κ .* ρ.^2           # γ_i  = κ_i / Z_i^2
        # computation of Hv
        U      = (ρ .* MQ) .- (γ .* M)                              # // dims n * c; O(nc)
        Hv_mat = X' * U                                             # // dims n * c; O(npc)
        if glr.fit_intercept
            Hv .= reshape(vcat(Hv_mat, sum(U, dims=1)), (p+1)*c)
        else
            Hv .= reshape(Hv_mat, p * c)
        end
        Hv .+= λ .* v
        glr.fit_intercept && (glr.penalize_intercept || (Hv[end] -= λ * v[end]))
    end
end

# -------------------------------------- #
#  -- L1/Elnet Multinomial Regression -- #
# -------------------------------------- #
# ->  J(θ)  = f(θ) + r(θ)
# ->  f(θ)  = MN + λ|θ|₂²  // smooth (MN = MultinomialLoss)
# ->  r(θ)  = γ|θ|₁        // non-smooth with prox
# -> ∇f(θ)  = ∇MN + λθ
# -> ∇²f(θ) = ∇²MN + λI
# -> prox_r = soft-thresh
# ---------------------------------------------------------

function smooth_fg!(glr::GLR{MultinomialLoss,<:ENR}, X, y)
    smooth = get_smooth(glr)
    (g, θ) -> fg!(smooth, X, y)(0.0, g, θ)
end
