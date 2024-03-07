# Proximal gradient methods

# Assumption: loss has gradient; penalty has prox e.g.: Lasso
# J(θ) = f(θ) + r(θ) where f is smooth
function _fit(::Type{T}, glr::GLR, solver::ProxGrad, X, y, scratch) where {T<:Real}
    n,p,c = npc(scratch)
    c > 0 && (p *= c)
    # vector caches + eval cache
    θ   = zeros(T, p)   # θ_k
    Δθ  = zeros(T, p)   # (θ_k - θ_{k-1})
    θ̄   = zeros(T, p)   # θ_k + ρ Δθ // extrapolation
    ∇fθ̄ = zeros(T, p)
    fθ̄  = T(0.0)        # useful for backtracking function
    θ̂   = zeros(T, p)   # candidate before becoming θ_k
    # cache for extrapolation constant and stepsizes
    ω   = T(0.0)   # ω_k
    ω_  = T(0.0)   # ω_{k-1}
    ω__ = T(0.0)   # ω_{k-2}
    η   = T(1.0)   # stepsize (1/L)
    acc = ifelse(solver.accel, T(1.0), T(0.0)) # if 0, no extrapolation (ISTA)
    # functions
    _f = if solver.gram
        smooth_gram_objective(T, glr, X, y, n)
    else
        smooth_objective(T, glr, X, y; c=c)
    end

    _fg! = if solver.gram
        smooth_gram_fg!(glr, X, y, n)
    else
        smooth_fg!(glr, X, y, scratch)
    end
    _prox!  = prox!(glr, n)
    bt_cond = θ̂ ->
                _f(θ̂) > fθ̄ + dot(θ̂ .- θ̄, ∇fθ̄) + sum(abs2.(θ̂ .- θ̄)) / (T(2)*η)
    # loop-related
    k, tol = 1, Inf
    while k ≤ solver.max_iter && tol > solver.tol
        # --------------------------------------------------
        # This loop corresponds to the implementation of the
        # FISTA + Backtracking  algorithm in Beck & Teboulle
        # "A Fast Iterative Shrinkage Thresholding Algorithm
        # for Linear Inverse Problems" (page 193)
        # --------------------------------------------------
        # 1. linear extrapolation of past iterates
        ω   = (T(1.0) + sqrt(T(1.0) + T(4.0) * ω_^T(2))) / T(2.0)
        ρ   = acc * ω__ / ω  # ω_{k-2}/ω; note that ρ != 0 only as k > 2
        θ̄  .= θ + ρ * Δθ
        # 2. attempt a prox step, modify the step until verifies condition
        fθ̄  = _fg!(∇fθ̄, θ̄)          # f and ∇f at θ̄
        _prox!(θ̂, η, θ̄ .- η .* ∇fθ̄) # candidate update
        inner = 0
        while bt_cond(θ̂) && inner < solver.max_inner
            η *= solver.beta            # shrink stepsize
            _prox!(θ̂, η, θ̄ .- η .* ∇fθ̄) # try another candidate
            inner += 1
        end
        if inner == solver.max_inner
            @warn "No appropriate stepsize found via backtracking; " *
                  "interrupting. The reason could be input data that is not standardized."
            break
        end
        # update caches
        ω__ = ω_
        ω_  = ω
        Δθ .= θ̂ .- θ
        copyto!(θ, θ̂)
        # update tolerance
        tol = norm(Δθ) / (norm(θ) + eps())
        # update niter
        k += 1
    end
    tol ≤ solver.tol || @warn "Proximal GD did not converge in " *
                              "$(solver.max_iter) iterations."
    return θ
end

function _fit(glr::GLR, solver::ProxGrad, X, y, scratch)
    return _fit(eltype(X), glr, solver, X, y, scratch)
end