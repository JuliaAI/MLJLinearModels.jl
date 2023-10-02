# Proximal gradient methods

# Assumption: loss has gradient; penalty has prox e.g.: Lasso
# J(θ) = f(θ) + r(θ) where f is smooth
function _fit(glr::GLR, solver::ProxGrad, X, y, scratch)
    n,p,c = npc(scratch)
    c > 0 && (p *= c)
    # vector caches + eval cache
    θ   = zeros(p)   # θ_k
    Δθ  = zeros(p)   # (θ_k - θ_{k-1})
    θ̄   = zeros(p)   # θ_k + ρ Δθ // extrapolation
    ∇fθ̄ = zeros(p)
    fθ̄  = 0.0        # useful for backtracking function
    θ̂   = zeros(p)   # candidate before becoming θ_k
    # cache for extrapolation constant and stepsizes
    ω   = 0.0   # ω_k
    ω_  = 0.0   # ω_{k-1}
    ω__ = 0.0   # ω_{k-2}
    η   = 1.0   # stepsize (1/L)
    acc = ifelse(solver.accel, 1.0, 0.0) # if 0, no extrapolation (ISTA)
    # functions
    _f = if solver.gram
        smooth_gram_objective(glr, X, y, n)
    else
        smooth_objective(glr, X, y; c=c)
    end

    _fg! = if solver.gram
        smooth_gram_fg!(glr, X, y, n)
    else
        smooth_fg!(glr, X, y, scratch)
    end
    _prox!  = prox!(glr, n)
    bt_cond = θ̂ ->
                _f(θ̂) > fθ̄ + dot(θ̂ .- θ̄, ∇fθ̄) + sum(abs2.(θ̂ .- θ̄)) / (2η)
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
        ω   = (1.0 + sqrt(1.0 + 4.0 * ω_^2)) / 2.0
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
