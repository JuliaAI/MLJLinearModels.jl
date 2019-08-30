function _fit(glr::GLR{RobustLoss{ρ},<:L2R}, solver::IWLSCG, X, y) where {ρ}
    λ    = getscale(glr.penalty)
    n    = size(X, 1)
    p    = size(X, 2) + Int(glr.fit_intercept)
    _Mv! = Mv!(glr, X, y; threshold=solver.threshold)
    κ    = solver.damping # between 0 and 1, 1 = fully take the new iteration
    # cache
    θ  = zeros(p)
    θ_ = zeros(p)
    b  = zeros(p)  # will contain X'Wy
    ω  = zeros(n)  # will contain the diagonal of W
    # params for the  loop
    max_cg_steps = min(solver.max_inner, p)
    k, tol = 0, Inf
    while k < solver.max_iter && tol > solver.tol
        # update the weights and retrieve the application function
        # Mθv! corresponds to the current application of (X'WX + λI) on v
        Mθv! = _Mv!(ω, θ)
        Mm  = LinearMap(Mθv!, p; ismutating=true, isposdef=true, issymmetric=true)
        Wy  = ω .* y
        b   = X'Wy
        if glr.fit_intercept
            b = vcat(b, sum(Wy))
        end
        # update
        θ  .= (1-κ) .* θ .+ κ .* cg(Mm, b; maxiter=max_cg_steps)
        # check tolerance
        tol = norm(θ .- θ_) / (norm(θ) + eps())
        # update cache
        copyto!(θ_, θ)
        k  += 1
    end
    tol ≤ solver.tol || @warn "IWLS did not converge in $(solver.max_iter) iterations."
    return θ
end
