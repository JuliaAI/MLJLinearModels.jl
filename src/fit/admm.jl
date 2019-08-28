# ADMM methods

function _fit(glr::GLR{L1Loss,<:L2R}, solver::ADMM, X, y)
    n, p = size(X)
    λ    = getscale(glr.penalty)
    φ    = 1.0 / solver.rho
    λφ   = λ * φ
    # pre-computations
    H = form_XtX(X, glr.fit_intercept, λφ) # O(np²)
    cholesky!(H) # O(p³) important assumption p << n
    # cache
    p_  = p + Int(glr.fit_intercept) # effective dim
    θ   = zeros(p_)    # cache for current  θ
    θ_  = zeros(p_)    # cache for previous θ
    b   = zeros(p_)    # cache for right-hand-side of θ-update
    bₐ  = view(b, 1:p) # only used in case of glr.fit_intercept
    Xθ  = zeros(n)     # cache for over-relaxed X*θ
    z   = zeros(n)     # cache for complimentary variable
    u   = zeros(n)     # cache for dual variable
    t   = zeros(n)     # cache for z .+ y .- u
    zpy = copy(y)      # cache for z + y which is used multiple times
    # loop-related
    k, tol = 1, Inf
    while k ≤ solver.max_iter && tol > solver.tol
        # θ-update
        # >> θ = (X'X + λ/ρ I) \ X'(y + z - u)  -- O(max{np, p²})
        t .= zpy .- u
        if glr.fit_intercept
            mul!(bₐ, X', t)
            b[end] = sum(t)
        else
            mul!(b, X', t)
        end
        ldiv!(θ, H, b)
        # over-relaxation
        # >> Xθ = α * (X * θ) + (1-α) * (z + y)   -- O(np)
        apply_X!(Xθ, X, θ)
        Xθ .= solver.alpha .* Xθ .+ (1.0 - solver.alpha) .* zpy
        # z-update          -- O(n)
        z  .= soft_thresh.(Xθ .- y .+ u, φ)
        # u-update
        zpy .= z .+ y
        u  .+= Xθ .- zpy
        # update tolerance
        tol = norm(θ .- θ_) / (norm(θ) + eps())
        copyto!(θ_, θ)
        # update niter
        k += 1
    end
    tol ≤ solver.tol || @warn "ADMM did not converge in $(solver.max_iter)."
    return θ
end


# FADMM methods

function _fit(glr::GLR{L1Loss,<:L2R}, solver::FADMM, X, y)
    n, p = size(X)
    λ    = getscale(glr.penalty)
    ρ    = solver.rho
    η    = solver.eta
    # pre-computations
    H = form_XtX(X, glr.fit_intercept, λ/ρ) # O(np²)
    cholesky!(H) # O(p³) important assumption p << n
    # cache
    p_  = p + Int(glr.fit_intercept) # effective dim
    θ   = zeros(p_)    # cache for current  θ
    θ_  = zeros(p_)    # cache for previous θ
    b   = zeros(p_)    # cache for right-hand-side of θ-update
    bₐ  = view(b, 1:p) # only used in case of glr.fit_intercept
    Xθ  = zeros(n)     # cache for X*θ
    z   = zeros(n)
    z_  = zeros(n)
    u   = zeros(n)
    u_  = zeros(n)
    ẑ   = zeros(n)
    û   = zeros(n)
    t   = zeros(n)
    α   = 1.0
    α__ = 1.0
    c_  = Inf  # force accelerate on first step
    # loop-related
    k, tol = 1, Inf
    while k ≤ solver.max_iter && tol > solver.tol
        # θ-update [u in ref]
        # >> θ = (X'X + λ/ρ) \ X'(y + ẑ + û/ρ)
        t .= y .+ ẑ .+ û ./ ρ
        if glr.fit_intercept
            mul!(bₐ, X', t)
            b[end] = sum(t)
        else
            mul!(b, X', t)
        end
        ldiv!(θ, H, b)
        # z-update [v in ref]
        # >> z = S_{1/ρ}(Xθ - y - û/ρ)
        apply_X!(Xθ, X, θ)
        z .= soft_thresh.(Xθ .- y .- û ./ ρ, 1.0/ρ)
        # u-update [λ in ref]
        # >> u = û + ρ(y - Aθ + z)
        u .= û .+ ρ .* (y .- Xθ .+ z)
        # c-update
        # >> c = 1/ρ norm(λ - λ̂)^2 + ρ norm(z - ẑ)^2
        c = 1.0/ρ * norm(u - û)^2 + ρ * norm(z - ẑ)^2
        # check if accelerate or restart
        α_ = α  # store α_{k} to update α_{k-1} later
        if c < η * c_
            # accelerate
            α   = (1.0+sqrt(1.0 + 4.0 * α^2))/2  # α_{k+1} from α_k
            ζ   = α__ / α                        # α_{k-1} / α_{k+1}
            ẑ  .= z .+ ζ .* (z .- z_)
            û  .= u .+ ζ .* (u .- u_)
        else
            # restart
            α = 1.0
            copyto!(ẑ, z_)
            copyto!(û, u_)
            c = c_ / η
        end
        # update cache
        α__ = α_        # next α_{k-1} (used in acceleration step)
        c_  = c         # next c_{k-1} (used in restart step)
        copyto!(z_, z)  # next z_{k-1} (used in acceleration step)
        copyto!(u_, u)  # next u_{k-1} (used in acceleration step)
        # update tolerance
        tol = norm(θ .- θ_) / (norm(θ) + eps())
        copyto!(θ_, θ)
        # update niter
        k += 1
    end
    tol ≤ solver.tol || @warn "FADMM did not converge in $(solver.max_iter)."
    return θ
end
