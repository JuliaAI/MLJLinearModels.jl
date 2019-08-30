# ADMM methods

# NOTE: I was not able to run either of those methods
# one thing that wasn't done properly is to re-factorize H every-time ρ is changed
# but that makes these algorithms super inefficient.
# without modifying ρ, I was not able to get the algorithms not to explode. Interestingly
# when running a litteral translation of https://web.stanford.edu/~boyd/papers/admm/least_abs_deviations/lad.html
# it also exploded.

# function _fit(glr::GLR{L1Loss,<:L2R}, solver::ADMM, X, y)
#     n, p = size(X)
#     λ    = getscale(glr.penalty)
#     φ    = 1.0 / solver.rho
#     λφ   = λ * φ
#     # pre-computations
#     H = form_XtX(X, glr.fit_intercept, λφ) # O(np²)
#     cholesky!(H) # O(p³) important assumption p << n
#     # cache
#     p_  = p + Int(glr.fit_intercept) # effective dim
#     θ   = zeros(p_)    # cache for current  θ
#     θ_  = zeros(p_)    # cache for previous θ
#     b   = zeros(p_)    # cache for right-hand-side of θ-update
#     bₐ  = view(b, 1:p) # only used in case of glr.fit_intercept
#     Xθ  = zeros(n)     # cache for over-relaxed X*θ
#     z   = zeros(n)     # cache for complimentary variable
#     u   = zeros(n)     # cache for dual variable
#     t   = zeros(n)     # cache for z .+ y .- u
#     zpy = copy(y)      # cache for z + y which is used multiple times
#     # loop-related
#     k, tol = 1, Inf
#     while k ≤ solver.max_iter && tol > solver.tol
#         # θ-update
#         # >> θ = (X'X + λ/ρ I) \ X'(y + z - u)  -- O(max{np, p²})
#         t .= zpy .- u
#         if glr.fit_intercept
#             mul!(bₐ, X', t)
#             b[end] = sum(t)
#         else
#             mul!(b, X', t)
#         end
#         ldiv!(θ, H, b)
#         # over-relaxation
#         # >> Xθ = α * (X * θ) + (1-α) * (z + y)   -- O(np)
#         apply_X!(Xθ, X, θ)
#         Xθ .= solver.alpha .* Xθ .+ (1.0 - solver.alpha) .* zpy
#         # z-update          -- O(n)
#         z  .= soft_thresh.(Xθ .- y .+ u, φ)
#         # u-update
#         zpy .= z .+ y
#         u  .+= Xθ .- zpy
#         # update tolerance
#         tol = norm(θ .- θ_) / (norm(θ) + eps())
#         copyto!(θ_, θ)
#         # update niter
#         k += 1
#     end
#     tol ≤ solver.tol || @warn "ADMM did not converge in $(solver.max_iter)."
#     return θ
# end
#
#
# # FADMM methods
#
# function _fit(glr::GLR{L1Loss,<:L2R}, solver::FADMM, X, y)
#     n, p = size(X)
#     λ    = getscale(glr.penalty)
#     ρ    = solver.rho
#     η    = solver.eta   # linked to restart frequency
#     τ    = solver.tau   # linked to updating ρ
#     μ    = solver.mu    # linked to updating ρ
#     # XXX store that in solver
#     ϵ_abs = 1e-3
#     ϵ_rel = 1e-3
#     # pre-computations
#     H = form_XtX(X, glr.fit_intercept, λ/ρ) # O(np²)
#     cholesky!(H) # O(p³) important assumption p << n
#     # cache
#     p_   = p + Int(glr.fit_intercept) # effective dim
#     θ    = zeros(p_)    # cache for current  θ
#     b    = zeros(p_)    # cache for right-hand-side of θ-update
#     bₐ   = view(b, 1:p)
#     Xθ   = zeros(n)     # cache for X*θ
#     z    = zeros(n)
#     z_   = zeros(n)
#     zmz_ = zeros(n)     # cache for (z-z_) used in dual residual and accel
#     u    = zeros(n)
#     u_   = zeros(n)
#     ẑ    = zeros(n)
#     û    = zeros(n)
#     t    = zeros(n)     # cache for θ update
#     r    = zeros(n)     # cache for primal residual
#     s    = zeros(p_)    # cache for dual residual
#     sₐ   = view(s, 1:p)
#     α    = 1.0          # cache for acceleration step
#     c_   = Inf          # force accelerate on first step
#     # loop-related
#     k = 1
#     while k ≤ solver.max_iter
#         # ----------------------------------------
#         # compute ϵ-primal and ϵ-dual XXX can store precomp
#         ϵ_primal = sqrt(p_) * ϵ_abs + ϵ_rel * max(norm(Xθ), norm(z), norm(y))
#         apply_Xt!(bₐ, X, u)
#         ϵ_dual   = sqrt(n) * ϵ_abs  + ϵ_rel * norm(b) / ρ
#         # ----------------------------------------
#         # θ-update
#         # >> θ = argmin_θ L(θ) - ⟨û,Xθ⟩ + ρ/2 |y-Xθ+ẑ|₂² with L(θ) = λ|θ|₂²/2
#         # >> θ = (X'X + λ/ρ) \ X'(y + ẑ + û/ρ) = H \ X't
#         t .= y .+ ẑ .+ û ./ ρ
#         apply_Xt!(bₐ, X, t)     # X'(x + ẑ + û/ρ)
#         ldiv!(θ, H, b)          # H = (X'X+λ/ρ)
#         # ----------------------------------------
#         # z-update
#         # >> z = argmin_z P(z) + ⟨û,z⟩ + ρ/2 |y-Xθ+z|₂² with P(z) = |z|₁
#         # >> z = S_{1/ρ}(Xθ - y - û/ρ)
#         apply_X!(Xθ, X, θ)
#         z .= soft_thresh.(Xθ .- y .- û ./ ρ, 1.0/ρ)
#         # ----------------------------------------
#         # r and u-update
#         # >> u = û + ρ(y - Xθ + z)
#         r .= Xθ .- z .- y  # primal residual
#         u .= û .- ρ .* r
#         # ----------------------------------------
#         # c-update
#         # >> c = 1/ρ norm(u - û)^2 + ρ norm(z - ẑ)^2
#         # note that (u-û).^2 = ρ²r.^2
#         c = ρ * sum(abs2, r) + ρ * sum(abs2, z .- ẑ)
#         # ----------------------------------------
#         # accelerate or restart
#         zmz_ .= z .- z_
#         if c < η * c_
#             println("accel")
#             # accelerate
#             α_  = α #  store α_k
#             α   = (1.0 + sqrt(1.0 + 4.0 * α^2)) / 2.0  # α_{k+1} from α_k
#             ζ   = (α_ - 1.0) / α                       # (α_{k}-1) / α_{k+1}
#             ẑ  .= z .+ ζ .* zmz_
#             û  .= u .+ ζ .* (u .- u_)
#         else
#             println("restart")
#             # restart
#             α = 1.0
#             copyto!(ẑ, z_)
#             copyto!(û, u_)
#             c = c_ / η
#         end
#         # ----------------------------------------
#         # ρ-update depending on primal-dual gap
#         apply_Xt!(sₐ, X, zmz_) # dual residual -ρX'(z-z_) without the ρ
#         norm_r = norm(r)
#         norm_s = ρ * norm(s)
#         if norm_r / ϵ_primal > μ * norm_s / ϵ_dual
#             ρ *= τ
#         elseif norm_s / ϵ_dual > μ * norm_r / ϵ_primal
#             ρ /= τ
#         end
#         # ----------------------------------------
#         # cache update
#         c_  = c         # next c_{k-1} (used in restart step)
#         copyto!(z_, z)  # next z_{k-1} (used in acceleration step)
#         copyto!(u_, u)  # next u_{k-1} (used in acceleration step)
#         # check if convergence
#         norm_r ≤ ϵ_primal && norm_s ≤ ϵ_dual && break
#         # update niter
#         k += 1
#     end
#     k == solver.max_iter+1 && @warn "FADMM did not converge in $(solver.max_iter)."
#     return θ
# end
