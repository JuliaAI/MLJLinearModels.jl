n, p = 500, 5
((X, y, θ), (X_, y1, θ1)) = generate_continuous(n, p; seed=525)

# adding some outliers (both positive and negative)
Random.seed!(543)
y1a = outlify(y1, 0.1)

@testset "QuantileReg" begin
    δ = 0.5 # effectively LAD regression
    λ = 1.0
    rr = QuantileRegression(δ, lambda=λ)
    J = objective(rr, X, y1a)
    o = RobustLoss(Quantile(δ)) + λ * L2Penalty()
    @test J(θ1) ≈ o(y1a, X_*θ1, θ1)
    ls = LinearRegression()
    θ_ls    = fit(ls, X, y1a)
    θ_lbfgs = fit(rr, X, y1a, solver=LBFGS())
    θ_iwls  = fit(rr, X, y1a, solver=IWLSCG())
    @test isapprox(J(θ1),      491.94661, rtol=1e-5)
    @test isapprox(J(θ_ls),    614.70403, rtol=1e-5)  # note that LS is crap due to outliers
    @test isapprox(J(θ_lbfgs), 491.65694, rtol=1e-5)
    @test isapprox(J(θ_iwls),  491.65694, rtol=1e-5)

    # NOTE: newton and newton-cg not available because ϕ = 0 identically
    # will throw an error if called.
    @test_throws ErrorException fit(rr, X, y1, solver=Newton())
    @test_throws ErrorException fit(rr, X, y1, solver=NewtonCG())

    if DO_COMPARISONS
        # Compare with R's QuantReg package
        # NOTE: QuantReg doesn't allow for penalties so re-fitting with λ=0
        rr = QuantileRegression(δ, lambda=0)
        J  = objective(rr, X, y1a)
        θ_ls     = fit(LinearRegression(), X, y1a)
        θ_lbfgs  = fit(rr, X, y1a, solver=LBFGS())
        θ_iwls   = fit(rr, X, y1a, solver=IWLSCG())
        θ_qr_br  = rcopy(QUANTREG.rq_fit_br(X_, y1a))[:coefficients]
        θ_qr_fnb = rcopy(QUANTREG.rq_fit_fnb(X_, y1a))[:coefficients]
        # NOTE: we take θ_qr_br as reference point
        @test isapprox(J(θ_ls), 610.41023,  rtol=1e-5)
        @test J(θ_qr_br) ≈      486.36730 # <- ref value
        # Their IP algorithm essentially gives the same answer
        @test (J(θ_qr_fnb) - J(θ_qr_br)) ≤ 1e-10
        # Our algorithms are close enough
        @test isapprox(J(θ_lbfgs), 486.36730, rtol=1e-5)
        @test isapprox(J(θ_iwls),  486.36730, rtol=1e-4)
    end
end

###########################
## With Sparsity penalty ##
###########################

n, p = 500, 100
((X, y, θ), (X_, y1, θ1)) = generate_continuous(n, p;  seed=51112, sparse=0.1)
# pepper with outliers
y1a  = outlify(y1, 0.1)

@testset "LAD+L1" begin
    λ = 5.0
    γ = 10.0
    rr = LADRegression(λ, γ)
    J  = objective(rr, X, y1a)
    θ_ls    = X_ \ y1a
    θ_fista = fit(rr, X, y1a, solver=FISTA())
    θ_ista  = fit(rr, X, y1a, solver=ISTA())
    @test isapprox(J(θ_ls),    943.33942, rtol=1e-5)
    @test isapprox(J(θ_fista), 390.53177, rtol=1e-5)
    @test isapprox(J(θ_ista),  390.53177, rtol=1e-3)
    @test nnz(θ_fista) == 55
    @test nnz(θ_ista)  == 51 # interesting, things look a bit unstable?

    if DO_COMPARISONS
        # Compare with R's QuantReg package
        # NOTE: QuantReg doesn't apply the penalty on the intercept
        rr = LADRegression(5.0; penalty=:l1)
        J  = objective(rr, X, y1a)
        θ_ls       = X_ \ y1a
        θ_fista    = fit(rr, X, y1a, solver=FISTA())
        θ_ista     = fit(rr, X, y1a, solver=ISTA())
        θ_qr_lasso = rcopy(QUANTREG.rq_fit_lasso(X_, y1a))[:coefficients]
        # NOTE: we take θ_qr_br as reference point
        @test isapprox(J(θ_ls),       803.57950, rtol=1e-5)
        @test isapprox(J(θ_qr_lasso), 373.21119, rtol=1e-5) # <- ref value
        # Our algorithms are close enough
        @test isapprox(J(θ_fista),    373.21119, rtol=1e-2)
        @test isapprox(J(θ_ista),     373.21119, rtol=1e-2)
        # in this case we do a fair bit better; we may be applying the penalty differently though
        @test nnz(θ_qr_lasso) == 101
        @test nnz(θ_fista)    == 85
        @test nnz(θ_ista)     == 85
        # in this case fista is best
        @test J(θ_fista) < J(θ_ista) < J(θ_qr_lasso)
    end
end
