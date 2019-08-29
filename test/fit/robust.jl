n, p = 500, 5
((X, y, θ), (X_, y1, θ1)) = generate_continuous(n, p; seed=525)

# NOTE: in these cases, if available, θ_newton is used as reference.
# so if the others are not far, it's assumed all have converged to the
# same (correct) minimizer. The reference J(θ) or J(θ1) is the objective
# at the generating θ; it should usually be >
# these are tests that should be considered as "sanity check" and are not
# stress tests testing corner cases.

@testset "HuberReg" begin
    # No intercept
    δ = 0.01
    λ = 3.0
    hr = HuberRegression(δ, λ; fit_intercept=false)
    J = objective(hr, X, y)
    o = RobustLoss(Huber(δ)) + λ * L2Penalty()
    @test J(θ) == o(y, X*θ, θ)
    θ_newton   = fit(hr, X, y, solver=Newton())
    θ_newtoncg = fit(hr, X, y, solver=NewtonCG())
    θ_lbfgs    = fit(hr, X, y, solver=LBFGS())
    θ_iwls     = fit(hr, X, y, solver=IWLSCG())
    @test (J(θ)          - 10.60971) ≤ 1e-5
    @test (J(θ_newton)   -  7.70113) ≤ 1e-5
    @test (J(θ_newtoncg) -  7.70113) ≤ 1e-5
    @test (J(θ_lbfgs)    -  7.70113) ≤ 1e-5
    @test (J(θ_iwls)     -  7.70113) ≤ 1e-5

    δ = 0.01
    λ = 3.0
    hr = HuberRegression(δ, λ)
    J = objective(hr, X, y1)
    o = RobustLoss(Huber(δ)) + λ * L2Penalty()
    @test J(θ1) == o(y1, X_*θ1, θ1)
    θ_newton   = fit(hr, X, y1, solver=Newton())
    θ_newtoncg = fit(hr, X, y1, solver=NewtonCG())
    θ_lbfgs    = fit(hr, X, y1, solver=LBFGS())
    θ_iwls     = fit(hr, X, y1, solver=IWLSCG())
    @test (J(θ1)         - 16.36661) ≤ 1e-5
    @test (J(θ_newton)   - 10.51384) ≤ 1e-5
    @test (J(θ_newtoncg) - 10.51384) ≤ 1e-5
    @test (J(θ_lbfgs)    - 10.51384) ≤ 1e-5
    @test (J(θ_iwls)     - 10.51384) ≤ 1e-5
end

# NOTE: small difference obtained with NCG and IWLS, might just be
# that the algorithm stops a bit earlier, negligible relative tol anyway.
@testset "AndrewsReg" begin
    δ = 0.1
    λ = 3.0
    rr = RobustRegression(rho=Andrews(δ), lambda=λ)
    J = objective(rr, X, y1)
    o = RobustLoss(AndrewsRho(δ)) + λ * L2Penalty()
    @test J(θ1) == o(y1, X_*θ1, θ1)
    θ_newton   = fit(rr, X, y1, solver=Newton())
    θ_newtoncg = fit(rr, X, y1, solver=NewtonCG())
    θ_lbfgs    = fit(rr, X, y1, solver=LBFGS())
    θ_iwls     = fit(rr, X, y1, solver=IWLSCG())
    @test (J(θ1)         - 16.11921) ≤ 1e-5
    @test (J(θ_newton)   -  0.49078) ≤ 1e-5
    @test (J(θ_newtoncg) -  0.49078) ≤ 1e-3
    @test (J(θ_lbfgs)    -  0.49078) ≤ 1e-5
    @test (J(θ_iwls)     -  0.49078) ≤ 1e-3
end

@testset "BisquareReg" begin
    δ = 0.1
    λ = 3.0
    rr = RobustRegression(rho=Bisquare(δ), lambda=λ)
    J = objective(rr, X, y1)
    o = RobustLoss(BisquareRho(δ)) + λ * L2Penalty()
    @test J(θ1) == o(y1, X_*θ1, θ1)
    θ_newton   = fit(rr, X, y1, solver=Newton())
    θ_newtoncg = fit(rr, X, y1, solver=NewtonCG())
    θ_lbfgs    = fit(rr, X, y1, solver=LBFGS())
    θ_iwls     = fit(rr, X, y1, solver=IWLSCG())
    @test (J(θ1)         - 16.54073) ≤ 1e-5
    @test (J(θ_newton)   -  0.82180) ≤ 1e-5
    @test (J(θ_newtoncg) -  0.82180) ≤ 1e-5
    @test (J(θ_lbfgs)    -  0.82180) ≤ 1e-5
    @test (J(θ_iwls)     -  0.82180) ≤ 1e-5
end

@testset "LogisticRReg" begin
    δ = 1.5
    λ = 1.0
    rr = RobustRegression(rho=Logistic(delta=δ), lambda=λ)
    J = objective(rr, X, y1)
    o = RobustLoss(LogisticRho(δ)) + λ * L2Penalty()
    @test J(θ1) ≈ o(y1, X_*θ1, θ1)
    θ_newton = fit(rr, X, y1, solver=Newton())
    θ_newtoncg = fit(rr, X, y1, solver=NewtonCG())
    θ_lbfgs = fit(rr, X, y1, solver=LBFGS())
    θ_iwls  = fit(rr, X, y1, solver=IWLSCG())
    @test (J(θ1)         - 7.69833) ≤ 1e-5
    @test (J(θ_newton)   - 7.66289) ≤ 1e-5
    @test (J(θ_newtoncg) - 7.66289) ≤ 1e-5
    @test (J(θ_lbfgs)    - 7.66289) ≤ 1e-5
    @test (J(θ_iwls)     - 7.66289) ≤ 1e-5
end

@testset "FairReg" begin
    δ = 0.1
    λ = 3.0
    rr = RobustRegression(rho=Fair(δ), lambda=λ)
    J = objective(rr, X, y1)
    o = RobustLoss(Fair(δ)) + λ * L2Penalty()
    @test J(θ1) ≈ o(y1, X_*θ1, θ1)
    θ_newton   = fit(rr, X, y1, solver=Newton())
    θ_newtoncg = fit(rr, X, y1, solver=NewtonCG())
    θ_lbfgs    = fit(rr, X, y1, solver=LBFGS())
    θ_iwls     = fit(rr, X, y1, solver=IWLSCG())
    @test (J(θ1)         - 17.26553) ≤ 1e-5
    @test (J(θ_newton)   - 16.96553) ≤ 1e-5
    @test (J(θ_newtoncg) - 16.96553) ≤ 1e-5
    @test (J(θ_lbfgs)    - 16.96553) ≤ 1e-5
    @test (J(θ_iwls)     - 16.96553) ≤ 1e-5
end

# NOTE: small difference obtained with NCG and IWLS, might just be
# that the algorithm stops a bit earlier, negligible relative tol anyway.
@testset "TalwarReg" begin
    δ = 0.1
    λ = 3.0
    rr = RobustRegression(rho=Talwar(δ), lambda=λ)
    J = objective(rr, X, y1)
    o = RobustLoss(Talwar(δ)) + λ * L2Penalty()
    @test J(θ1) == o(y1, X_*θ1, θ1)
    θ_newton   = fit(rr, X, y1, solver=Newton())
    θ_newtoncg = fit(rr, X, y1, solver=NewtonCG())
    θ_lbfgs    = fit(rr, X, y1, solver=LBFGS())
    θ_iwls     = fit(rr, X, y1, solver=IWLSCG())
    @test (J(θ1)         - 17.25562) ≤ 1e-5
    @test (J(θ_newton)   -  2.44343) ≤ 1e-5
    @test (J(θ_newtoncg) -  2.44343) ≤ 2e-3
    @test (J(θ_lbfgs)    -  2.44343) ≤ 1e-5
    @test (J(θ_iwls)     -  2.44343) ≤ 2e-3
end

# adding some outliers (both positive and negative)
Random.seed!(543)
y1a = y1 .+ 20 .* randn(n) .* (rand(n) .< 0.1)
# adding some outliers, all positive
y1b = y1 .+ 20 .* rand(n) .* (rand(n) .< 0.1)

@testset "QuantileReg" begin
    δ = 0.5 # LAD regression
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
end
