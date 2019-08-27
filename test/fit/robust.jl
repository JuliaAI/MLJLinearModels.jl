n, p = 500, 5
((X, y, θ), (X_, y1, θ1)) = generate_continuous(n, p; seed=525)

@testset "HuberReg" begin
    # No intercept
    δ = 0.01
    λ = 3.0
    hr = HuberRegression(δ, λ, fit_intercept=false)
    J = objective(hr, X, y)
    o = RobustLoss(Huber(δ)) + λ * L2Penalty()
    @test J(θ) == o(y, X*θ, θ)
    @test J(θ)          ≤ 10.61
    θ_newton = fit(hr, X, y, solver=Newton())
    @test J(θ_newton)   ≤ 7.71
    θ_newtoncg = fit(hr, X, y, solver=NewtonCG())
    @test J(θ_newtoncg) ≤ 7.71
    θ_lbfgs = fit(hr, X, y, solver=LBFGS())
    @test J(θ_lbfgs)    ≤ 7.71
    θ_iwls  = fit(hr, X, y, solver=IWLSCG())
    @test J(θ_iwls)     ≤ 7.71

    δ = 0.01
    λ = 3.0
    hr = HuberRegression(δ, λ)
    J = objective(hr, X, y1)
    o = RobustLoss(Huber(δ)) + λ * L2Penalty()
    @test J(θ1) == o(y1, X_*θ1, θ1)
    @test J(θ1)         ≤ 16.37
    θ_newton = fit(hr, X, y1, solver=Newton())
    @test J(θ_newton)   ≤ 10.52
    θ_newtoncg = fit(hr, X, y1, solver=NewtonCG())
    @test J(θ_newtoncg) ≤ 10.52
    θ_lbfgs = fit(hr, X, y1, solver=LBFGS())
    @test J(θ_lbfgs)    ≤ 10.52
    θ_iwls  = fit(hr, X, y1, solver=IWLSCG())
    @test J(θ_iwls)     ≤ 10.52
end

@testset "AndrewsReg" begin
    δ = 0.1
    λ = 3.0
    rr = RobustRegression(rho=Andrews(δ), lambda=λ)
    J = objective(rr, X, y1)
    o = RobustLoss(AndrewsRho(δ)) + λ * L2Penalty()
    @test J(θ1) == o(y1, X_*θ1, θ1)
    @test J(θ1)         ≤ 16.12
    θ_newton = fit(rr, X, y1, solver=Newton())
    @test J(θ_newton)   ≤ 0.491
    θ_newtoncg = fit(rr, X, y1, solver=NewtonCG())
    @test J(θ_newtoncg) ≤ 0.492
    θ_lbfgs = fit(rr, X, y1, solver=LBFGS())
    @test J(θ_lbfgs)    ≤ 0.492
    θ_iwls  = fit(rr, X, y1, solver=IWLSCG())
    @test J(θ_iwls)     ≤ 0.492
end

@testset "BisquareReg" begin
    δ = 0.1
    λ = 3.0
    rr = RobustRegression(rho=Bisquare(δ), lambda=λ)
    J = objective(rr, X, y1)
    o = RobustLoss(BisquareRho(δ)) + λ * L2Penalty()
    @test J(θ1) == o(y1, X_*θ1, θ1)
    @test J(θ1)         ≤ 16.55
    θ_newton = fit(rr, X, y1, solver=Newton())
    @test J(θ_newton)   ≤ 0.822
    θ_newtoncg = fit(rr, X, y1, solver=NewtonCG())
    @test J(θ_newtoncg) ≤ 0.822
    θ_lbfgs = fit(rr, X, y1, solver=LBFGS())
    @test J(θ_lbfgs)    ≤ 0.822
    θ_iwls  = fit(rr, X, y1, solver=IWLSCG())
    @test J(θ_iwls)     ≤ 0.822
end

@testset "LogisticRReg" begin
    δ = 1.5
    λ = 1.0
    rr = RobustRegression(rho=Logistic(delta=δ), lambda=λ)
    J = objective(rr, X, y1)
    o = RobustLoss(LogisticRho(δ)) + λ * L2Penalty()
    @test J(θ1) ≈ o(y1, X_*θ1, θ1)
    @test J(θ1)         ≤ 7.70
    θ_newton = fit(rr, X, y1, solver=Newton())
    @test J(θ_newton)   ≤ 7.67
    θ_newtoncg = fit(rr, X, y1, solver=NewtonCG())
    @test J(θ_newtoncg) ≤ 7.67
    θ_lbfgs = fit(rr, X, y1, solver=LBFGS())
    @test J(θ_lbfgs)    ≤ 7.67
    θ_iwls  = fit(rr, X, y1, solver=IWLSCG())
    @test J(θ_iwls)     ≤ 7.67
end

@testset "FairReg" begin
    δ = 0.1
    λ = 3.0
    rr = RobustRegression(rho=Fair(δ), lambda=λ)
    J = objective(rr, X, y1)
    o = RobustLoss(Fair(δ)) + λ * L2Penalty()
    @test J(θ1) ≈ o(y1, X_*θ1, θ1)
    @test J(θ1)         ≤ 17.27
    θ_newton = fit(rr, X, y1, solver=Newton())
    @test J(θ_newton)   ≤ 16.97
    θ_newtoncg = fit(rr, X, y1, solver=NewtonCG())
    @test J(θ_newtoncg) ≤ 16.97
    θ_lbfgs = fit(rr, X, y1, solver=LBFGS())
    @test J(θ_lbfgs)    ≤ 16.97
    θ_iwls  = fit(rr, X, y1, solver=IWLSCG())
    @test J(θ_iwls)     ≤ 16.97
end

@testset "TalwarReg" begin
    δ = 0.1
    λ = 3.0
    rr = RobustRegression(rho=Talwar(δ), lambda=λ)
    J = objective(rr, X, y1)
    o = RobustLoss(Talwar(δ)) + λ * L2Penalty()
    @test J(θ1) == o(y1, X_*θ1, θ1)
    @test J(θ1)         ≤ 17.26
    θ_newton = fit(rr, X, y1, solver=Newton())
    @test J(θ_newton)   ≤ 2.45
    θ_newtoncg = fit(rr, X, y1, solver=NewtonCG())
    @test J(θ_newtoncg) ≤ 2.45
    θ_lbfgs = fit(rr, X, y1, solver=LBFGS())
    @test J(θ_lbfgs)    ≤ 2.45
    θ_iwls  = fit(rr, X, y1, solver=IWLSCG())
    @test J(θ_iwls)     ≤ 2.45
end
