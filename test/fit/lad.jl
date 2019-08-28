n, p = 500, 3
((X, y, θ), (X_, y1, θ1)) = generate_continuous(n, p; seed=5225)

@testset "LADReg" begin
    λ  = 1.0
    lr = LADRegression(λ; fit_intercept=false)
    J  = objective(lr, X, y)
    o  = L1Loss() + λ * L2Penalty()
    @test J(θ) == o(y, X*θ, θ)
    @test J(θ) ≤ 39.95
    θ_admm  = fit(lr, X, y, solver=FADMM(rho=0.1, max_iter=100))
    θ_fadmm = fit(lr, X, y, solver=FADMM(rho=0.1, max_iter=100))
    @test_broken J(θ_admm) ≤ 39
    @test_broken J(θ_fadmm) ≤ 39
end
