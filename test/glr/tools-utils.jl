n, p = 50, 5
((X, y, θ), (X1, y1, θ1)) = generate_continuous(n, p; seed=1234)

@testset "Tools" begin
    lr  = LogisticRegression(1.0, 2.0; fit_intercept=false)
    obj = objective(lr, X, y)
    J   = LogisticLoss() + L2Penalty() + 2L1Penalty()
    @test obj(θ) ≈ J(y, X*θ, θ)
    lr  = LogisticRegression(1.0, 2.0; fit_intercept=true, penalize_intercept=true)
    obj = objective(lr, X, y)
    J   = LogisticLoss() + L2Penalty() + 2L1Penalty()
    @test obj(θ1) ≈ J(y, X1*θ1, θ1)
end

@testset "Objectives" begin
    λ = 0.5
    γ = 0.3
    δ = 0.7

    # L2 / L2
    r = RidgeRegression(λ; fit_intercept=false)
    J = objective(r, X, y)
    @test J(θ) ≈ sum(abs2.(X*θ - y))/2 + λ * sum(abs2.(θ))/2
    r = RidgeRegression(λ; fit_intercept=true, penalize_intercept=true)
    J = objective(r, X, y1)
    @test J(θ1) ≈ sum(abs2.(X1*θ1 - y1))/2 + λ * sum(abs2.(θ1))/2
    r = RidgeRegression(λ; fit_intercept=true, penalize_intercept=false)
    J = objective(r, X, y1)
    @test J(θ1) ≈ sum(abs2.(X1*θ1 - y1))/2 + λ * sum(abs2.(θ1 .* vcat(ones(p), 0)))/2

    # L2 / L2+L1
    r = ElasticNetRegression(λ, γ; fit_intercept=true, penalize_intercept=true)
    J = objective(r, X, y1)
    @test J(θ1) ≈ sum(abs2.(X1*θ1 - y1))/2 +
                    λ * sum(abs2.(θ1))/2 +
                    γ * sum(abs.(θ1))
    r = ElasticNetRegression(λ, γ; fit_intercept=true, penalize_intercept=false)
    J = objective(r, X, y1)
    @test J(θ1) ≈ sum(abs2.(X1*θ1 - y1))/2 +
                    λ * sum(abs2.(θ1 .* vcat(ones(p), 0)))/2 +
                    γ * sum(abs.(θ1 .* vcat(ones(p), 0)))

    # Logistic / L2 + L1
    r = LogisticRegression(λ, γ; fit_intercept=true, penalize_intercept=true)
    J = objective(r, X, y1)
    @test J(θ1) ≈ LogisticLoss()(y1, X1*θ1) +
                    λ * sum(abs2.(θ1))/2 +
                    γ * sum(abs.(θ1))
    r = LogisticRegression(λ, γ; fit_intercept=true, penalize_intercept=false)
    J = objective(r, X, y1)
    @test J(θ1) ≈ LogisticLoss()(y1, X1*θ1) +
                    λ * sum(abs2.(θ1 .* vcat(ones(p), 0)))/2 +
                    γ * sum(abs.(θ1 .* vcat(ones(p), 0)))

    # Robust / L2
    r = HuberRegression(δ, λ, γ; fit_intercept=true, penalize_intercept=true)
    J = objective(r, X, y1)
    @test J(θ1) ≈ RobustLoss(HuberRho(δ))(y1, X1*θ1) +
                    λ * sum(abs2.(θ1))/2 +
                    γ * sum(abs.(θ1))
    r = HuberRegression(δ, λ, γ; fit_intercept=true, penalize_intercept=false)
    J = objective(r, X, y1)
    @test J(θ1) ≈ RobustLoss(HuberRho(δ))(y1, X1*θ1) +
                    λ * sum(abs2.(θ1 .* vcat(ones(p), 0)))/2 +
                    γ * sum(abs.(θ1 .* vcat(ones(p), 0)))
end
