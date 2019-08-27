Random.seed!(1234)
n, p = 50, 5
X = randn(n, p)
X_ = R.augment_X(X, true)
θ  = randn(p)
θ1 = randn(p+1)
y = rand(n)

@testset "Tools" begin
    lr  = LogisticRegression(1.0, 2.0; fit_intercept=false)
    obj = objective(lr, X, y)
    J   = LogisticLoss() + L2Penalty() + 2L1Penalty()
    @test obj(θ) ≈ J(y, X*θ, θ)
    lr  = LogisticRegression(1.0, 2.0; fit_intercept=true)
    obj = objective(lr, X, y)
    J   = LogisticLoss() + L2Penalty() + 2L1Penalty()
    @test obj(θ1) ≈ J(y, X_*θ1, θ1)
end
