@testset "ridge-reg" begin
    Random.seed!(6161)
    n, p = 100, 5
    X = randn(n, p)
    y = randn(n)
    X1 = R.augment_X(X, true)

    λ = 0.3

    Xt = MLJBase.table(X)
    rr = RidgeRegressor(lambda=λ, penalize_intercept=true)
    fr, = MLJBase.fit(rr, 1, Xt, y)
    ŷ = MLJBase.predict(rr, fr, Xt)

    θ = (X1'X1 + λ*I)\(X1'y)
    coefs = θ[1:end-1]
    intercept = θ[end]

    fp = MLJBase.fitted_params(rr, fr)
    @test fp.coefs ≈ coefs
    @test fp.intercept ≈ intercept
end

@testset "logistic" begin
    ((X, y, θ), (X1, y1, θ1)) = generate_binary(100, 5)

    λ = 0.7
    γ = 0.1

    Xt = MLJBase.table(X)
    yc = MLJBase.categorical(y1)

    lr = LogisticClassifier(lambda=λ, gamma=γ)
    fr, = MLJBase.fit(lr, 1, Xt, yc)

    fp = MLJBase.fitted_params(lr, fr)
    ŷ = MLJBase.predict(lr, fr, Xt)
    ŷ = MLJBase.mode.(ŷ)

    mcr = MLJBase.misclassification_rate(ŷ, yc)
    @test mcr ≤ 0.2
end

@testset "multinomial" begin
    ((X, y, θ), (X1, y1, θ1)) = generate_multiclass(100,  5, 3)

    λ = 0.5
    γ = 0.2

    Xt = MLJBase.table(X)
    yc = MLJBase.categorical(y1)

    mc = MultinomialClassifier(lambda=λ, gamma=γ)
    fr, = MLJBase.fit(mc, 1, Xt, yc)

    fp = MLJBase.fitted_params(mc, fr)
    ŷ = MLJBase.predict(mc, fr, Xt)
    ŷ = MLJBase.mode.(ŷ)

    mcr = MLJBase.misclassification_rate(ŷ, yc)
    @test mcr ≤ 0.2
end
