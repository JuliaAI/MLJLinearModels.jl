n, p = 50, 5
((X, y, θ), (X1, y1, θ1)) = generate_continuous(n, p; seed=1212224)
maskint = vcat(ones(p), 0.0)

@testset "GH> Ridge" begin
    rng = StableRNG(551551)
    λ = 0.5
    # without fit_intercept
    s = R.scratch(X; i=false)
    # >> Hv!
    r = RidgeRegression(λ; fit_intercept=false)
    hv! = R.Hv!(r, X, y, s)
    v = randn(rng, length(θ))
    hv = similar(v)
    hv!(hv, v)
    @test hv ≈              X' * (X * v) .+ λ * v
    # ------------------
    # with fit_intercept
    s = R.scratch(X; i=true)
    r = RidgeRegression(λ; penalize_intercept=true)
    hv! = R.Hv!(r, X, y, s)
    v = randn(rng, p+1)
    hv = similar(v)
    hv!(hv, v)
    @test hv ≈              X1' * (X1 * v) .+ λ * v
    # ------------------
    # with fit_intercept but no penalty
    r = RidgeRegression(λ)
    hv! = R.Hv!(r, X, y, s)
    v = randn(rng, p+1)
    hv = similar(v)
    hv!(hv, v)
    @test hv ≈              X1' * (X1 * v) .+ λ * v .* vcat(ones(p), 0.0)
end

@testset "GH> EN/Lasso" begin
    # without fit_intercept
    s = R.scratch(X; i=false)
    λ = 6.2
    γ = 0.7
    r = LassoRegression(λ; fit_intercept=false)
    fg! = R.smooth_fg!(r, X, y, s)
    g = similar(θ)
    f = fg!(g, θ)
    @test f ≈               sum(abs2.(X*θ .- y))/2
    @test g ≈               X' * (X * θ .- y)
    # with fit_intercept
    s = R.scratch(X; i=true)
    λ = 3.4
    γ = 2.7
    r = LassoRegression(λ; penalize_intercept=true)
    fg! = R.smooth_fg!(r, X, y1, s)
    g = similar(θ1)
    f = fg!(g, θ1)
    @test f ≈               sum(abs2.(X1*θ1 .- y1))/2
    @test g ≈               X1' * (X1 * θ1 .- y1)

    # elasticnet (with intercept)
    r = ElasticNetRegression(λ, γ; penalize_intercept=true)
    fg! = R.smooth_fg!(r, X, y1, s)
    g = similar(θ1)
    f = fg!(g, θ1)
    @test f ≈               sum(abs2.(X1*θ1 .- y1))/2 + λ .* norm(θ1)^2/2
    @test g ≈               X1' * (X1*θ1 .- y1) .+ λ .* θ1

    # elasticnet with intercept but no penalty of intercept
    r = ElasticNetRegression(λ, γ)
    fg! = R.smooth_fg!(r, X, y1, s)
    g = similar(θ1)
    f = fg!(g, θ1)
    @test f ≈               sum(abs2.(X1*θ1 .- y1))/2 + λ .* norm(θ1 .* maskint)^2/2
    @test g ≈               X1' * (X1*θ1 .- y1) .+ λ .* θ1 .* vcat(ones(p), 0)
end

n, p = 50, 5
((X, y, θ), (X1, y1, θ1)) = generate_binary(n, p; seed=1212224)
maskint = vcat(ones(p), 0.0)

@testset "GH> LogitL2" begin
    rng = StableRNG(551551)
    # fgh! without fit_intercept
    s = R.scratch(X; i=false)
    λ = 0.5
    lr = LogisticRegression(λ; fit_intercept=false)
    fgh! = R.fgh!(lr, X, y, s)
    θ = randn(rng, p)
    J = objective(lr, X, y)
    f = 0.0
    g = similar(θ)
    H = zeros(p, p)
    f = fgh!(f, g, H, θ)

    wminus = R.σ.(-y .* (X * θ))
    wplus  = R.σ.(y .* (X * θ))

    @test f == J(θ)
    @test g ≈ -X' * (y .* wminus) .+ λ .* θ
    @test H ≈  X' * (Diagonal(wplus .* (1 .- wplus)) * X) + λ * I

    # Use ForwardDiff
    logsigmoid(x)  = -log1p(exp(-x))
    objective2(θ_) = -sum(logsigmoid.(y .* (X*θ_))) + λ * sum(abs2, θ_) / 2
    fd_grad = ForwardDiff.gradient(θ_ -> objective2(θ_), θ)
    fd_hess = ForwardDiff.hessian(θ_ -> objective2(θ_), θ)
    @test g ≈ fd_grad
    @test H ≈ fd_hess

    # Hv! without  fit_intercept
    s = R.scratch(X; i=false)
    Hv! = R.Hv!(lr, X, y, s)
    v   = randn(rng, p)
    Hv  = similar(v)
    Hv!(Hv, θ, v)
    @test Hv ≈ H * v

    # fgh! with fit_intercept
    s = R.scratch(X; i=true)
    λ = 0.5
    lr1 = LogisticRegression(λ; penalize_intercept=true)
    fgh! = R.fgh!(lr1, X, y, s)
    θ1 = randn(rng, p+1)
    J  = objective(lr1, X, y)
    f1 = 0.0
    g1 = similar(θ1)
    H1 = zeros(p+1, p+1)
    f1 = fgh!(f1, g1, H1, θ1)

    wminus = R.σ.(-y .* (X1 * θ1))
    wplus  = R.σ.(y .* (X1 * θ1))

    @test f1 ≈ J(θ1)
    @test g1 ≈ -X1' * (y .* wminus) .+ λ .* θ1
    @test H1 ≈ X1' * (Diagonal(wplus .* (1 .- wplus)) * X1) + λ * I

    # Use ForwardDiff
    logsigmoid(x)  = -log1p(exp(-x))
    objective2(θ_) = -sum(logsigmoid.(y .* (X1*θ_))) + λ * sum(abs2, θ_) / 2
    fd_grad = ForwardDiff.gradient(θ_ -> objective2(θ_), θ1)
    fd_hess = ForwardDiff.hessian(θ_ -> objective2(θ_), θ1)
    @test g1 ≈ fd_grad
    @test H1 ≈ fd_hess

    # Hv! with fit_intercept
    Hv! = R.Hv!(lr1, X, y, s)
    v   = randn(rng, p+1)
    Hv  = similar(v)
    Hv!(Hv, θ1, v)
    @test Hv ≈ H1 * v

    # fgh! with fit intercept and no penalty on intercept
    lr1 = LogisticRegression(λ)
    fgh! = R.fgh!(lr1, X, y, s)
    θ1 = randn(rng, p+1)
    J  = objective(lr1, X, y)
    f1 = 0.0
    g1 = similar(θ1)
    H1 = zeros(p+1, p+1)
    f1 = fgh!(f1, g1, H1, θ1)

    wminus = R.σ.(-y .* (X1 * θ1))
    wplus  = R.σ.(y .* (X1 * θ1))

    @test f1 ≈ J(θ1)
    @test g1 ≈ -X1' * (y .* wminus) .+ λ .* θ1 .* maskint
    @test H1 ≈  X1' * (Diagonal(wplus .* (1.0 .- wplus)) * X1) + λ * Diagonal(maskint)
    Hv! = R.Hv!(lr1, X, y, s)
    v   = randn(rng, p+1)
    Hv  = similar(v)
    Hv!(Hv, θ1, v)
    @test Hv ≈               H1 * v
end

# Comparison with sklearn
@testset "GH> MultinL2" begin
    X = [ 0.78843 -0.28336;
         -0.75568  0.22546;
         -0.09012  0.68069;
         -0.34437 -0.98773;
          1.09285 -0.37161 ]
    n, p = size(X)
    c = 3
    y = [1, 2, 3, 1, 3]
    # comparison sklearn // no intercept
    θ = [-0.04843, 0.99519, -0.67237, 1.08812, 0.13362, 0.77136]
    g_sk = [-0.12941349639677957, 1.033822503077806, 0.6025709048825946, -0.3233237353163467,
            -0.47315740848581506, -0.7104987677614594]
    v = [-0.28802, -0.90018, 1.48613, 1.77976, -1.06333, 1.36275]
    Hv_sk = [0.18500344954627015, -0.9109175305006267, 0.8560420396655112, 0.4034408910070668,
            -1.0410454892117813, 0.5074766394935599]
    s = R.scratch(n, p, c; i=false)
    mnr = MultinomialRegression(0.0; fit_intercept=false)
    fg! = R.fg!(mnr, X, y, s)
    f = fg!(0.0, nothing, θ)
    mnl = MultinomialLoss()
    @test f ≈ mnl(y, X*reshape(θ, 2, 3))
    g = similar(θ)
    fg!(nothing, g, θ)
    @test g ≈ g_sk
    hv! = R.Hv!(mnr, X, y, s)
    Hv = similar(θ)
    hv!(Hv, θ, v)
    @test Hv ≈ Hv_sk

    # -- with intercept
    θ1 = [0.17905, 1.91598, 1.30329, -1.03438, -1.26994, -0.38288, -0.96238, -0.47912, 0.70813]
    g_sk1 = [0.26979349788581053, 1.5036100824512861, 0.7042460191768531, 0.7282930655963799,
            -0.6529621156985143, -0.2504598040370489, -0.9980865634821908, -0.8506479667527724,
            -0.45378621513980455]
    v1 = [-1.05534, -0.12016, 0.28984, 0.81068, 0.1617, -0.82832, -0.60992, 0.74291, -0.14735]
    Hv1_sk = [-0.20815994906492608, -0.03987039776759468, 0.38689109859835424, 0.3516398288195569,
               0.0703034889451111, -0.3054162498173413, -0.14347987975463106, -0.03043309117751647,
               -0.08147484878101276]
    s = R.scratch(n, p, c; i=true)
    mnr1 = MultinomialRegression(0.0; penalize_intercept=true)
    fg! = R.fg!(mnr1, X, y, s)
    f = fg!(0.0, nothing, θ1)
    @test  f ≈ mnl(y, R.apply_X(X, θ1, 3))
    g = similar(θ1)
    fg!(nothing, g, θ1)
    @test g ≈ g_sk1
    hv! = R.Hv!(mnr1, X, y, s)
    Hv1 = similar(θ1)
    hv!(Hv1, θ1, v1)
    @test Hv1 ≈ Hv1_sk
end

@testset "GH> Huber" begin
    rng = StableRNG(12341412)
    δ, λ  = 0.5, 3.4

    # without intercept
    s = R.scratch(X; i=false)
    hlr  = HuberRegression(δ, λ; fit_intercept=false)
    fgh! = R.fgh!(hlr, X, y, s)
    θ_   = randn(rng, p) # otherwise the residuals are too small and everything is in the l1-ball

    g = similar(θ)
    H = zeros(p, p)

    f = fgh!(0.0, g, H, θ_)
    r = X*θ_ .- y
    @test f == hlr.loss(r) + hlr.penalty(θ_)
    mask = abs.(r) .<= δ
    @test g ≈               (X' * (r .* mask)) .+ (X' * (δ .* sign.(r) .* .!mask)) .+ λ .* θ_
    @test H ≈                X' * (mask .* X) + λ*I

    Hv! = R.Hv!(hlr, X, y, s)
    Hv = similar(θ_)
    v = randn(rng, p)
    Hv!(Hv, θ_, v)

    @test Hv ≈               H * v

    # with intercept
    s = R.scratch(X; i=true)
    hlr1  = HuberRegression(δ, λ; penalize_intercept=true)
    fgh1! = R.fgh!(hlr1, X, y1, s)
    θ1_   = randn(rng, p+1)

    g1 = similar(θ1)
    H1 = zeros(p+1, p+1)

    Hv! = R.Hv!(hlr, X, y, s)
    Hv = similar(θ_)
    v = randn(rng, p)
    Hv!(Hv, θ_, v)

    @test Hv ≈ H * v

    # with intercept
    hlr1  = HuberRegression(δ, λ; penalize_intercept=true)
    fgh1! = R.fgh!(hlr1, X, y1, s)
    θ1_   = randn(rng, p+1)

    g1 = similar(θ1)
    H1 = zeros(p+1, p+1)

    f1 = fgh1!(0.0, g1, H1, θ1_)
    r1 = X1*θ1_ .- y1
    mask = abs.(r1) .<= δ
    @test f1 ≈ hlr1.loss(r1) + hlr1.penalty(θ1_)
    @test g1 ≈              (X1' * (r1 .* mask)) .+ (X1' * (δ .* sign.(r1) .* .!mask)) .+ λ .* θ1_
    @test H1 ≈               X1' * (mask .* X1) + λ*I

    Hv1! = R.Hv!(hlr1, X, y1, s)
    Hv1 = similar(θ1_)
    v1 = randn(rng, p+1)
    Hv1!(Hv1, θ1_, v1)

    @test Hv1 ≈              H1 * v1

    # with intercept and no penalty on intercept
    hlr1  = HuberRegression(δ, λ)
    fgh1! = R.fgh!(hlr1, X, y1, s)
    θ1_   = randn(rng, p+1)

    g1 = similar(θ1)
    H1 = zeros(p+1, p+1)

    f1 = fgh1!(0.0, g1, H1, θ1_)
    r1 = X1*θ1_ .- y1
    mask = abs.(r1) .<= δ
    @test f1 ≈ hlr1.loss(r1) + hlr1.penalty(θ1_ .* maskint)
    @test g1 ≈              (X1' * (r1 .* mask)) .+ (X1' * (δ .* sign.(r1) .* .!mask)) .+
                                λ .* θ1_ .* maskint
    @test H1 ≈               X1' * (mask .* X1) + λ * Diagonal(maskint)

    Hv1! = R.Hv!(hlr1, X, y1, s)
    Hv1 = similar(θ1_)
    v1 = randn(rng, p+1)
    Hv1!(Hv1, θ1_, v1)

    @test Hv1 ≈              H1 * v1
end
