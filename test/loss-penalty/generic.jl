Random.seed!(134)
n = 7
p = 5
x = randn(n)
y = randn(n)
θ = randn(p)

δ  = x .- y
δ1 = norm(δ, 1)
δ2 = norm(δ, 2)^2 / 2
δ3 = norm(δ, 3)^3 / 3
θ1 = norm(θ, 1)
θ2 = norm(θ, 2)^2 / 2
θ3 = norm(θ, 3)^3 / 3
y2 = (sign.(randn(n)) .+ 1) ./ 2


@testset "LPCost basics" begin
    noloss = NoLoss()
    nopenalty = NoPenalty()

    @test noloss(x, y) == 0
    @test nopenalty(θ) == 0

    l1 = L1Loss()
    l2 = L2Loss()
    l3 = LPLoss{3}()
    p1 = L1Penalty()
    p2 = L2Penalty()
    p3 = LPPenalty{3}()

    @test l1(x, y) ≈ δ1
    @test l2(x, y) ≈ δ2
    @test l3(x, y) ≈ δ3
    @test p1(θ) ≈ θ1
    @test p2(θ) ≈ θ2
    @test p3(θ) ≈ θ3

    @test R.getp(p3) == 3
    @test R.lp(θ, 2) ≈ θ2
    @test_throws DomainError R.lp(θ, -1)
end


@testset "Comp LPs" begin
    l1 = L1Loss()
    l2 = L2Loss()
    p1 = L1Penalty()
    p2 = L2Penalty()

    lc = l2 - l1 + NoLoss()
    pc = p1 + p2 - NoPenalty()

    @test (2l2)(x, y) ≈ 2δ2
    @test (l2+l1)(x, y) ≈ δ2 + δ1
    @test (l2+2l1)(x, y) ≈ δ2 + 2δ1
    @test (l2-l1)(x, y) ≈ δ2 - δ1
    @test (l2+2l2-l1)(x, y) ≈ 3δ2-δ1
    @test (lc + lc)(x, y) ≈ 2(δ2 - δ1)
    @test (l2 * 2)(x, y) ≈ 2δ2
    @test 2 * (2 * l2)(x, y) ≈ 4δ2
    @test ((l1 + 2l2) + 2l1)(x, y) ≈ 3δ1 + 2δ2
    @test (l1/3)(x, y) ≈ δ1/3
    @test ((2l1)/3)(x, y) ≈ 2δ1/3
    @test ((l1+l2)/3)(x, y) ≈ (δ1+δ2)/3
    @test ((2l1)+l2)(x, y) ≈ 2δ1+δ2
    @test ((2l1)-3(l1+l2))(x, y) ≈ -δ1 - 3δ2
    @test ((2l1 + 3l2) - (l1 + 2l2))(x, y) ≈ δ1 + δ2

    @test (2p1)(θ) ≈ 2θ1
    @test (p1+p2)(θ) ≈ θ1 + θ2
    @test (p1+2p2)(θ) ≈ θ1 + 2θ2
    @test (p2-p1+3p2)(θ) ≈ θ2 - θ1 + 3θ2
    @test (pc + pc)(θ) ≈ 2(θ1 + θ2)
    @test (p1 * 2)(θ) ≈ 2θ1
    @test 2 * (p1 * 3)(θ) ≈ 6θ1
    @test ((6p1 + 3p2) - 5p1)(θ) ≈ θ1 + 3θ2
    @test (p1/3)(θ) ≈ θ1/3
    @test ((2p1)/4)(θ) ≈ θ1/2
    @test ((2p1 - 5p2)/3)(θ) ≈ (2/3)θ1 - (5/3)θ2
    @test ((2p1) + (p1 + 2p2))(θ) ≈ 3θ1 + 2θ2
    @test ((2p1) - (p1 + p2))(θ) ≈ θ1 - θ2
    @test ((p1 * 2) + 2(p1 - p2))(θ) ≈ 4θ1 - 2θ2
    @test ((4p1 + p2) - (3p2 + p1))(θ) ≈ 3θ1 - 2θ2
end


@testset "Compose" begin
    nl = NoLoss()
    np = NoPenalty()

    @test (nl * 5) isa NoLoss
    @test (nl * 5)(x, y) == 0.0 == (5 * nl)(x, y)
    @test (np * pi) isa NoPenalty
    @test (np * pi)(θ) == 0.0 == (pi * np)(θ)

    @test (np + L1Penalty())(θ) ≈ (L1Penalty() + np)(θ) ≈ θ1

    sl = 2.0 * L1Penalty()
    @test (3sl)(θ) ≈ 6θ1

    @test (nl + L1Loss())(x, y) ≈ (L1Loss() + nl)(x, y) ≈ δ1

    a = 0.5 * L1Loss() + L2Loss() + L1Loss() + 0.3 * L2Loss()
    @test isa(a, CompositeLoss)
    @test length(a.losses) == 2
    @test isa(a.losses[1], ScaledLoss{L2Loss})
    @test a.losses[1].scale == 1.3
    @test isa(a.losses[2], ScaledLoss{L1Loss})
    @test a.losses[2].scale == 1.5

    b = 0.5 * L1Penalty() + L2Penalty() + L1Penalty() + 0.3 * L2Penalty()
    @test isa(b, CompositePenalty)
    @test length(b.penalties) == 2
    @test isa(b.penalties[1], ScaledPenalty{L2Penalty})
    @test b.penalties[1].scale == 1.3
    @test isa(b.penalties[2], ScaledPenalty{L1Penalty})
    @test b.penalties[2].scale == 1.5
end


@testset "Logistic" begin
    ll = LogisticLoss()

    @test sum(log.(1 .+ exp.(-(x .* y)))) ≈ ll(x, y)
end


@testset "Multinomial" begin
    # comparison with sklearn
    θ = [-0.04843, 0.99519, -0.67237, 1.08812, 0.13362, 0.77136]
    X = [ 0.78843 -0.28336;
         -0.75568  0.22546;
         -0.09012  0.68069;
         -0.34437 -0.98773;
          1.09285 -0.37161 ]
    y = [1, 2, 3, 1, 3]
    mnl = MultinomialLoss()
    P = X * reshape(θ, 2, 3)
    @test mnl(P, y) ≈ 4.994135263476211
    @test mnl(y, P) == mnl(P, y)
end
