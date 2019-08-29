@testset "Checks" begin
    X = randn(5, 3)
    y = randn(5)
    @test R.check_nrows(X, y) === nothing
    y = randn(4)
    @test_throws DimensionMismatch R.check_nrows(X, y)
    @test R.check_pos(1)
    @test_throws ArgumentError R.check_pos(-1)
end

@testset "Augment" begin
    X = randn(5, 3)
    X_ = R.augment_X(X, false)
    @test X_ === X
    X_ = R.augment_X(X, true)
    @test X_ == hcat(X, ones(5, 1))
end

@testset "Apply" begin
    n, p = 5, 3
    X = randn(n, p)
    X_ = R.augment_X(X, true)
    θ = randn(p)
    θ1 = randn(p+1)
    @test R.apply_X(X, θ) == X*θ
    @test R.apply_X(X, θ1) ≈ X_*θ1
    # multiclass
    c = 3
    θa  = randn(p)
    θb  = randn(p)
    θc  = randn(p)
    θ   = vcat(θa, θb, θc)
    θa1 = randn(p+1)
    θb1 = randn(p+1)
    θc1 = randn(p+1)
    θ1  = vcat(θa1, θb1, θc1)
    @test R.apply_X(X, θ, c) ≈  X * hcat(θa, θb, θc)
    @test R.apply_X(X, θ1, c) ≈ X_ * hcat(θa1, θb1, θc1)

    θ1 = randn(p+1)
    Xθ = zeros(n)
    R.apply_X!(Xθ, X, θ1)
    @test Xθ ≈ X_ * θ1

    z = randn(n)
    Xtz = zeros(p)
    R.apply_Xt!(Xtz, X, z)
    @test Xtz ≈ X'z

    Xtz2 = zeros(p+1)
    R.apply_Xt!(Xtz2, X, z)
    @test Xtz2 ≈ X_'z

    v = view(Xtz2, 1:p)
    R.apply_Xt!(v, X, z)
    @test Xtz2 ≈ X_'z
end

@testset "Sigmoid" begin
    @test R.sigmoid(zero(Float32)) == 0.5f0
    @test R.sigmoid(zero(Float64)) == 0.5
    @test R.logsigmoid(zero(Float32)) == log(0.5f0)
    @test R.logsigmoid(zero(Float64)) == log(0.5)

    @test R.sigmoid(50) == 1.0
    @test R.sigmoid(50f0) == 1.0f0
    @test R.sigmoid(-50) == 0.0
    @test R.sigmoid(-50f0) == 0.0f0

    @test R.logsigmoid(50) == 0.0
    @test R.logsigmoid(50f0) == 0.0f0
    @test R.logsigmoid(-50) == -50.0
    @test R.logsigmoid(-50f0) == -50.0f0

    x = randn()
    @test -R.σ(-x) ≈ (R.σ(x) - 1.0)
end

@testset "Hat matrix" begin
    λ = 3
    X = randn(5, 3)
    X_ = R.augment_X(X, true)
    @test R.form_XtX(X, true, λ) ≈ X_'X_ + λ*I
end

@testset "Soft-thresh" begin
    x = randn(50)
    η = 0.5
    y = R.soft_thresh.(x, η)

    z = copy(x)
    m1 = x .> η
    m2 = abs.(x) .<= η
    m3 = x .< -η

    z[m1] .= x[m1] .- η
    z[m2] .= 0.0
    z[m3] .= x[m3] .+ η

    @test z == y
end

@testset "Clip" begin
    x = randn(10)
    x[[1, 4, 7]] .= 1e-7 .* rand(3)
    tau = 1e-5
    z = R.clip.(x, tau)
    mask = abs.(x) .>= tau
    @test z[mask] == x[mask]
    @test all(zi == tau for zi in z[.!mask])
end
