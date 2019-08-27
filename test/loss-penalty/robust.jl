x = randn(10)
y = randn(10)
r = x .- y

@testset "Robust Loss" begin
    δ = 0.5
    rlδ = RobustLoss(Huber(δ))
    @test rlδ isa RobustLoss{HuberRho{δ}}
    @test rlδ(r) == rlδ(x, y) == sum(ifelse(abs(rᵢ)≤δ, rᵢ^2/2, δ*(abs(rᵢ)-δ/2)) for rᵢ in r)

    rlδ = RobustLoss(Andrews(δ))
    p1(z) = -cos(π * z/δ)/(π/δ)^2
    p2(z) = p1(δ)
    @test rlδ(r) ≈ sum(ifelse(abs(rᵢ)≤δ, p1.(rᵢ), p2.(rᵢ)) for rᵢ in r)

    rlδ = RobustLoss(Bisquare(δ))
    p1(z) = δ^2/6 * (1-(1-(z/δ)^2)^3)
    p2(z) = p1(δ)
    @test rlδ(r) ≈ sum(ifelse(abs(rᵢ)≤δ, p1.(rᵢ), p2.(rᵢ)) for rᵢ in r)

    rlδ = RobustLoss(Logistic(δ))
    @test rlδ(r) ≈ sum(δ^2 * log(cosh(rᵢ/δ)) for rᵢ in r)

    rlδ = RobustLoss(FairRho(δ))
    @test rlδ(r) ≈ sum(δ^2 * (abs(rᵢ)/δ - log(1+abs(rᵢ/δ))) for rᵢ in r)

    rlδ = RobustLoss(Talwar(δ))
    p1(z) = z^2/2
    p2(z) = p1(δ)
    @test rlδ(r) ≈ sum(ifelse(abs(rᵢ)≤δ, p1.(rᵢ), p2.(rᵢ)) for rᵢ in r)
end
