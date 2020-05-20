n, p = 500, 5
((X, y, θ), (X1, y1, θ1)) = generate_continuous(n, p; seed=525)

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
    @test J(θ) ≈ o(y, X*θ, θ)
    θ_newton   = fit(hr, X, y, solver=Newton())
    θ_newtoncg = fit(hr, X, y, solver=NewtonCG())
    θ_lbfgs    = fit(hr, X, y, solver=LBFGS())
    θ_iwls     = fit(hr, X, y, solver=IWLSCG())
    @test isapprox(J(θ),          5.456121, rtol=1e-5)
    @test isapprox(J(θ_newton),   4.87426,  rtol=1e-5)
    @test isapprox(J(θ_newtoncg), 4.87426,  rtol=1e-5)
    @test isapprox(J(θ_lbfgs),    4.87426,  rtol=1e-5)
    @test isapprox(J(θ_iwls),     4.87426,  rtol=1e-5)

    δ = 0.01
    λ = 3.0
    hr = HuberRegression(δ, λ; penalize_intercept=true)
    J = objective(hr, X, y1)
    o = RobustLoss(Huber(δ)) + λ * L2Penalty()
    @test J(θ1) ≈ o(y1, X1*θ1, θ1)
    θ_newton   = fit(hr, X, y1, solver=Newton())
    θ_newtoncg = fit(hr, X, y1, solver=NewtonCG())
    θ_lbfgs    = fit(hr, X, y1, solver=LBFGS())
    θ_iwls     = fit(hr, X, y1, solver=IWLSCG())
    @test isapprox(J(θ1),         7.601925, rtol=1e-5)
    @test isapprox(J(θ_newton),   6.259785, rtol=1e-5)
    @test isapprox(J(θ_newtoncg), 6.259785, rtol=1e-5)
    @test isapprox(J(θ_lbfgs),    6.259785, rtol=1e-5)
    @test isapprox(J(θ_iwls),     6.259785, rtol=1e-5)

    # don't penalize intercept
    δ = 0.01
    λ = 3.0
    hr = HuberRegression(δ, λ)
    J = objective(hr, X, y1)
    θ_newton   = fit(hr, X, y1, solver=Newton())
    θ_newtoncg = fit(hr, X, y1, solver=NewtonCG())
    θ_lbfgs    = fit(hr, X, y1, solver=LBFGS())
    θ_iwls     = fit(hr, X, y1, solver=IWLSCG())
    @test isapprox(J(θ1),         7.536531, rtol=1e-5)
    @test isapprox(J(θ_newton),   6.200183, rtol=1e-5)
    @test isapprox(J(θ_newtoncg), 6.200183, rtol=1e-5)
    @test isapprox(J(θ_lbfgs),    6.200183, rtol=1e-5)
    @test isapprox(J(θ_iwls),     6.200183, rtol=1e-5)
end

# NOTE: small difference obtained with NCG and IWLS, might just be
# that the algorithm stops a bit earlier, negligible relative tol anyway.
@testset "AndrewsReg" begin
    δ = 0.1
    λ = 3.0
    rr = RobustRegression(rho=Andrews(δ), lambda=λ; penalize_intercept=true)
    J = objective(rr, X, y1)
    o = RobustLoss(AndrewsRho(δ)) + λ * L2Penalty()
    @test J(θ1) ≈ o(y1, X1*θ1, θ1)
    θ_newton   = fit(rr, X, y1, solver=Newton())
    θ_newtoncg = fit(rr, X, y1, solver=NewtonCG())
    θ_lbfgs    = fit(rr, X, y1, solver=LBFGS())
    θ_iwls     = fit(rr, X, y1, solver=IWLSCG())
    @test isapprox(J(θ1),         7.359476, rtol=1e-5)
    @test isapprox(J(θ_newton),   0.486388, rtol=1e-5)
    @test isapprox(J(θ_newtoncg), 0.486388, rtol=1e-5)
    @test isapprox(J(θ_lbfgs),    0.486388, rtol=1e-5)
    @test isapprox(J(θ_iwls),     0.486388, rtol=1e-5)
end

@testset "BisquareReg" begin
    δ = 0.1
    λ = 3.0
    rr = RobustRegression(rho=Bisquare(δ), lambda=λ; penalize_intercept=true)
    J = objective(rr, X, y1)
    o = RobustLoss(BisquareRho(δ)) + λ * L2Penalty()
    @test J(θ1) ≈ o(y1, X1*θ1, θ1)
    θ_newton   = fit(rr, X, y1, solver=Newton())
    θ_newtoncg = fit(rr, X, y1, solver=NewtonCG())
    θ_lbfgs    = fit(rr, X, y1, solver=LBFGS())
    θ_iwls     = fit(rr, X, y1, solver=IWLSCG())
    @test isapprox(J(θ1),         7.773512, rtol=1e-5)
    @test isapprox(J(θ_newton),   0.818677, rtol=1e-5)
    @test isapprox(J(θ_newtoncg), 0.818677, rtol=1e-4)
    @test isapprox(J(θ_lbfgs),    0.818677, rtol=1e-4)
    @test isapprox(J(θ_iwls),     0.818677, rtol=1e-4)
end

@testset "LogisticReg" begin
    δ = 1.5
    λ = 1.0
    rr = RobustRegression(rho=Logistic(delta=δ), lambda=λ, penalize_intercept=true)
    J = objective(rr, X, y1)
    o = RobustLoss(LogisticRho(δ)) + λ * L2Penalty()
    @test J(θ1) ≈ o(y1, X1*θ1, θ1)
    θ_newton = fit(rr, X, y1, solver=Newton())
    θ_newtoncg = fit(rr, X, y1, solver=NewtonCG())
    θ_lbfgs = fit(rr, X, y1, solver=LBFGS())
    θ_iwls  = fit(rr, X, y1, solver=IWLSCG())
    @test isapprox(J(θ1),         5.00079, rtol=1e-5)
    @test isapprox(J(θ_newton),   4.98146, rtol=1e-5)
    @test isapprox(J(θ_newtoncg), 4.98146, rtol=1e-5)
    @test isapprox(J(θ_lbfgs),    4.98146, rtol=1e-5)
    @test isapprox(J(θ_iwls),     4.98146, rtol=1e-5)
end

@testset "FairReg" begin
    δ = 0.1
    λ = 3.0
    rr = RobustRegression(rho=Fair(δ), lambda=λ, penalize_intercept=true)
    J = objective(rr, X, y1)
    o = RobustLoss(Fair(δ)) + λ * L2Penalty()
    @test J(θ1) ≈ o(y1, X1*θ1, θ1)
    θ_newton   = fit(rr, X, y1, solver=Newton())
    θ_newtoncg = fit(rr, X, y1, solver=NewtonCG())
    θ_lbfgs    = fit(rr, X, y1, solver=LBFGS())
    θ_iwls     = fit(rr, X, y1, solver=IWLSCG())
    @test isapprox(J(θ1),         8.586636, rtol=1e-5)
    @test isapprox(J(θ_newton),   8.484584, rtol=1e-5)
    @test isapprox(J(θ_newtoncg), 8.484584, rtol=1e-5)
    @test isapprox(J(θ_lbfgs),    8.484584, rtol=1e-5)
    @test isapprox(J(θ_iwls),     8.484584, rtol=1e-5)
end

# NOTE: small difference obtained with NCG and IWLS, might just be
# that the algorithm stops a bit earlier, negligible relative tol anyway.
@testset "TalwarReg" begin
    δ = 0.1
    λ = 3.0
    rr = RobustRegression(rho=Talwar(δ), lambda=λ, penalize_intercept=true)
    J = objective(rr, X, y1)
    o = RobustLoss(Talwar(δ)) + λ * L2Penalty()
    @test J(θ1) ≈ o(y1, X1*θ1, θ1)
    θ_newton   = fit(rr, X, y1, solver=Newton())
    θ_newtoncg = fit(rr, X, y1, solver=NewtonCG())
    θ_lbfgs    = fit(rr, X, y1, solver=LBFGS())
    θ_iwls     = fit(rr, X, y1, solver=IWLSCG())
    @test isapprox(J(θ1),          8.564334, rtol=1e-5)
    @test isapprox(J(θ_newton),    2.43470,  rtol=1e-5)
    @test isapprox(J(θ_newtoncg),  2.43803,  rtol=1e-5)
    @test isapprox(J(θ_lbfgs),     2.43202,  rtol=1e-5)
    @test isapprox(J(θ_iwls),      2.43803,  rtol=1e-5)
end

###########################
## With Sparsity penalty ##
###########################

n, p = 500, 100
((X, y, θ), (X1, y1, θ1)) = generate_continuous(n, p;  seed=51112, sparse=0.1)
# pepper with outliers
y1a  = outlify(y1, 0.1)

@testset "Robust+L1" begin
    δ  = 0.1
    λ  = 5.0
    γ  = 10.0

    # this is a nice case where the robust is actually smooth
    rr = RobustRegression(rho=Huber(δ), lambda=λ, gamma=γ, penalize_intercept=true)
    J  = objective(rr, X, y1a)
    θ_ls    = X1 \ y1a
    θ_fista = fit(rr, X, y1a, solver=FISTA())
    θ_ista  = fit(rr, X, y1a, solver=ISTA())
    @test isapprox(J(θ_ls),    453.12684, rtol=1e-5)
    @test isapprox(J(θ_fista), 124.20330, rtol=1e-5) # <- ref value
    @test isapprox(J(θ_ista),  124.20330, rtol=1e-5) # ista stops a bit early?
    @test nnz(θ_ls)    == 101
    @test nnz(θ_fista) == 4
    @test nnz(θ_ista)  == 4
end
