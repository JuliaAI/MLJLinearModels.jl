n, p = 100, 5
((X, y, θ), (X_, y1, θ1)) = generate_continuous(n, p; seed=52)

@testset "linreg" begin
    lr  = LinearRegression(fit_intercept=false)
    lr1 = LinearRegression()
    β_ref = X \ y
    @test β_ref == fit(lr, X, y)

    # fit_intercept
    β_ref = X_ \ y1
    @test β_ref == fit(lr1, X, y1)

    # == iterative solvers
    β_cg = fit(lr1, X, y1; solver=CG())
    @test norm(β_cg - β_ref) / norm(β_ref) ≤ 1e-12
end

@testset "ridgereg" begin
    λ   = 1.0
    rr  = RidgeRegression(lambda=λ, fit_intercept=false)
    rr1 = RidgeRegression(λ, penalize_intercept=true)

    β_ref  = (X'X + λ*I) \ (X'y)
    β_ref1 = (X_'X_ + λ*I) \ (X_'y1)
    @test β_ref ≈ fit(rr, X, y)
    @test β_ref1 ≈ fit(rr1, X, y1)

    β_cg  = fit(rr, X, y; solver=CG())
    β_cg1 = fit(rr1, X, y1; solver=CG())

    @test norm(β_cg - β_ref) / norm(β_ref)   ≤ 1e-12
    @test norm(β_cg1 - β_ref1) / norm(β_ref) ≤ 1e-12
end

n, p = 500, 100
((X, y, θ), (X_, y1, θ1)) = generate_continuous(n, p;  seed=512, sparse=0.1)

@testset "lasso" begin
    # no intercept
    λ  = 50
    lr = LassoRegression(λ; fit_intercept=false)
    J  = objective(lr, X, y)
    θ_ls    = X \ y
    θ_ista  = fit(lr, X, y, solver=ISTA())
    θ_fista = fit(lr, X, y)
    @test isapprox(J(θ_ls),    680.26525, rtol=1e-5)
    @test isapprox(J(θ_fista), 631.58622, rtol=1e-5) # <- ref value
    @test isapprox(J(θ_ista),  631.58622, rtol=1e-5)
    # sparsity
    @test nnz(θ_ls)    == 100  # not sparse
    @test nnz(θ_fista) == 12 # sparse
    @test nnz(θ_ista)  == 12

    # with intercept
    lr1 = LassoRegression(λ, penalize_intercept=true)
    J   = objective(lr1, X, y1)
    θ_ls    = X_ \ y1
    θ_fista = fit(lr1, X, y1)
    θ_ista  = fit(lr1, X, y1, solver=ISTA())
    @test isapprox(J(θ_ls),    353.85709, rtol=1e-5)
    @test isapprox(J(θ_fista), 312.93487, rtol=1e-5) # <- ref values
    @test isapprox(J(θ_ista),  312.93487, rtol=1e-5)
    # sparsity
    @test nnz(θ_ls)    == 101 # not sparse
    @test nnz(θ_fista) == 9   # sparse
    @test nnz(θ_ista)  == 9

    # with intercept and not penalizing intercept
    lr1 = LassoRegression(λ)
    J   = objective(lr1, X, y1)
    θ_ls    = X_ \ y1
    θ_fista = fit(lr1, X, y1)
    θ_ista  = fit(lr1, X, y1, solver=ISTA())
    @test isapprox(J(θ_ls),    353.71225, rtol=1e-5)
    @test isapprox(J(θ_fista), 312.93073, rtol=1e-5) # <- ref values
    @test isapprox(J(θ_ista),  312.93073, rtol=1e-5)
    # sparsity
    @test nnz(θ_ls)    == 101 # not sparse
    @test nnz(θ_fista) == 10  # sparse
    @test nnz(θ_ista)  == 10

    if DO_COMPARISONS
        lr_sk = SKLEARN_LM.Lasso(alpha=λ/n)
        lr_sk.fit(X, y1)
        θ1_sk = vcat(lr_sk.coef_[:], lr_sk.intercept_)
        @test isapprox(J(θ1_sk), 312.93072, rtol=1e-5)
        @test nnz(θ1_sk) == 10
    end
end

@testset "elnet" begin
    ρ = 0.3
    α = 0.1
    λ = α * (1 - ρ) * n
    γ = α * ρ * n
    enr = ElasticNetRegression(λ, γ; penalize_intercept=true)
    J   = objective(enr, X, y1)
    θ_ls    = X_ \ y1
    θ_fista = fit(enr, X, y1)
    θ_ista  = fit(enr, X, y1, solver=ISTA())
    @test isapprox(J(θ_ls),    220.38693, rtol=1e-5)
    @test isapprox(J(θ_fista), 199.79860, rtol=1e-5)
    @test isapprox(J(θ_ista),  199.79860, rtol=1e-5)
    # sparsity
    @test nnz(θ_ls)    == 101 # not sparse
    @test nnz(θ_fista) == 10   # sparse
    @test nnz(θ_ista)  == 10

    # not penalizing intercept
    enr = ElasticNetRegression(λ, γ)
    J   = objective(enr, X, y1)
    θ_ls    = X_ \ y1
    θ_fista = fit(enr, X, y1)
    θ_ista  = fit(enr, X, y1, solver=ISTA())
    @test isapprox(J(θ_ls),    220.34333, rtol=1e-5)
    @test isapprox(J(θ_fista), 199.78497, rtol=1e-5)
    @test isapprox(J(θ_ista),  199.78497, rtol=1e-5)
    # sparsity
    @test nnz(θ_ls)    == 101 # not sparse
    @test nnz(θ_fista) == 11   # sparse
    @test nnz(θ_ista)  == 11

    if DO_COMPARISONS
        enr_sk = SKLEARN_LM.ElasticNet(alpha=α, l1_ratio=ρ)
        enr_sk.fit(X, y1)
        θ_sk = vcat(enr_sk.coef_[:], enr_sk.intercept_)
        @test isapprox(J(θ_sk), 199.78496, rtol=1e-5)
        @test nnz(θ_sk) == 11
    end
end
