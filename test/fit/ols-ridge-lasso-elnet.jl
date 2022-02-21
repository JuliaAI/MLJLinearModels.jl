n, p = 100, 5
((X, y, θ), (X1, y1, θ1)) = generate_continuous(n, p; seed=52)

@testset "linreg" begin
    lr  = LinearRegression(fit_intercept=false)
    lr1 = LinearRegression()
    β_ref = X \ y
    @test β_ref == fit(lr, X, y)

    # fit_intercept
    β_ref = X1 \ y1
    @test β_ref == fit(lr1, X, y1)

    # == iterative solvers
    β_cg = fit(lr1, X, y1; solver=CG())
    @test norm(β_cg - β_ref) / norm(β_ref) ≤ 1e-12
end

@testset "ridgereg" begin
    λ   = 1.0
    rr  = RidgeRegression(lambda=λ, fit_intercept=false,
                          scale_penalty_with_samples = false)
    rr1 = RidgeRegression(λ, penalize_intercept=true,
                             scale_penalty_with_samples = false)
    rr2 = RidgeRegression(λ, fit_intercept = true,
                          penalize_intercept = false,
                          scale_penalty_with_samples = false)

    β_ref  = (X'X + λ*I) \ (X'y)
    β_ref1 = (X1'X1 + λ*I) \ (X1'y1)
    β_ref2 = (X1'X1 + diagm(push!(fill(λ, p), 0))) \ (X1'y1)
    @test β_ref ≈ fit(rr, X, y)
    @test β_ref1 ≈ fit(rr1, X, y1)
    @test β_ref2 ≈ fit(rr2, X, y1)

    β_cg  = fit(rr, X, y; solver=CG())
    β_cg1 = fit(rr1, X, y1; solver=CG())

    @test norm(β_cg - β_ref) / norm(β_ref)   ≤ 1e-12
    @test norm(β_cg1 - β_ref1) / norm(β_ref) ≤ 1e-12
end

n, p = 500, 100
((X, y, θ), (X1, y1, θ1)) = generate_continuous(n, p;  seed=512, sparse=0.1)

@testset "lasso" begin
    # no intercept
    λ  = 50
    lr = LassoRegression(λ; fit_intercept=false,
                            scale_penalty_with_samples = false)
    J  = objective(lr, X, y)
    θ_ls    = X \ y
    θ_ista  = fit(lr, X, y, solver=ISTA())
    θ_fista = fit(lr, X, y)

    @test isapprox(J(θ_ls),    586.25012, rtol=1e-5)
    @test isapprox(J(θ_fista), 539.11397, rtol=1e-5) # <- ref value
    @test isapprox(J(θ_ista),  539.11397, rtol=1e-5)
    # sparsity
    @test nnz(θ_ls)    == 100  # not sparse
    @test nnz(θ_fista) == 12 # sparse
    @test nnz(θ_ista)  == 12

    # with intercept
    lr1 = LassoRegression(λ, penalize_intercept=true,
                             scale_penalty_with_samples = false)
    J   = objective(lr1, X, y1)
    θ_ls    = X1 \ y1
    θ_fista = fit(lr1, X, y1)
    θ_ista  = fit(lr1, X, y1, solver=ISTA())
    @test isapprox(J(θ_ls),    214.35485, rtol=1e-5)
    @test isapprox(J(θ_fista), 178.56434, rtol=1e-5)
    @test isapprox(J(θ_ista),  178.56433, rtol=1e-5)
    # sparsity
    @test nnz(θ_ls)    == 101 # not sparse
    @test nnz(θ_fista) == 5   # sparse
    @test nnz(θ_ista)  == 5

    # with intercept and not penalizing intercept
    lr1 = LassoRegression(λ, scale_penalty_with_samples = false)
    J   = objective(lr1, X, y1)
    θ_ls    = X1 \ y1
    θ_fista = fit(lr1, X, y1)
    θ_ista  = fit(lr1, X, y1, solver=ISTA())
    @test isapprox(J(θ_ls),    214.31393, rtol=1e-5)
    @test isapprox(J(θ_fista), 178.54691, rtol=1e-5)
    @test isapprox(J(θ_ista),  178.54690, rtol=1e-5)
    # sparsity
    @test nnz(θ_ls)    == 101 # not sparse
    @test nnz(θ_fista) == 6   # sparse
    @test nnz(θ_ista)  == 6

    if DO_COMPARISONS
        lr_sk = SKLEARN_LM.Lasso(alpha=λ/n, random_state=156123)
        lr_sk.fit(X, y1)
        θ1_sk = vcat(lr_sk.coef_[:], lr_sk.intercept_)
        @test isapprox(J(θ1_sk), 178.5, rtol=1e-3)
        @test nnz(θ1_sk) == 6
    end
end

@testset "elnet" begin
    ρ = 0.3
    α = 0.1
    λ = α * (1 - ρ) * n
    γ = α * ρ * n
    enr = ElasticNetRegression(λ, γ; penalize_intercept=true,
                                     scale_penalty_with_samples = false)
    J   = objective(enr, X, y1)
    θ_ls    = X1 \ y1
    θ_fista = fit(enr, X, y1)
    θ_ista  = fit(enr, X, y1, solver=ISTA())
    @test isapprox(J(θ_ls),    140.07105, rtol=1e-5)
    @test isapprox(J(θ_fista), 123.91270, rtol=1e-5)
    @test isapprox(J(θ_ista),  123.91268, rtol=1e-5)
    # sparsity
    @test nnz(θ_ls)    == 101 # not sparse
    @test nnz(θ_fista) == 7   # sparse
    @test nnz(θ_ista)  == 7

    # not penalizing intercept
    enr = ElasticNetRegression(λ, γ, scale_penalty_with_samples = false)
    J   = objective(enr, X, y1)
    θ_ls    = X1 \ y1
    θ_fista = fit(enr, X, y1)
    θ_ista  = fit(enr, X, y1, solver=ISTA())
    @test isapprox(J(θ_ls),    140.05876, rtol=1e-5)
    @test isapprox(J(θ_fista), 123.89384, rtol=1e-5)
    @test isapprox(J(θ_ista),  123.89383, rtol=1e-5)
    # sparsity
    @test nnz(θ_ls)    == 101 # not sparse
    @test nnz(θ_fista) == 8   # sparse
    @test nnz(θ_ista)  == 8

    if DO_COMPARISONS
        enr_sk = SKLEARN_LM.ElasticNet(alpha=α, l1_ratio=ρ, random_state=5515)
        enr_sk.fit(X, y1)
        θ_sk = vcat(enr_sk.coef_[:], enr_sk.intercept_)
        @test isapprox(J(θ_sk), 123.89, rtol=1e-3)
        @test nnz(θ_sk) == 8
    end
end
