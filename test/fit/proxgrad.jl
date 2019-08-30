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
    lr1 = LassoRegression(λ)
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

    if DO_COMPARISONS
        lr_sk = SKLEARN_LM.Lasso(alpha=λ/n)
        lr_sk.fit(X, y1)
        θ1_sk = vcat(lr_sk.coef_[:], lr_sk.intercept_)
        @test isapprox(J(θ1_sk), 312.93487, rtol=1e-3)
        @test nnz(θ1_sk) == 10
    end
end

@testset "elnet" begin
    ρ = 0.3
    α = 0.1
    λ = α * (1 - ρ) * n
    γ = α * ρ * n
    enr = ElasticNetRegression(λ, γ)
    J   = objective(enr, X, y1)
    θ_ls    = X_ \ y1
    θ_fista = fit(enr, X, y1)
    θ_ista  = fit(enr, X, y1, solver=ISTA())
    @test isapprox(J(θ_ls),    220.38693, rtol=1e-5)
    @test isapprox(J(θ_fista), 199.79860, rtol=1e-5) # <- ref values
    @test isapprox(J(θ_ista),  199.79860, rtol=1e-5)
    # sparsity
    @test nnz(θ_ls)    == 101 # not sparse
    @test nnz(θ_fista) == 10   # sparse
    @test nnz(θ_ista)  == 10

    if DO_COMPARISONS
        enr_sk = SKLEARN_LM.ElasticNet(alpha=α, l1_ratio=ρ)
        enr_sk.fit(X, y1)
        θ_sk = vcat(enr_sk.coef_[:], enr_sk.intercept_)
        @test isapprox(J(θ_sk), 199.79860, rtol=5e-4)
        @test nnz(θ_sk) == 11
    end
end

n, p = 500, 100
(_, (X, y, θ)) = generate_binary(n, p;  seed=52551, sparse=0.1)

@testset "Logreg/EN" begin
    ρ = 0.8
    α = 0.03
    λ = α * (1 - ρ) * n
    γ = α * ρ * n
    enlr = LogisticRegression(λ, γ)
    J    = objective(enlr, X, y)
    θ_fista = fit(enlr, X, y)
    θ_ista  = fit(enlr, X, y, solver=ISTA())
    @test isapprox(J(θ),       271.43962, rtol=1e-5)
    @test isapprox(J(θ_fista), 250.46547, rtol=1e-5) # <- ref values
    @test isapprox(J(θ_ista),  250.46547, rtol=1e-5)
    # sparsity
    @test nnz(θ)       == 7  # generating θ
    @test nnz(θ_fista) == 18
    @test nnz(θ_ista)  == 18

    # pure l1 regularization (unclear how sklearn does the mixing of l1/l2 for logreg)
    enlr = LogisticRegression(γ; penalty=:l1)
    J    = objective(enlr, X, y)
    θ_fista = fit(enlr, X, y)
    @test isapprox(J(θ),       263.76834, rtol=1e-5)
    @test isapprox(J(θ_fista), 246.66020, rtol=1e-5) # <- ref value
    @test nnz(θ_fista) == 18

    if DO_COMPARISONS
        # NOTE: this algorithm is stochastic so can't have hard tests
        PY_RND.seed(1531)
        enlr_sk = SKLEARN_LM.LogisticRegression(penalty="elasticnet", C=1.0/γ, l1_ratio=1, solver="saga")
        enlr_sk.fit(X, y)
        θ_sk = vcat(enlr_sk.coef_[:], enlr_sk.intercept_)
        @test isapprox(J(θ_sk), 246.66020, rtol=1e-2)
        @test 17 ≤ nnz(θ_sk) ≤ 19
    end
end

n, p, c = 1000, 100, 3
(_, (X, y, θ)) = generate_multiclass(n, p, c;  seed=33, sparse=0.1)

@testset "Multin/EN" begin
    λ = 10
    γ = 50
    enmnr = MultinomialRegression(λ, γ)
    J     = objective(enmnr, X, y; c=c)
    θ_fista = fit(enmnr, X, y)
    θ_ista  = fit(enmnr, X, y)
    @test isapprox(J(θ),       1574.61741, rtol=1e-5)
    @test isapprox(J(θ_fista),  922.36776, rtol=1e-5) # <- ref value
    @test isapprox(J(θ_ista),   922.36776, rtol=1e-5)
    @test nnz(θ_fista) == 15
    @test nnz(θ_ista)  == 15

    # pure l1 regularization for sklearn comp
    enmnr = MultinomialRegression(γ; penalty=:l1)
    J     = objective(enmnr, X, y; c=c)
    θ_fista = fit(enmnr, X, y)
    @test isapprox(J(θ),       1455.77040, rtol=1e-5)
    @test isapprox(J(θ_fista),  912.37720, rtol=1e-5) # <- ref value
    @test nnz(θ_fista) == 15

    if DO_COMPARISONS
        # NOTE: this algorithm is stochastic
        PY_RND.seed(1531)
        enmnr_sk = SKLEARN_LM.LogisticRegression(penalty="elasticnet", C=1.0/γ, l1_ratio=1,
                                            solver="saga", multi_class="multinomial")
        enmnr_sk.fit(X, y)
        θ_sk = enmnr_sk.coef_'[:]
        @test isapprox(J(θ_sk), 912.37720, rtol=1e-2)
        @test 14 ≤ nnz(θ_sk) ≤ 16
    end
end
