n, p = 500, 100
((X, y, θ), (X_, y1, θ1)) = generate_continuous(n, p;  seed=512, sparse=0.1)

@testset "lasso" begin
    λ   = 50
    lr  = LassoRegression(λ; fit_intercept=false)
    J   = objective(lr, X, y)

    # no intercept
    θ_ref = X \ y
    @test J(θ_ref)         ≤ 680.27
    @test nnz(θ_ref)      == p      # not sparse
    θ_fista = fit(lr, X, y)
    @test J(θ_fista)       ≤ 631.6
    @test nnz(θ_fista)    == 12      # sparse
    θ_ista = fit(lr, X, y, solver=ISTA())
    @test J(θ_ista)        ≤ 631.6
    @test nnz(θ_ista)     == 12

    # with intercept
    lr1 = LassoRegression(λ)
    J1  = objective(lr1, X, y1)
    θ_ref = X_ \ y1
    @test J1(θ_ref)        ≤ 353.86
    @test nnz(θ_ref)      == p+1
    θ_lasso = fit(lr1, X, y1)
    @test J1(θ_lasso)      ≤ 312.94
    @test nnz(θ_lasso)    == 9
    θ_ista = fit(lr1, X, y1, solver=ISTA())
    @test J1(θ_ista) ≤ 312.94
    @test nnz(θ_ista)     == 9

    if SKLEARN
        lr_sk = SK_LM.Lasso(alpha=λ/n)
        lr_sk.fit(X, y1)
        θ1_sk = vcat(lr_sk.coef_[:], lr_sk.intercept_)
        @test nnz(θ1_sk) == 10
        @test J1(θ1_sk)   ≤ 313.14
    end
end


@testset "elnet" begin
    ρ = 0.3
    α = 0.1
    λ = α * (1 - ρ) * n
    γ = α * ρ * n
    enr = ElasticNetRegression(λ, γ)
    J   = objective(enr, X, y1)
    θ_ref = X_ \ y1
    @test J(θ_ref)      ≤ 220.4
    θ_fista = fit(enr, X, y1)
    @test J(θ_fista)    ≤ 199.8
    @test nnz(θ_fista) == 10
    θ_ista = fit(enr, X, y1, solver=ISTA())
    @test J(θ_ista)     ≤ 199.8
    @test nnz(θ_ista)    == 10

    if SKLEARN
        enr_sk = SK_LM.ElasticNet(alpha=α, l1_ratio=ρ)
        enr_sk.fit(X, y1)
        θ_sk = vcat(enr_sk.coef_[:], enr_sk.intercept_)
        @test nnz(θ_sk) == 11
        @test J(θ_sk)    ≤ 199.9
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
    @test J(θ)        ≤ 271.44
    θ_fista = fit(enlr, X, y)
    @test J(θ_fista)  ≤ 250.47
    @test nnz(θ_fista) == 18
    θ_ista  = fit(enlr, X, y, solver=ISTA())
    @test J(θ_ista)   ≤ 250.47
    @test nnz(θ_ista)  == 18

    # pure l1 regularization (not clear how sklearn does the mixing of l1/l2 for logreg)
    enlr = LogisticRegression(γ; penalty=:l1)
    J    = objective(enlr, X, y)
    @test J(θ)          ≤ 263.77
    θ_fista = fit(enlr, X, y)
    @test J(θ_fista)    ≤ 246.67
    @test nnz(θ_fista) == 18

    if SKLEARN
        # Note: this algorithm is stochastic
        PY_RND.seed(1531)
        enlr_sk = SK_LM.LogisticRegression(penalty="elasticnet", C=1.0/γ, l1_ratio=1, solver="saga")
        enlr_sk.fit(X, y)
        θ_sk = vcat(enlr_sk.coef_[:], enlr_sk.intercept_)
        @test J(θ_sk) ≤ 247.61 # sometimes will do better
        @test nnz(θ_sk) < 20
    end
end

n, p, c = 1000, 100, 3
(_, (X, y, θ)) = generate_multiclass(n, p, c;  seed=33, sparse=0.1)

@testset "Multin/EN" begin
    λ = 10
    γ = 50
    enmnr = MultinomialRegression(λ, γ)
    J     = objective(enmnr, X, y; c=c)
    @test J(θ)          ≤ 1_574.62
    θ_fista = fit(enmnr, X, y)
    @test J(θ_fista)    ≤ 922.37
    @test nnz(θ_fista) == 15
    θ_ista  = fit(enmnr, X, y)
    @test J(θ_ista)     ≤ 922.37
    @test nnz(θ_ista)  == 15

    # pure l1 regularization
    enmnr = MultinomialRegression(γ; penalty=:l1)
    J     = objective(enmnr, X, y; c=c)
    @test J(θ)          ≤ 1_455.771
    θ_fista = fit(enmnr, X, y)
    @test J(θ_fista)    ≤ 912.38
    @test nnz(θ_fista) == 15

    if SKLEARN
        # Note: this algorithm is stochastic
        PY_RND.seed(1531)
        enmnr_sk = SK_LM.LogisticRegression(penalty="elasticnet", C=1.0/γ, l1_ratio=1,
                                            solver="saga", multi_class="multinomial")
        enmnr_sk.fit(X, y)
        θ_sk = enmnr_sk.coef_'[:]
        @test J(θ_sk)    ≤ 912.4
        @test nnz(θ_sk) == 15
    end
end
