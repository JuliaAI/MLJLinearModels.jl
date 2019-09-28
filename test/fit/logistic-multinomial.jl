n, p = 500, 5
((X, y, θ), (X_, y1, θ1)) = generate_binary(n, p; seed=52)

@testset "Logreg" begin
    # No intercept
    λ = 5.0
    lr = LogisticRegression(λ; fit_intercept=false)
    J  = objective(lr, X, y)
    o  = LogisticLoss() + λ * L2Penalty()
    @test J(θ) == o(y, X*θ, θ)
    θ_newton   = fit(lr, X, y, solver=Newton())
    θ_newtoncg = fit(lr, X, y, solver=NewtonCG())
    θ_lbfgs    = fit(lr, X, y, solver=LBFGS())
    @test isapprox(J(θ),          282.09960, rtol=1e-5)
    @test isapprox(J(θ_newton),   280.37472, rtol=1e-5) # <- ref value
    @test isapprox(J(θ_newtoncg), 280.37472, rtol=1e-5)
    @test isapprox(J(θ_lbfgs),    280.37472, rtol=1e-5)

    # With intercept
    lr1 = LogisticRegression(λ; penalize_intercept=true)
    J   = objective(lr1, X, y1)
    θ_newton   = fit(lr1, X, y1, solver=Newton())
    θ_newtoncg = fit(lr1, X, y1, solver=NewtonCG())
    θ_lbfgs    = fit(lr1, X, y1, solver=NewtonCG())
    @test isapprox(J(θ1),         213.80970, rtol=1e-5)
    @test isapprox(J(θ_newton),   209.32539, rtol=1e-5) # <- ref value
    @test isapprox(J(θ_newtoncg), 209.32539, rtol=1e-5)
    @test isapprox(J(θ_lbfgs),    209.32539, rtol=1e-5)

    # with intercept and not penalizing it
    lr1 = LogisticRegression(λ)
    J   = objective(lr1, X, y1)
    θ_newton   = fit(lr1, X, y1, solver=Newton())
    θ_newtoncg = fit(lr1, X, y1, solver=NewtonCG())
    θ_lbfgs    = fit(lr1, X, y1, solver=NewtonCG())
    @test isapprox(J(θ1),         212.29192, rtol=1e-5)
    @test isapprox(J(θ_newton),   207.71595, rtol=1e-5) # <- ref value
    @test isapprox(J(θ_newtoncg), 207.71595, rtol=1e-5)
    @test isapprox(J(θ_lbfgs),    207.71595, rtol=1e-5)

    if DO_COMPARISONS
        # This checks that the parameters recovered using Sklearn lead
        # to a similar loss than the one given by our code to verify the
        # correctness of the code.
        lr_sk_ncg = SKLEARN_LM.LogisticRegression(C=1.0/λ, solver="newton-cg")
        lr_sk_ncg.fit(X, y1)
        θ1_sk_ncg = vcat(lr_sk_ncg.coef_[:], lr_sk_ncg.intercept_)
        lr_sk_lbfgs = SKLEARN_LM.LogisticRegression(C=1.0/λ, solver="lbfgs")
        lr_sk_lbfgs.fit(X, y1)
        θ1_sk_lbfgs = vcat(lr_sk_lbfgs.coef_[:], lr_sk_lbfgs.intercept_)
        # Comparing with ours
        @test isapprox(J(θ1_sk_ncg),   207.71595, rtol=1e-5)
        @test isapprox(J(θ1_sk_lbfgs), 207.71595, rtol=1e-5)
    end
end

n, p, c = 500, 5, 4
((X, y, θ), (X_, y1, θ1)) = generate_multiclass(n, p, c; seed=525)

@testset "Multinomial" begin
    # No intercept
    λ = 5.0
    mnr = MultinomialRegression(λ; fit_intercept=false)
    J   = objective(mnr, X, y; c=c)
    θ_newtoncg = fit(mnr, X, y, solver=NewtonCG())
    θ_lbfgs    = fit(mnr, X, y, solver=R.LBFGS())
    @test isapprox(J(θ),          419.67795, rtol=1e-5)
    @test isapprox(J(θ_newtoncg), 384.33810, rtol=1e-5) # <- ref value
    @test isapprox(J(θ_lbfgs),    384.33810, rtol=1e-5)

    #  With intercept
    mnr = MultinomialRegression(λ; penalize_intercept=true)
    J   = objective(mnr, X, y1; c=c)
    θ_newtoncg = fit(mnr, X, y1, solver=NewtonCG())
    θ_lbfgs    = fit(mnr, X, y1, solver=R.LBFGS())
    @test isapprox(J(θ),          1244.46404, rtol=1e-5)
    @test isapprox(J(θ_newtoncg),  388.75018, rtol=1e-5) # <- ref value
    @test isapprox(J(θ_lbfgs),     388.75018, rtol=1e-5)

    mnr = MultinomialRegression(λ)
    J   = objective(mnr, X, y1; c=c)
    θ_newtoncg = fit(mnr, X, y1, solver=NewtonCG())
    θ_lbfgs    = fit(mnr, X, y1, solver=R.LBFGS())
    @test isapprox(J(θ),          1242.70990, rtol=1e-5)
    @test isapprox(J(θ_newtoncg),  383.93018, rtol=1e-5) # <- ref value
    @test isapprox(J(θ_lbfgs),     383.93018, rtol=1e-5)

    if DO_COMPARISONS
        lr_sk_ncg = SKLEARN_LM.LogisticRegression(C=1.0/λ, solver="newton-cg",
                                             multi_class="multinomial")
        lr_sk_ncg.fit(X, y1)
        θ1_sk_ncg = vec(vcat(lr_sk_ncg.coef_', lr_sk_ncg.intercept_'))
        lr_sk_lbfgs = SKLEARN_LM.LogisticRegression(C=1.0/λ, solver="lbfgs",
                                               multi_class="multinomial")
        lr_sk_lbfgs.fit(X, y1)
        θ1_sk_lbfgs = vec(vcat(lr_sk_ncg.coef_', lr_sk_ncg.intercept_'))
        # Comparing with ours
        @test isapprox(J(θ1_sk_ncg),   385.67895, rtol=1e-5)
        @test isapprox(J(θ1_sk_lbfgs), 385.67895, rtol=1e-5)
    end
end

n, p = 500, 100
(_, (X, y, θ)) = generate_binary(n, p;  seed=52551, sparse=0.1)

@testset "Logreg/EN" begin
    ρ = 0.8
    α = 0.03
    λ = α * (1 - ρ) * n
    γ = α * ρ * n
    enlr = LogisticRegression(λ, γ; penalize_intercept=true)
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
    @test isapprox(J(θ_fista), 245.65841, rtol=1e-5) # <- ref value
    @test nnz(θ_fista) == 17

    if DO_COMPARISONS
        # NOTE: this algorithm is stochastic so can't have hard tests
        PY_RND.seed(1531)
        enlr_sk = SKLEARN_LM.LogisticRegression(penalty="elasticnet", C=1.0/γ, l1_ratio=1, solver="saga")
        enlr_sk.fit(X, y)
        θ_sk = vcat(enlr_sk.coef_[:], enlr_sk.intercept_)
        @test isapprox(J(θ_sk), 245.658, rtol=1e-3)
        @test 17 ≤ nnz(θ_sk) ≤ 19
    end
end

n, p, c = 1000, 100, 3
(_, (X, y, θ)) = generate_multiclass(n, p, c;  seed=33, sparse=0.1)

@testset "Multin/EN" begin
    λ = 10
    γ = 50
    enmnr = MultinomialRegression(λ, γ; penalize_intercept=true)
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
    @test isapprox(J(θ),       1432.37281, rtol=1e-5)
    @test isapprox(J(θ_fista),  912.25297, rtol=1e-5) # <- ref value
    @test nnz(θ_fista) == 16

    if DO_COMPARISONS
        # NOTE: this algorithm is stochastic
        PY_RND.seed(1531)
        enmnr_sk = SKLEARN_LM.LogisticRegression(penalty="elasticnet", C=1.0/γ, l1_ratio=1,
                                            solver="saga", multi_class="multinomial")
        enmnr_sk.fit(X, y)
        θ_sk = enmnr_sk.coef_'[:]
        @test isapprox(J(θ_sk), 912.39215, rtol=1e-3)
        @test 14 ≤ nnz(θ_sk) ≤ 16
    end
end
