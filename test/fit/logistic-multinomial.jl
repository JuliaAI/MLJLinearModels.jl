n, p = 500, 5
((X, y, θ), (X1, y1, θ1)) = generate_binary(n, p; seed=52661)

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
    @test isapprox(J(θ),          191.949109, rtol=1e-5)
    @test isapprox(J(θ_newton),   184.704499, rtol=1e-5)
    @test isapprox(J(θ_newtoncg), 184.704499, rtol=1e-5)
    @test isapprox(J(θ_lbfgs),    184.704499, rtol=1e-5)

    # With intercept
    lr1 = LogisticRegression(λ; penalize_intercept=true)
    J   = objective(lr1, X, y1)
    θ_newton   = fit(lr1, X, y1, solver=Newton())
    θ_newtoncg = fit(lr1, X, y1, solver=NewtonCG())
    θ_lbfgs    = fit(lr1, X, y1, solver=NewtonCG())
    @test isapprox(J(θ1),         236.95486, rtol=1e-5)
    @test isapprox(J(θ_newton),   230.47826, rtol=1e-5)
    @test isapprox(J(θ_newtoncg), 230.47826, rtol=1e-5)
    @test isapprox(J(θ_lbfgs),    230.47826, rtol=1e-5)

    # with intercept and not penalizing it
    lr1 = LogisticRegression(λ)
    J   = objective(lr1, X, y1)
    θ_newton   = fit(lr1, X, y1, solver=Newton())
    θ_newtoncg = fit(lr1, X, y1, solver=NewtonCG())
    θ_lbfgs    = fit(lr1, X, y1, solver=NewtonCG())
    @test isapprox(J(θ1),         231.44778, rtol=1e-5)
    @test isapprox(J(θ_newton),   226.44523, rtol=1e-5)
    @test isapprox(J(θ_newtoncg), 226.44523, rtol=1e-5)
    @test isapprox(J(θ_lbfgs),    226.44523, rtol=1e-5)

    if DO_COMPARISONS
        # This checks that the parameters recovered using Sklearn lead
        # to a similar loss than the one given by our code to verify the
        # correctness of the code.
        lr_sk_ncg = SKLEARN_LM.LogisticRegression(
                        C=1.0/λ, solver="newton-cg", random_state=155151)
        lr_sk_ncg.fit(X, y1)
        θ1_sk_ncg = vcat(lr_sk_ncg.coef_[:], lr_sk_ncg.intercept_)
        lr_sk_lbfgs = SKLEARN_LM.LogisticRegression(
                        C=1.0/λ, solver="lbfgs", random_state=155151)
        lr_sk_lbfgs.fit(X, y1)
        θ1_sk_lbfgs = vcat(lr_sk_lbfgs.coef_[:], lr_sk_lbfgs.intercept_)
        # Comparing with ours
        @test isapprox(J(θ1_sk_ncg),   226.4, rtol=1e-2)
        @test isapprox(J(θ1_sk_lbfgs), 226.4, rtol=1e-2)
    end
end

n, p, c = 500, 5, 4
((X, y, θ), (X1, y1, θ1)) = generate_multiclass(n, p, c; seed=525)

@testset "Multinomial" begin
    # No intercept
    λ = 5.0
    mnr = MultinomialRegression(λ; fit_intercept=false)
    J   = objective(mnr, X, y; c=c)
    θ_newtoncg = fit(mnr, X, y, solver=NewtonCG())
    θ_lbfgs    = fit(mnr, X, y, solver=R.LBFGS())
    @test isapprox(J(θ),          428.25787, rtol=1e-5)
    @test isapprox(J(θ_newtoncg), 408.90295, rtol=1e-5)
    @test isapprox(J(θ_lbfgs),    408.90295, rtol=1e-5)

    #  With intercept
    mnr = MultinomialRegression(λ; penalize_intercept=true)
    J   = objective(mnr, X, y1; c=c)
    θ_newtoncg = fit(mnr, X, y1, solver=NewtonCG())
    θ_lbfgs    = fit(mnr, X, y1, solver=R.LBFGS())
    @test isapprox(J(θ),          1415.66534, rtol=1e-5)
    @test isapprox(J(θ_newtoncg),  375.95659, rtol=1e-5)
    @test isapprox(J(θ_lbfgs),     375.95659, rtol=1e-5)

    mnr = MultinomialRegression(λ)
    J   = objective(mnr, X, y1; c=c)
    θ_newtoncg = fit(mnr, X, y1, solver=NewtonCG())
    θ_lbfgs    = fit(mnr, X, y1, solver=R.LBFGS())
    @test isapprox(J(θ),          1414.01629, rtol=1e-5)
    @test isapprox(J(θ_newtoncg),  372.42948, rtol=1e-5)
    @test isapprox(J(θ_lbfgs),     372.42948, rtol=1e-5)

    if DO_COMPARISONS
        lr_sk_ncg = SKLEARN_LM.LogisticRegression(
                        C=1.0/λ, solver="newton-cg", multi_class="multinomial",
                        random_state=551551)
        lr_sk_ncg.fit(X, y1)
        θ1_sk_ncg = vec(vcat(lr_sk_ncg.coef_', lr_sk_ncg.intercept_'))
        lr_sk_lbfgs = SKLEARN_LM.LogisticRegression(
                        C=1.0/λ, solver="lbfgs", multi_class="multinomial",
                        random_state=551551)
        lr_sk_lbfgs.fit(X, y1)
        θ1_sk_lbfgs = vec(vcat(lr_sk_ncg.coef_', lr_sk_ncg.intercept_'))
        # Comparing with ours
        @test isapprox(J(θ1_sk_ncg),   374.697, rtol=1e-3)
        @test isapprox(J(θ1_sk_lbfgs), 374.697, rtol=1e-3)
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
    @test isapprox(J(θ),       273.58196, rtol=1e-5)
    @test isapprox(J(θ_fista), 226.92630, rtol=1e-5)
    @test isapprox(J(θ_ista),  226.92627, rtol=1e-5)
    # sparsity
    @test nnz(θ)       == 8  # generating θ
    @test nnz(θ_fista) == 15
    @test nnz(θ_ista)  == 15

    # pure l1 regularization (unclear how sklearn does the mixing of l1/l2 for logreg)
    enlr = LogisticRegression(γ; penalty=:l1)
    J    = objective(enlr, X, y)
    θ_fista = fit(enlr, X, y)
    @test isapprox(J(θ),       255.83313, rtol=1e-5)
    @test isapprox(J(θ_fista), 218.79232, rtol=1e-5)
    @test nnz(θ_fista) == 14

    if DO_COMPARISONS
        enlr_sk = SKLEARN_LM.LogisticRegression(
                    penalty="elasticnet", C=1.0/γ, l1_ratio=1, solver="saga",
                    random_state=6618)
        enlr_sk.fit(X, y)
        θ_sk = vcat(enlr_sk.coef_[:], enlr_sk.intercept_)
        @test isapprox(J(θ_sk), 218.79, rtol=1e-3)
        @test nnz(θ_sk) == 14
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
    @test isapprox(J(θ),       1614.90802, rtol=1e-5)
    @test isapprox(J(θ_fista),  931.85109, rtol=1e-5)
    @test isapprox(J(θ_ista),   931.85109, rtol=1e-5)
    @test nnz(θ_fista) == 14
    @test nnz(θ_ista)  == 14

    # pure l1 regularization for sklearn comp
    enmnr = MultinomialRegression(γ; penalty=:l1)
    J     = objective(enmnr, X, y; c=c)
    θ_fista = fit(enmnr, X, y)
    @test isapprox(J(θ),       1492.82898, rtol=1e-5)
    @test isapprox(J(θ_fista),  917.72513, rtol=1e-5)
    @test nnz(θ_fista) == 15

    if DO_COMPARISONS
        enmnr_sk = SKLEARN_LM.LogisticRegression(
                    penalty="elasticnet", C=1.0/γ, l1_ratio=1, solver="saga", multi_class="multinomial", random_state=1616)
        enmnr_sk.fit(X, y)
        θ_sk = enmnr_sk.coef_'[:]
        @test isapprox(J(θ_sk), 921.54, rtol=1e-3)
        @test nnz(θ_sk) == 14
    end
end
