n, p = 500, 5
((X, y, θ), (X_, y1, θ1)) = generate_binary(n, p; seed=52)

@testset "Logreg" begin
    # No intercept
    λ = 5.0
    lr = LogisticRegression(λ; fit_intercept=false)
    J  = objective(lr, X, y)
    o  = LogisticLoss() + λ * L2Penalty()
    @test J(θ) == o(y, X*θ, θ)
    @test J(θ)          ≤ 282.1
    θ_newton = fit(lr, X, y, solver=Newton())
    @test J(θ_newton)   ≤ 280.4
    θ_newtoncg = fit(lr, X, y, solver=NewtonCG())
    @test J(θ_newtoncg) ≤ 280.4
    θ_lbfgs = fit(lr, X, y, solver=R.LBFGS())
    @test J(θ_lbfgs)    ≤ 280.4

    # With intercept
    lr1 = LogisticRegression(λ)
    J = objective(lr1, X, y1)
    @test J(θ1)          ≤ 214
    θ1_newton = fit(lr1, X, y1, solver=Newton())
    @test J(θ1_newton)   ≤ 209.4
    θ1_newtoncg = fit(lr1, X, y1, solver=NewtonCG())
    @test J(θ1_newtoncg) ≤ 209.4
    θ1_lbfgs = fit(lr1, X, y1, solver=NewtonCG())
    @test J(θ1_lbfgs)    ≤ 209.4

    if SKLEARN
        # This checks that the parameters recovered using Sklearn lead
        # to a similar loss than the one given by our code to verify the
        # correctness of the code.
        lr_sk_ncg = SK_LM.LogisticRegression(C=1.0/λ, solver="newton-cg")
        lr_sk_ncg.fit(X, y1)
        θ1_sk_ncg = vcat(lr_sk_ncg.coef_[:], lr_sk_ncg.intercept_)
        @test J(θ1_sk_ncg)   ≤ 209.5
        lr_sk_lbfgs = SK_LM.LogisticRegression(C=1.0/λ, solver="lbfgs")
        lr_sk_lbfgs.fit(X, y1)
        θ1_sk_lbfgs = vcat(lr_sk_lbfgs.coef_[:], lr_sk_lbfgs.intercept_)
        @test J(θ1_sk_lbfgs) ≤ 209.5
    end
end

n, p, c = 500, 5, 4
((X, y, θ), (X_, y1, θ1)) = generate_multiclass(n, p, c; seed=525)

@testset "Multinomial" begin
    # No intercept
    λ = 5.0
    mnr = MultinomialRegression(λ; fit_intercept=false)
    J  = objective(mnr, X, y; c=c)
    @test J(θ)          ≤ 419.68
    θ_newtoncg = fit(mnr, X, y, solver=NewtonCG())
    @test J(θ_newtoncg) ≤ 384.34
    θ_lbfgs = fit(mnr, X, y, solver=R.LBFGS())
    @test J(θ_lbfgs)    ≤ 384.34

    #  With intercept
    mnr = MultinomialRegression(λ)
    J  = objective(mnr, X, y1; c=c)
    @test J(θ1)         ≤ 419.48
    θ_newtoncg = fit(mnr, X, y1, solver=NewtonCG())
    @test J(θ_newtoncg) ≤ 388.76
    θ_lbfgs = fit(mnr, X, y1, solver=R.LBFGS())
    @test J(θ_lbfgs)    ≤ 388.76

    if SKLEARN
        lr_sk_ncg = SK_LM.LogisticRegression(C=1.0/λ, solver="newton-cg",
                                             multi_class="multinomial")
        lr_sk_ncg.fit(X, y1)
        θ1_sk_ncg = vec(vcat(lr_sk_ncg.coef_', lr_sk_ncg.intercept_'))
        @test J(θ1_sk_ncg)   ≤ 390.31
        lr_sk_lbfgs = SK_LM.LogisticRegression(C=1.0/λ, solver="lbfgs",
                                               multi_class="multinomial")
        lr_sk_lbfgs.fit(X, y1)
        θ1_sk_lbfgs = vec(vcat(lr_sk_ncg.coef_', lr_sk_ncg.intercept_'))
        @test J(θ1_sk_lbfgs) ≤ 390.31
    end
end
