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
    lr1 = LogisticRegression(λ)
    J   = objective(lr1, X, y1)
    θ_newton   = fit(lr1, X, y1, solver=Newton())
    θ_newtoncg = fit(lr1, X, y1, solver=NewtonCG())
    θ_lbfgs    = fit(lr1, X, y1, solver=NewtonCG())
    @test isapprox(J(θ1),         213.80970, rtol=1e-5)
    @test isapprox(J(θ_newton),   209.32539, rtol=1e-5) # <- ref value
    @test isapprox(J(θ_newtoncg), 209.32539, rtol=1e-5)
    @test isapprox(J(θ_lbfgs),    209.32539, rtol=1e-5)

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
        @test isapprox(J(θ1_sk_ncg),   209.32539, rtol=1e-3)
        @test isapprox(J(θ1_sk_lbfgs), 209.32539, rtol=1e-3)
        # NOTE in fact here we get better results but that's not really meaningful given
        # the stopping criterions are not identical etc.
        @test 209.32539 < J(θ1_sk_ncg)
        @test 209.32539 < J(θ1_sk_lbfgs)
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
    mnr = MultinomialRegression(λ)
    J   = objective(mnr, X, y1; c=c)
    θ_newtoncg = fit(mnr, X, y1, solver=NewtonCG())
    θ_lbfgs    = fit(mnr, X, y1, solver=R.LBFGS())
    @test isapprox(J(θ),          1244.46404, rtol=1e-5)
    @test isapprox(J(θ_newtoncg),  388.75018, rtol=1e-5) # <- ref value
    @test isapprox(J(θ_lbfgs),     388.75018, rtol=1e-5)

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
        @test isapprox(J(θ1_sk_ncg),   388.75018, rtol=5e-3)
        @test isapprox(J(θ1_sk_lbfgs), 388.75018, rtol=5e-3)
        # Again we get better results but it doesn't really matter
        @test 388.75018 < J(θ1_sk_ncg)
        @test 388.75018 < J(θ1_sk_lbfgs)
    end
end
