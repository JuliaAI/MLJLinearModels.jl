@testset "Constructors" begin
    λ, γ, δ = 1.5, 2.5, 3.5

    # direct args constructor
    glr     = GeneralizedLinearRegression()
    ols     = LinearRegression()
    ridge   = RidgeRegression(λ; fit_intercept=false)
    lasso   = LassoRegression(λ; penalize_intercept=true)
    logreg0 = LogisticRegression(penalty=:none)
    logreg1 = LogisticRegression(λ)
    logreg2 = LogisticRegression(λ, γ)
    mnreg2  = MultinomialRegression(λ, γ)
    hlreg   = RobustRegression(HuberRho(δ), λ)
    ladreg  = LADRegression(λ)

    @test isa(glr.loss, L2Loss)
    @test isa(glr.penalty, NoPenalty)
    @test glr.fit_intercept
    @test !glr.penalize_intercept

    @test isa(ols.loss, L2Loss)
    @test isa(ols.penalty, NoPenalty)
    @test ols.fit_intercept

    @test isa(ridge.loss, L2Loss)
    @test isa(ridge.penalty, ScaledPenalty{L2Penalty})
    @test !ridge.fit_intercept
    @test ridge.penalty.scale == λ

    @test isa(lasso.loss, L2Loss)
    @test isa(lasso.penalty, ScaledPenalty{L1Penalty})
    @test lasso.penalty.scale == λ
    @test lasso.penalize_intercept

    @test isa(logreg0.loss, LogisticLoss)
    @test isa(logreg0.penalty, NoPenalty)

    @test isa(logreg1.loss, LogisticLoss)
    @test isa(logreg1.penalty, ScaledPenalty{L2Penalty})
    @test logreg1.penalty.scale == λ

    @test isa(logreg2.loss, LogisticLoss)
    @test isa(logreg2.penalty, CompositePenalty)
    @test isa(logreg2.penalty.penalties[1], ScaledPenalty{L2Penalty})
    @test isa(logreg2.penalty.penalties[2], ScaledPenalty{L1Penalty})
    @test logreg2.penalty.penalties[1].scale == λ
    @test logreg2.penalty.penalties[2].scale == γ

    @test isa(mnreg2.loss, MultinomialLoss)
    @test isa(mnreg2.penalty.penalties[2], ScaledPenalty{L1Penalty})
    @test mnreg2.penalty.penalties[1].scale == λ
    @test mnreg2.penalty.penalties[2].scale == γ

    @test isa(hlreg.loss, RobustLoss{HuberRho{δ}})
    @test isa(hlreg.penalty, ScaledPenalty{L2Penalty})
    @test hlreg.penalty.scale == λ
    @test hlreg.fit_intercept

    @test isa(ladreg.loss, RobustLoss{QuantileRho{0.5}})
    @test isa(ladreg.penalty, ScaledPenalty{L2Penalty})
    @test ladreg.fit_intercept

    # ======

    ridge   = RidgeRegression(lambda=λ, fit_intercept=true)
    lasso   = LassoRegression(lambda=λ)
    logreg1 = LogisticRegression(lambda=λ)
    logreg2 = LogisticRegression(lambda=λ, gamma=γ)
    mnreg2  = MultinomialRegression(lambda=λ, gamma=γ)
    hlreg   = RobustRegression(rho=HuberRho(delta=δ), lambda=λ)
    qreg    = QuantileRegression(delta=δ, lambda=λ)

    @test ridge.penalty.scale == λ
    @test ridge.fit_intercept
    @test lasso.penalty.scale == λ
    @test logreg1.penalty.scale == λ
    @test logreg2.penalty.penalties[1].scale == λ
    @test logreg2.penalty.penalties[2].scale == γ
    @test mnreg2.penalty.penalties[1].scale == λ
    @test mnreg2.penalty.penalties[2].scale == γ
    @test hlreg.loss.rho isa HuberRho{δ}
    @test hlreg.penalty.scale == λ
    @test qreg.loss.rho isa QuantileRho{δ}
    @test qreg.penalty.scale == λ
end
