# Follow up from issue #147 comparing quantreg more specifically.

if DO_COMPARISONS
    @testset "Comp-QR-147" begin
        using CSV, DataFrames

        dataset = CSV.read(download("http://freakonometrics.free.fr/rent98_00.txt"), DataFrame)
        tau     = 0.3

        y  = Vector(dataset[!,:rent_euro])
        X  = Matrix(dataset[!,[:area, :yearc]])
        X1 = hcat(X[:,1], X[:, 2], ones(size(X, 1)))

        qr  = QuantileRegression(tau; penalty=:none)
        obj = objective(qr, X, y)

        θ_lbfgs = fit(qr, X, y)
        @test isapprox(obj(θ_lbfgs), 226_639, rtol=1e-4)

        # in this case QR with BR method does better
        θ_qr_br = rcopy(getproperty(QUANTREG, :rq_fit_br)(X1, y; tau=tau))[:coefficients]
        @test isapprox(obj(θ_qr_br), 207_551, rtol=1e-4)

        # lasso doesn't
        θ_qr_lasso = rcopy(getproperty(QUANTREG, :rq_fit_lasso)(X1, y; tau=tau))[:coefficients]
        obj(θ_qr_lasso) # 229_172
        @test 228_000 ≤ obj(θ_qr_lasso) ≤ 231_000
    end
end
