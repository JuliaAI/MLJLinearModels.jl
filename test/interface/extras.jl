# https://github.com/alan-turing-institute/MLJ.jl/issues/540
@testset "Levels" begin
    X, y = MLJBase.@load_iris
    df = DataFrame(hcat(DataFrame(X), y))
    X = filter(r -> r.x1 != "virginica", df)
    y = X.x1
    X = DataFrames.select(X, Not(:x1))
    mdl = LogisticClassifier()
    mach = MLJBase.machine(mdl, X, y)
    MLJBase.fit!(mach)
    p = MLJBase.predict(mach, rows=1:2)[1]
    @test typeof(p) <: MLJBase.UnivariateFinite

    X, y = MLJBase.@load_iris
    mach = MLJBase.machine(mdl, X, y)
    MLJBase.fit!(mach)
    p = MLJBase.predict(mach, rows=1:2)[1]
    @test typeof(p) <: MLJBase.UnivariateFinite
end
