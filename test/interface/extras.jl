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

# https://github.com/JuliaAI/MLJLinearModels.jl/issues/123
@testset "Binary" begin
    X = (x1=rand(10),)
    y = MLJBase.coerce(vcat(fill("a", 10), ["b", ]), MLJBase.Multiclass)[1:10]
    mach = MLJBase.machine(MultinomialClassifier(), X, y) |> MLJBase.fit!
end

# https://github.com/JuliaAI/MLJLinearModels.jl/issues/129
@testset "Crabs" begin
    data = MLJ.load_crabs()
    y_, X = MLJ.unpack(data, ==(:sp), col->col in [:FL, :RW]);
    y = MLJ.coerce(y_, MLJ.OrderedFactor);
    model = MultinomialClassifier()
    mach = MLJ.machine(model, X, y) |> MLJ.fit!
    yhat = MLJ.predict_mode(mach, X)

    # crappy "test" but we're just testing that predict works fine    
    @test MLJ.misclassification_rate(yhat, y) < 0.4

    model = LogisticClassifier()
    mach = MLJ.machine(model, X, y) |> MLJ.fit!
    yhat = MLJ.predict_mode(mach, X)

    @test MLJ.misclassification_rate(yhat, y) < 0.4
end