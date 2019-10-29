@testset "meta-pkg" begin
    rr = RidgeRegressor()
    d = MLJBase.info_dict(rr)
    @test d[:package_name] == "MLJLinearModels"
    @test d[:package_url] == "https://github.com/alan-turing-institute/MLJLinearModels.jl"
    @test d[:package_license] == "MIT"
    @test d[:is_pure_julia] == true
    @test d[:is_wrapper] == false
end

@testset "meta-reg" begin
    lr = LinearRegressor()
    d = MLJBase.info_dict(lr)
    @test d[:input_scitype] == MLJBase.Table(MLJBase.Continuous)
    @test d[:target_scitype] == AbstractVector{MLJBase.Continuous}
    @test d[:supports_weights] == false
    @test !isempty(d[:docstring])
    @test d[:load_path] == "MLJLinearModels.LinearRegressor"
end

@testset "meta-clf" begin
    lr = LogisticClassifier()
    d = MLJBase.info_dict(lr)
    @test d[:input_scitype] == MLJBase.Table(MLJBase.Continuous)
    @test d[:target_scitype] == AbstractVector{<:MLJBase.Finite}
    @test d[:supports_weights] == false
    @test !isempty(d[:docstring])
    @test d[:load_path] == "MLJLinearModels.LogisticClassifier"
end
