import MLJBase: package_name, package_url, package_license,
    is_wrapper, is_pure_julia, input_scitype, target_scitype,
    supports_weights, docstring, load_path

@testset "meta-pkg" begin
    @show methods(package_name)
    rr = RidgeRegressor()
    @test package_name(rr) == "MLJLinearModels"
    @test package_url(rr) ==
        "https://github.com/alan-turing-institute/MLJLinearModels.jl"
    @test package_license(rr) == "MIT"
    @test is_pure_julia(rr) == true
    @test is_wrapper(rr) == false
end

@testset "meta-reg" begin
    lr = LinearRegressor()
    @test input_scitype(lr) == MLJBase.Table(MLJBase.Continuous)
    @test target_scitype(lr) == AbstractVector{MLJBase.Continuous}
    @test supports_weights(lr) == false
    @test !isempty(docstring(lr))
    @test load_path(lr) == "MLJLinearModels.LinearRegressor"
end

@testset "meta-clf" begin
    lr = LogisticClassifier()
    @test input_scitype(lr) == MLJBase.Table(MLJBase.Continuous)
    @test target_scitype(lr) == AbstractVector{<:MLJBase.Finite}
    @test supports_weights(lr) == false
    @test !isempty(docstring(lr))
    @test load_path(lr) == "MLJLinearModels.LogisticClassifier"
end
