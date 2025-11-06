# Julia code to generate the R test data in test/fit/quantile_r.toml

# RCall needs to be added to test environment

using CSV, DataFrames, Downloads
using MLJLinearModels
using RCall
using TOML
rimport("quantreg")

include(joinpath(@__DIR__, "..", "testutils.jl"))

url = "http://freakonometrics.free.fr/rent98_00.txt"
dataset = CSV.read(Downloads.download(url), DataFrame)
tau     = 0.3

y  = Vector(dataset[!,:rent_euro])
X  = Matrix(dataset[!,[:area, :yearc]])
X1 = hcat(X[:,1], X[:, 2], ones(size(X, 1)))

θ_qr_br = rcopy(getproperty(QUANTREG, :rq_fit_br)(X1, y; tau))[:coefficients]
θ_qr_lasso = rcopy(getproperty(QUANTREG, :rq_fit_lasso)(X1, y; tau=tau))[:coefficients]

write_to_TOML(
    joinpath(@__DIR__, "robust.toml"),
    (; θ_qr_br, θ_qr_lasso),
)
