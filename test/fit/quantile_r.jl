# Julia code to generate the R test data in test/fit/quantile_r.toml

# RCall needs to be added to test environment

using MLJLinearModels
using RCall
using TOML
rimport("quantreg")

include(joinpath(@__DIR__, "..", "testutils.jl"))

# QuantileReg
n, p = 500, 5
((X, y, θ), (X1, y1, θ1)) = generate_continuous(n, p; seed=525)
y1a = outlify(y1, 0.1)
# computations using R:
δ = 0.5
θ_qr_br  = rcopy(QUANTREG.rq_fit_br(X1, y1a, tau=δ))[:coefficients]
θ_qr_fnb = rcopy(QUANTREG.rq_fit_fnb(X1, y1a, tau=δ))[:coefficients]
δ  = 0.75
θ_qr_br2  = rcopy(QUANTREG.rq_fit_br(X1, y1a, tau=δ))[:coefficients]

# LAD+L1
n, p = 500, 100
((X, y, θ), (X1, y1, θ1)) = generate_continuous(n, p;  seed=51112, sparse=0.1)
y1a  = outlify(y1, 0.1)
λ = 5.0
θ_qr_lasso = rcopy(QUANTREG.rq_fit_lasso(X1, y1a, lambda=λ))[:coefficients]

write_to_TOML(
    joinpath(@__DIR__, "quantile.toml"),
    (; θ_qr_br, θ_qr_fnb, θ_qr_br2, θ_qr_lasso),
)
