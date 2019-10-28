#=  ======================
    LINEAR REGRESSOR (OLS)
    ====================== =#

@with_kw_noshow mutable struct LinearRegressor <: MLJBase.Deterministic
    fit_intercept::Bool    = true
    solver::Option{Solver} = nothing
end

glr(m::LinearRegressor) = LinearRegression(fit_intercept=m.fit_intercept)

descr(::Type{LinearRegressor}) = "Regression with objective function ``|Xθ - y|₂²/2``."

#=  ===============
    RIDGE REGRESSOR
    =============== =#

@with_kw_noshow mutable struct RidgeRegressor <: MLJBase.Deterministic
    lambda::Real             = 1.0
    fit_intercept::Bool      = true
    penalize_intercept::Bool = false
    solver::Option{Solver}   = nothing
end

glr(m::RidgeRegressor) = RidgeRegression(m.lambda, fit_intercept=m.fit_intercept,
                                         penalize_intercept=m.penalize_intercept)

descr(::Type{RidgeRegressor}) = "Regression with objective function ``|Xθ - y|₂²/2 + λ|θ|₂²/2``."

#=  ===============
    LASSO REGRESSOR
    =============== =#

@with_kw_noshow mutable struct LassoRegressor <: MLJBase.Deterministic
    lambda::Real             = 1.0
    fit_intercept::Bool      = true
    penalize_intercept::Bool = false
    solver::Option{Solver}   = nothing
end

glr(m::LassoRegressor) = LassoRegression(m.lambda, fit_intercept=m.fit_intercept,
                                         penalize_intercept=m.penalize_intercept)

descr(::Type{LassoRegressor}) = "Regression with objective function ``|Xθ - y|₂²/2 + λ|θ|₁``."

#=  =====================
    ELASTIC NET REGRESSOR
    ===================== =#

@with_kw_noshow mutable struct ElasticNetRegressor <: MLJBase.Deterministic
    lambda::Real             = 1.0
    gamma::Real              = 0.0
    fit_intercept::Bool      = true
    penalize_intercept::Bool = false
    solver::Option{Solver}   = nothing
end

glr(m::ElasticNetRegressor) = ElasticNetRegression(m.lambda, m.gamma,
                                                   fit_intercept=m.fit_intercept,
                                                   penalize_intercept=m.penalize_intercept)

descr(::Type{ElasticNetRegressor}) = "Regression with objective function ``|Xθ - y|₂²/2 + λ|θ|₂²/2 + γ|θ|₁``."

#=  ==========================
    ROBUST REGRESSOR (General)
    ========================== =#

@with_kw_noshow mutable struct RobustRegressor <: MLJBase.Deterministic
    rho::RobustRho           = HuberRho(0.1)
    lambda::Real             = 1.0
    gamma::Real              = 0.0
    penalty::Symbol          = :l2
    fit_intercept::Bool      = true
    penalize_intercept::Bool = false
    solver::Option{Solver}   = nothing
end

glr(m::RobustRegressor) = RobustRegression(m.rho, m.lambda, m.gamma; penalty=m.penalty,
                                           fit_intercept=m.fit_intercept,
                                           penalize_intercept=m.penalize_intercept)

descr(::Type{RobustRegressor}) = "Robust regression with objective ``∑ρ(Xθ - y) + λ|θ|₂² + γ|θ|₁`` for a given robust `ρ`."

#=  ===============
    HUBER REGRESSOR
    =============== =#

@with_kw_noshow mutable struct HuberRegressor <: MLJBase.Deterministic
    delta::Real              = 0.5
    lambda::Real             = 1.0
    gamma::Real              = 0.0
    penalty::Symbol          = :l2
    fit_intercept::Bool      = true
    penalize_intercept::Bool = false
    solver::Option{Solver}   = nothing
end

glr(m::HuberRegressor) = HuberRegression(m.delta, m.lambda, m.gamma; penalty=m.penalty,
                                         fit_intercept=m.fit_intercept,
                                         penalize_intercept=m.penalize_intercept)

descr(::Type{HuberRegressor}) = "Robust regression with objective ``∑ρ(Xθ - y) + λ|θ|₂² + γ|θ|₁`` where `ρ` is the Huber Loss."

#=  ==================
    QUANTILE REGRESSOR
    ================== =#

@with_kw_noshow mutable struct QuantileRegressor <: MLJBase.Deterministic
    delta::Real              = 0.5
    lambda::Real             = 1.0
    gamma::Real              = 0.0
    penalty::Symbol          = :l2
    fit_intercept::Bool      = true
    penalize_intercept::Bool = false
    solver::Option{Solver}   = nothing
end

glr(m::QuantileRegressor) = QuantileRegression(m.delta, m.lambda, m.gamma; penalty=m.penalty,
                                               fit_intercept=m.fit_intercept,
                                               penalize_intercept=m.penalize_intercept)

descr(::Type{QuantileRegressor}) = "Robust regression with objective ``∑ρ(Xθ - y) + λ|θ|₂² + γ|θ|₁`` where `ρ` is the Quantile Loss."

#=  ==================================
    LEAST ABSOLUTE DEVIATION REGRESSOR
    ================================== =#

@with_kw_noshow mutable struct LADRegressor <: MLJBase.Deterministic
    lambda::Real             = 1.0
    gamma::Real              = 0.0
    penalty::Symbol          = :l2
    fit_intercept::Bool      = true
    penalize_intercept::Bool = false
    solver::Option{Solver}   = nothing
end

glr(m::LADRegressor) = LADRegression(m.lambda, m.gamma; penalty=m.penalty,
                                     fit_intercept=m.fit_intercept,
                                     penalize_intercept=m.penalize_intercept)

descr(::Type{LADRegressor}) = "Robust regression with objective ``∑ρ(Xθ - y) + λ|θ|₂² + γ|θ|₁`` where `ρ` is the Absolute Loss."
