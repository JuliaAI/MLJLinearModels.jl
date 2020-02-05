export LinearRegressor, RidgeRegressor, LassoRegressor, ElasticNetRegressor,
       RobustRegressor, HuberRegressor, QuantileRegressor, LADRegressor,
       LogisticClassifier, MultinomialClassifier

const SymStr = Union{Symbol,String}

include("regressors.jl")
include("classifiers.jl")

const REG_MODELS = (LinearRegressor, RidgeRegressor, LassoRegressor,
                    ElasticNetRegressor, RobustRegressor, HuberRegressor,
                    QuantileRegressor, LADRegressor)
const CLF_MODELS = (LogisticClassifier, MultinomialClassifier)
const ALL_MODELS = (REG_MODELS..., CLF_MODELS...)

#= ==========
   REGRESSORS
   ========== =#

function MMI.fit(m::Union{REG_MODELS...}, verb::Int, X, y)
    Xmatrix = MMI.matrix(X)
    reg     = glr(m)
    solver  = m.solver === nothing ? _solver(reg, size(Xmatrix)) : m.solver
    # get the parameters
    θ = fit(reg, Xmatrix, y; solver=solver)
    # return
    return θ, nothing, NamedTuple{}()
end

MMI.predict(m::Union{REG_MODELS...}, θ, Xnew) = apply_X(MMI.matrix(Xnew), θ)

function MMI.fitted_params(m::Union{REG_MODELS...}, θ)
    m.fit_intercept && return (coefs = θ[1:end-1], intercept = θ[end])
    return (coefs = θ, intercept = nothing)
end

#= ===========
   CLASSIFIERS
   =========== =#

function MMI.fit(m::Union{CLF_MODELS...}, verb::Int, X, y)
    Xmatrix  = MMI.matrix(X)
    yplain   = convert.(Int, MMI.int(y))
    classes  = MMI.classes(y[1])
    nclasses = length(classes)
    if nclasses == 2
        # recode
        yplain[yplain .== 1] .= -1
        yplain[yplain .== 2] .= 1
        c = 1
    else
        c = nclasses
    end
    clf    = glr(m)
    solver = m.solver === nothing ? _solver(clf, size(Xmatrix)) : m.solver
    # get the parameters
    θ = fit(clf, Xmatrix, yplain, solver=solver)
    # return
    return (θ, c, classes), nothing, NamedTuple{}()
end

function MMI.predict(m::Union{CLF_MODELS...}, (θ, c, classes), Xnew)
    Xmatrix = MMI.matrix(Xnew)
    preds   = apply_X(Xmatrix, θ, c)
    # binary classification
    if c == 1
        preds  .= sigmoid.(preds)
        preds   = hcat(1.0 .- preds, preds) # scores for -1 and 1
        return [MMI.UnivariateFinite(classes, preds[i, :]) for i in 1:size(Xmatrix,1)]
    end
    # multiclass
    preds .= softmax(preds)
    return [MMI.UnivariateFinite(classes, preds[i, :]) for i in 1:size(Xmatrix,1)]
end

function MMI.fitted_params(m::Union{CLF_MODELS...}, (θ, c, classes))
    if c > 1
        if m.fit_intercept
            W = reshape(θ, div(length(θ), c), c)
            return (coefs = W, intercept = nothing)
        end
        W = reshape(θ, p+1, c)
        return (coefs = W[1:p, :], intercept = W[end, :])
    end
    # single class
    m.fit_intercept && return (coefs = θ[1:end-1], intercept = θ[end])
    return (coefs = θ, intercept = nothing)
end

#= =======================
   METADATA FOR ALL MODELS
   ======================= =#

MMI.metadata_pkg.(ALL_MODELS,
    name="MLJLinearModels",
    uuid="6ee0df7b-362f-4a72-a706-9e79364fb692",
    url="https://github.com/alan-turing-institute/MLJLinearModels.jl",
    julia=true,
    license="MIT",
    is_wrapper=false)

descr_(M) = descr(M) *
    "\n→ based on [MLJLinearModels](https://github.com/alan-turing-institute/MLJLinearModels.jl)" *
    "\n→ do `@load $(MMI.name(M)) pkg=\"MLJLinearModels\" to use the model.`" *
    "\n→ do `?$(MMI.name(M))` for documentation."
lp_(M) = "MLJLinearModels.$(MMI.name(M))"

for M in REG_MODELS
    MMI.metadata_model(M,
        input=MMI.Table(MMI.Continuous),
        target=AbstractVector{MMI.Continuous},
        weights=false,
        descr=descr_(M), path=lp_(M))
end
for M in CLF_MODELS
    MMI.metadata_model(M,
        input=MMI.Table(MMI.Continuous),
        target=AbstractVector{<:MMI.Finite},
        weights=false,
        descr=descr_(M), path=lp_(M))
end
