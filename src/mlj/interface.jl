export LinearRegressor, RidgeRegressor, LassoRegressor, ElasticNetRegressor,
       RobustRegressor, HuberRegressor, QuantileRegressor, LADRegressor,
       LogisticClassifier, MultinomialClassifier

const SymStr = Union{Symbol,String}

include("doc_tools.jl")
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
    sch = MMI.schema(X)
    features = (sch === nothing) ? nothing : sch.names
    reg      = glr(m)
    solver   = m.solver === nothing ? _solver(reg, size(Xmatrix)) : m.solver
    verb > 0 && @info "Solver: $(solver)"
    # get the parameters
    θ = fit(reg, Xmatrix, y; solver=solver)
    # return
    return (θ, features), nothing, NamedTuple{}()
end

MMI.predict(m::Union{REG_MODELS...}, (θ, features), Xnew) = apply_X(MMI.matrix(Xnew), θ)

function MMI.fitted_params(m::Union{REG_MODELS...}, (θ, features))
    m.fit_intercept && return (coefs = coef_vec(θ[1:end-1], features), intercept = θ[end])
    return (coefs = coef_vec(θ, features), intercept = nothing)
end

#= ===========
   CLASSIFIERS
   =========== =#

function MMI.fit(m::Union{CLF_MODELS...}, verb::Int, X, y)
    Xmatrix  = MMI.matrix(X)
    sch = MMI.schema(X)
    features = (sch === nothing) ? nothing : sch.names
    yplain   = convert.(Int, MMI.int(y))
    classes  = MMI.classes(y[1])
    nclasses = length(classes)
    if nclasses < 2
        throw(
            DomainError(
                "The target `y` needs to have two or more levels."
            )
        )
    elseif nclasses == 2 && m isa LogisticClassifier
        # recode to ± 1
        yplain[yplain .== 1] .= -1
        yplain[yplain .== 2] .= 1
        # NOTE: fallback for dimensions so that we're fully
        # in the Logistic case and not in the Multinomial {0} case!
        nclasses = 0
    end
    # NOTE: here nclasses is either
    #   - 0  Logistic, binary
    #   - 2  Multinomial, binary
    #   - >2 or > 2 (Multinomial)
    clf    = glr(m, nclasses)
    solver = m.solver === nothing ? _solver(clf, size(Xmatrix)) : m.solver
    verb > 0 && @info "Solver: $(solver)"
    # get the parameters
    θ = fit(clf, Xmatrix, yplain, solver=solver)
    # return
    return (θ, features, classes, nclasses), nothing, NamedTuple{}()
end

function MMI.predict(m::Union{CLF_MODELS...}, (θ, features, classes, c), Xnew)
    Xmatrix = MMI.matrix(Xnew)
    preds = apply_X(Xmatrix, θ, c)

    if c > 2 || m isa MultinomialClassifier # multiclass
        preds .= softmax(preds)

    else # binary logistic (necessarily c==0)
        preds  .= sigmoid.(preds)
        preds   = hcat(1.0 .- preds, preds) # scores for -1 and 1
        return MMI.UnivariateFinite(classes, preds)
    end

    return MMI.UnivariateFinite(classes, preds)
end

function MMI.fitted_params(m::Union{CLF_MODELS...}, (θ, features, classes, c))
    # helper function to assemble the results
    _fitted_params(coefs, features, intercept) =
        (classes=classes, coefs=coef_vec(coefs, features), intercept=intercept)
    # multiclass or binary?
    if c > 2
        W = reshape(θ, :, c)
        if m.fit_intercept
            return _fitted_params(W, features, W[end, :])
        end
        return _fitted_params(W, features, nothing)
    end
    # single class (necessarily c==0)
    m.fit_intercept && return _fitted_params(θ[1:end-1], features, θ[end])
    return _fitted_params(θ, features, nothing)
end

coef_vec(W::AbstractMatrix, features) =
    [feature => coef for (feature, coef) in zip(features, eachrow(W))]
coef_vec(θ::AbstractVector, features) =
    [feature => coef for (feature, coef) in zip(features, θ)]
coef_vec(W::AbstractMatrix, ::Nothing) = W
coef_vec(θ::AbstractVector, ::Nothing) = θ

#= =======================
   METADATA FOR ALL MODELS
   ======================= =#

MMI.metadata_pkg.(ALL_MODELS,
    package_name="MLJLinearModels",
    package_uuid="6ee0df7b-362f-4a72-a706-9e79364fb692",
    package_url="https://github.com/alan-turing-institute/MLJLinearModels.jl",
    is_pure_julia=true,
    package_license="MIT",
    is_wrapper=false)

lp_(M) = "MLJLinearModels.$(MMI.name(M))"

for M in REG_MODELS
    MMI.metadata_model(
        M,
        input_scitype=MMI.Table(MMI.Continuous),
        target_scitype=AbstractVector{MMI.Continuous},
        load_path=lp_(M),
    )
end
for M in CLF_MODELS
    MMI.metadata_model(
        M,
        input_scitype=MMI.Table(MMI.Continuous),
        target_scitype=AbstractVector{<:MMI.Finite},
        load_path=lp_(M),
    )
end

MMI.human_name(::Type{<:LADRegressor}) = "least absolute deviation regressor"
