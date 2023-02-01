const DOC_SOLVERS =
    "Different solver options exist, as indicated under "*
    "\"Hyperparameters\" below. "

function example_docstring(m; nclasses = nothing)
"""
## Example

    using MLJ
    X, y = $(nclasses == nothing ? "make_regression()" : "make_blobs(centers = $nclasses)")
    mach = fit!(machine($m(), X, y))
    predict(mach, X)
    fitted_params(mach)

"""
end

function doc_header(ModelType)
    name = MLJModelInterface.name(ModelType)
    human_name = MLJModelInterface.human_name(ModelType)

    """
        $name

    A model type for constructing a $human_name, based on
    [MLJLinearModels.jl](https://github.com/alan-turing-institute/MLJLinearModels.jl), and
    implementing the MLJ model interface.

    From MLJ, the type can be imported using

        $name = @load $name pkg=MLJLinearModels

    Do `model = $name()` to construct an instance with default
    hyper-parameters.

    """
end

const DOC_PROXGRAD = "Aliases `ISTA` and `FSTA` correspond to "*
    "`ProxGrad` with the option `acceleration=false` or `true` respectively. "
