# # LOGIC AND MESSAGING FOR HANDLING COMPARISONS WITH R AND PYTHON

# tests and benchmarks requiring the R and python models are suppressed when
# `DO_COMPARISONS == false`.

const DO_COMPARISONS = get(ENV, "", "true") == "true"
if DO_COMPARISONS
    @info """
          Running comparisons with R and python models.
          Run `ENV["DO_COMPARISONS"] = "false"` to suppress these.
          """
else
    @info """
          Excluding comparisons with R and python models.
          Run `ENV["DO_COMPARISONS"] = "true"` to re-instate these.
          """
end

if DO_COMPARISONS
    using PyCall
    using RCall
end

SKLEARN_LM = nothing
PY_RND     = nothing
if DO_COMPARISONS
    SKLEARN_LM = pyimport("sklearn.linear_model")
    PY_RND     = pyimport("random")
    QUANTREG   = rimport("quantreg")
end


# # TESTS

using MLJLinearModels, Test, LinearAlgebra
using Random, StableRNGs, DataFrames, ForwardDiff
import Optim
import MLJ, MLJBase

include("testutils.jl")

m("UTILS"); include("utils.jl")

m("LOSS-PENALTY", false); begin
    mm("generic"); include("loss-penalty/generic.jl")
    mm("utils");   include("loss-penalty/utils.jl")
    mm("robust");  include("loss-penalty/robust.jl")
end

m("GLR", false); begin
    mm("constructors"); include("glr/constructors.jl")
    mm("utils");        include("glr/tools-utils.jl")
    mm("grads-hess");   include("glr/grad-hess-prox.jl")
end

m("FIT", false); begin
    mm("ols-ridge-lasso-elnet");  include("fit/ols-ridge-lasso-elnet.jl")
    mm("logistic & multinomial"); include("fit/logistic-multinomial.jl")
    mm("robust");                 include("fit/robust.jl")
    mm("quantile & LAD");         include("fit/quantile.jl")
end

m("MLJ", false); begin
    mm("metadata");    include("interface/meta.jl")
    mm("fit-predict"); include("interface/fitpredict.jl")
    mm("extras");      include("interface/extras.jl")
end

m("MISC", false); begin
    mm("benchmarks"); include("benchmarks/robust.jl")
end
