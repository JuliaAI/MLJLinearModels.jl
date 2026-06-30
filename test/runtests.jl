# # LOGIC AND MESSAGING FOR HANDLING COMPARISONS WITH R AND PYTHON

# tests and benchmarks requiring the R and python models are suppressed when
# `DO_COMPARISONS == false`.

const DO_COMPARISONS = get(ENV, "DO_COMPARISONS", "true") == "true"

if DO_COMPARISONS
    python  = get(ENV, "PYTHON", "<unset>")
    @info """

          Running comparisons with python models.
          Run `ENV["DO_COMPARISONS"] = "false"` to suppress these.

          These comparisons may fail unless ENV["PYTHON"] points a python installation
          that includes the sklearn library (currently set to $python).

          You may need to explicitly rebuild PyCall after changing these paths.

          Attention maintainers of MLJLinearModels:

          In the GitHub testing workflow be sure `PyCall` are explicitly added
          to the load path and built with a valid ENV before julia-runtest is
          executed. Otherwise, julia-actions/cache may be caching invalid builds.

          """
else @info """

           Excluding comparisons with R and python models.  Run `ENV["DO_COMPARISONS"] =
           "true"` to re-instate these.

           """
end

SKLEARN_LM = nothing
PY_RND     = nothing
if DO_COMPARISONS
    using PyCall
    SKLEARN_LM = pyimport("sklearn.linear_model")
    PY_RND     = pyimport("random")
end


# # TESTS

using MLJLinearModels, Test, LinearAlgebra
using Random, StableRNGs, DataFrames, ForwardDiff
import Optim
import MLJ, MLJBase
using TOML

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
