using MLJLinearModels, Test, LinearAlgebra, Random
DO_COMPARISONS = true; include("testutils.jl")

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
    R.deallocate()
end

m("FIT", false); begin
    mm("ols-ridge-lasso-elnet");  include("fit/ols-ridge-lasso-elnet.jl")
    mm("logistic & multinomial"); include("fit/logistic-multinomial.jl")
    mm("robust");                 include("fit/robust.jl")
    mm("quantile & LAD");         include("fit/quantile.jl")
end
