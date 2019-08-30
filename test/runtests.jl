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
end

m("FIT", false); begin
    mm("analytical"); include("fit/analytical.jl")
    mm("newton");     include("fit/newton.jl")
    mm("proxgrad");   include("fit/proxgrad.jl")
    mm("robust");     include("fit/robust.jl")
    mm("quantile");   include("fit/quantile.jl")
end
