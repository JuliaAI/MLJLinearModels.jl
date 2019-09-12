module MLJLinearModels

using Parameters, DocStringExtensions
using LinearAlgebra, IterativeSolvers
import LinearMaps: LinearMap
import IterativeSolvers: cg
import Optim

import Base.+, Base.-, Base.*, Base./, Base.convert

const AVR = AbstractVector{<:Real}

include("scratchspace.jl")

include("utils.jl")

# > Loss / penalty definitions <
include("loss-penalty/generic.jl")
include("loss-penalty/standard.jl")
include("loss-penalty/robust.jl")
include("loss-penalty/utils.jl")

# > Constructors for regression models <
include("glr/constructors.jl")
include("glr/d_l2loss.jl")
include("glr/d_logistic.jl")
include("glr/d_robust.jl")
include("glr/prox.jl")
include("glr/utils.jl")

# > Solvers <
include("fit/solvers.jl")
include("fit/default.jl")
include("fit/analytical.jl")
# include("fit/grad.jl")
include("fit/newton.jl")
# include("fit/pnewton.jl")
include("fit/proxgrad.jl")
include("fit/iwls.jl")
# include("fit/admm.jl")

end # module
