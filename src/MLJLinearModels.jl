module MLJLinearModels

using Parameters, DocStringExtensions
using LinearAlgebra, IterativeSolvers
import LinearMaps: LinearMap
import IterativeSolvers: cg
import Optim

import Base.+, Base.-, Base.*, Base./, Base.convert

const AVR = AbstractVector{<:Real}

const TEMP_N   = Ref(zeros(0))
const TEMP_N2  = Ref(zeros(0))
const TEMP_N3  = Ref(zeros(0))
const TEMP_P   = Ref(zeros(0))
const TEMP_NC  = Ref(zeros(0,0))
const TEMP_NC2 = Ref(zeros(0,0))
allocate(n, p, c=0) = begin
    TEMP_N[]  = zeros(n)
    TEMP_N2[] = zeros(n)
    TEMP_N3[] = zeros(n)
    TEMP_P[]  = zeros(p)
    if !iszero(c)
        TEMP_NC[]  = zeros(n, c)
        TEMP_NC2[] = zeros(n, c)
    end
end
deallocate() = begin
    TEMP_N[]   = zeros(0)
    TEMP_N2[]  = zeros(0)
    TEMP_N3[]  = zeros(0)
    TEMP_P[]   = zeros(0)
    TEMP_NC[]  = zeros(0,0)
    TEMP_NC2[] = zeros(0,0)
end

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
