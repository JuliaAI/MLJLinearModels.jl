export LPLoss, LPPenalty,
        L1Loss, L1Penalty,
        L2Loss, L2Penalty,
        LogisticLoss, MultinomialLoss

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
# No Loss / No Penalty
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

(l::NoLoss)(a::AVR, b::AVR) = 0.0
(p::NoPenalty)(θ::AVR)      = 0.0

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
# LP Losses and Penalties
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

"""
$TYPEDEF

Scaled L-p loss of the residual.

``L(x, y) = ||x-y||_{p}^{p}/p``

The scaling simplifies expressions in the common L2 case.
"""
struct LPLoss{p} <: AtomicLoss where p <: Real end


"""
$TYPEDEF

Scaled L-p norm of the parameter vector.

``P(θ) = ||θ||_{p}^{p}/p``

The scaling simplifies expressions in the common L2 case.
"""
struct LPPenalty{p} <: AtomicPenalty where p <: Real end


# Useful Shortcuts
const L1Loss    = LPLoss{1}
const L1Penalty = LPPenalty{1}
const L2Loss    = LPLoss{2}
const L2Penalty = LPPenalty{2}
const LPCost{p} = Union{LPLoss{p},LPPenalty{p}}

const L1R = ScaledPenalty{L1Penalty}
const L2R = Union{NoPenalty,ScaledPenalty{L2Penalty}}
const ENR = Union{L1R,CompositePenalty}

"""
$SIGNATURES

Return the `p` in an `LPCost{p}`.
"""
getp(lpc::LPCost{p}) where p = p


"""
$SIGNATURES

Compute the lp norm to the `p`-th power of a vector given `p` scaled by `p`.
"""
function lp(v::AbstractVector{<:Real}, p)
    p == Inf && return maximum(v)
    p == 1   && return sum(abs.(v))
    p == 2   && return sum(abs2.(v)) / 2
    p  > 0   && return sum(abs.(v).^p) / p
    throw(DomainError("[lp] `p` has to be greater than 0"))
end

(l::LPLoss)(a::AVR, b::AVR) = lp(a .- b, getp(l))
(l::LPLoss)(r::AVR)         = lp(r, getp(l))
(p::LPPenalty)(θ::AVR)      = lp(θ, getp(p))

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
# Logistic loss
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

"""
$TYPEDEF

``L(x, y) = -∑logσ(xᵢyᵢ)``

where `logσ` is the log of the sigmoid function; `yᵢ ∈ {±1}`. In a logistic regression `x`
corresponds to `Xθ` where `X` is the design matrix and `θ` the vector of parameters.
See [`logsigmoid`](@ref).
"""
struct LogisticLoss <: AtomicLoss end

(::LogisticLoss)(x::AVR, y::AVR) = -sum(logsigmoid.(x .* y))


"""
$TYPEDEF

``L(P, y) = ∑log Zᵢ - ∑∑ 1(yᵢ=j)Pᵢⱼ``

where `P` is a matrix where each row contains class probabilities, `yᵢ ∈ {1, 2, ..., K}`
corresponding to column indices and,

``Zᵢ = ∑ exp(Pᵢ)``

In a multinomial regression, `P` corresponds to `XW` where `X` is the design matrix and `W` the
matrix of size `p * K` where each column corresponds to the parameters corresponding to that class.
"""
struct MultinomialLoss <: AtomicLoss end

(::MultinomialLoss)(P::Matrix{<:Real}, y::Vector{Int}) = begin
    L = 0.0
    @inbounds for i in eachindex(y)
        Pᵢ = P[i, :]
        m  = maximum(Pᵢ)
        sᵢ = sum(exp.(Pᵢ .- m)) # avoid overflow
        logZᵢ = log(sᵢ) + m
        L += logZᵢ - P[i, y[i]]
    end
    return L
end
(l::MultinomialLoss)(y::Vector{Int}, P::Matrix{<:Real}) = l(P, y)
