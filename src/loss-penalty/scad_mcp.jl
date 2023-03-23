export ScadPenalty, MCPPenalty

# FoldedConcavePenalty
abstract type FCPenalty{λ,γ} <: AtomicPenalty where {λ<:Real,γ<:Real} end

struct ScadPenalty{λ,γ} <: FCPenalty{λ,γ} end
struct MCPPenalty{λ,γ} <: FCPenalty{λ,γ} end

getlambda(p::FCPenalty{λ,γ}) where {λ,γ} = λ
getgamma(p::FCPenalty{λ,γ}) where {λ,γ} = γ

# not efficient but doesn't matter, the derivative matters more
(p::ScadPenalty{λ,γ})(x::Real) where {λ, γ} = begin
	abs_x = abs(x)
	if abs_x <= λ
		λ * abs_x
	elseif abs_x >= γ * λ
		λ^2 * (γ + 1) / 2
	else
		(2γ * λ * abs_x - x^2 - λ^2) / (2 * (γ - 1))
	end
end

(p::FCPenalty{λ,γ})(θ) where {λ,γ} = p.(θ)

∇(p::ScadPenalty{λ,γ}) where {λ,γ} = θ -> begin
	T 	   = promote_type(eltype(θ), typeof(λ), typeof(γ))
	abs_θ  = abs.(T.(θ))
	λ̂, γ̂   = T(λ), T(γ)
	λγ 	   = λ̂ * γ̂

	left   = λ̂ * ones(T, length(θ))
	middle = @. (λγ - abs_θ) / (γ̂ - T(1))

	return @. (abs_θ <= λ̂) * left + (λ̂ < abs_θ < λγ) * middle
end

∇(p::MCPPenalty{λ,γ}) where {λ,γ} = θ -> begin
	T 	  = promote_type(eltype(θ), typeof(λ), typeof(γ))
	abs_θ = abs.(T.(θ))
	λ̂, γ̂   = T(λ), T(γ)
	λγ 	   = λ̂ * γ̂

	return  @. (abs_θ <= λγ) * (λ̂ - abs_θ / γ̂) * sign(θ)
end
