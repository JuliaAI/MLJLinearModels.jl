"""
$SIGNATURES

Return nothing if the number of rows of `X` and `y` match and throws a
`DimensionMismatch` error otherwise.
"""
function check_nrows(X::Matrix, y::VecOrMat)::Nothing
	size(X, 1) == size(y, 1) && return nothing
	throw(DimensionMismatch("`X` and `y` must have the same number of rows."))
end

"""
$SIGNATURES

Throws an error if the argument (e.g. penalty scaling) is negative.
"""
check_pos(λ::Real) = λ ≥ 0 || throw(ArgumentError("Penalty scaling should be positive."))

"""
$SIGNATURES

Given a matrix `X`, append a column of ones if `fit_intercept` is true.
"""
function augment_X(X::Matrix{<:Real}, fit_intercept::Bool)
	fit_intercept || return X
	return hcat(X, ones(eltype(X), size(X, 1)))
end

"""
$SIGNATURES

Return `X*θ` if `c=1` (default) or `X*P` where `P=reshape(θ, size(X, 2), p)` in the multi-class
case.
"""
function apply_X(X, θ, c=1)
	p = size(X, 2)
	if c == 1
		length(θ) == p || return X * θ[1:p] .+ θ[end]
		return X * θ
	else
		noβ = length(θ) == p * c
		W = reshape(θ, p + Int(!noβ), c)
		noβ || return X * view(W, 1:p, :) .+ view(W, p+1, :)'
		return X * W
	end
end


"""
$SIGNATURES

In-place application of X*θ.
"""
function apply_X!(Xθ, X, θ, c=1)
	p = size(X, 2)
	if c == 1
		if length(θ) == p
			mul!(Xθ, X, θ)
		else
			mul!(Xθ, X, view(θ, 1:p))
			Xθ .+= θ[end]
		end
	else
		noβ = length(θ) == p * c
		W 	= SCRATCH_PC[]
		copyto!(W, reshape(θ, p + Int(!noβ), c))
		if noβ
			mul!(Xθ, X, W)
		else
			mul!(Xθ, X, view(W, 1:p, :))
			Xθ .+= view(W, p+1, :)'
		end
	end
end


"""
$SIGNATURES

In-place application of X'*z (only for regression case).
"""
function apply_Xt!(Xtv, X, z)
	p  = size(X, 2)
	p_ = length(Xtv)
	if p == p_
		mul!(Xtv, X', z)
	else
		if Xtv isa SubArray
			mul!(Xtv, X', z)
			Xtv.parent[end] = sum(z)
		else
			mul!(view(Xtv, 1:p), X', z)
			Xtv[end] = sum(z)
		end
	end
end


"""
$SIGNATURES

Form (X'X) while being memory aware (assuming p ≪ n).
"""
function form_XtX(X, fit_intercept, lambda=0)
	if fit_intercept
		n, p = size(X)
        XtX  = zeros(p+1, p+1)
        Xt1  = sum(X, dims=1)
        mul!(view(XtX, 1:p, 1:p), X', X) # O(np²)
        @inbounds for i in 1:p
            XtX[i, end] = XtX[end, i] = Xt1[i]
        end
        XtX[end, end] = n
    else
        XtX = X'*X # O(np²)
    end
	if !iszero(lambda)
		λ = convert(eltype(XtX), lambda)
		@inbounds for i in 1:size(XtX, 1)
			XtX[i,i] += λ
		end
	end
	return Hermitian(XtX)
end


# Sigmoid and log-sigmoid

const SIGMOID_64 = log(Float64(1)/eps(Float64) - Float64(1))
const SIGMOID_32 = log(Float32(1)/eps(Float32) - Float32(1))

"""
$SIGNATURES

Return the sigmoid computed in a numerically stable way:

``σ(x) = 1/(1+exp(-x))``
"""
function sigmoid(x::Float64)
	x > SIGMOID_64  && return one(x)
	x < -SIGMOID_64 && return zero(x)
	return one(x) / (one(x) + exp(-x))
end
function sigmoid(x::Float32)
	x > SIGMOID_32  && return one(x)
	x < -SIGMOID_32 && return zero(x)
	return one(x) / (one(x) + exp(-x))
end
sigmoid(x) = sigmoid(float(x))
σ = sigmoid

"""
$SIGNATURES

Return the softmax computed in a numerically stable way:

``σ(x) = exp.(x) ./ sum(exp.(x))``

Implementation taken from NNlib.jl.
"""
function softmax(X::AbstractMatrix{<:Real})
	max_ = maximum(X, dims=2)
	exp_ = exp.(X .- max_)
	return exp_ ./ sum(exp_, dims=2)
end

"""
$SIGNATURES

Return the log sigmoid computed in a numerically stable way:

``logσ(x) = -log(1+exp(-x)) = log(exp(x)/(exp(x) + 1)) = x - log(1+exp(x))``
"""
function logsigmoid(x::Float64)
	x > SIGMOID_64  && return zero(x)
	x < -SIGMOID_64 && return x
	return -log1p(exp(-x))
end
function logsigmoid(x::Float32)
	x > SIGMOID_32  && return zero(x)
	x < -SIGMOID_32 && return x
	return -log1p(exp(-x))
end
logsigmoid(x) = logsigmoid(float(x))
logσ = logsigmoid


"""
$SIGNATURES

In place computation of `H = H + λI` where  `H` is a square matrix.
"""
function add_λI!(H::Matrix, λ::Real, penalize_intercept::Bool=true)
	λ = convert(eltype(H), λ)
	@inbounds for i in 1:size(H, 1)-1
		H[i,i] += λ
	end
	H[end, end] += ifelse(penalize_intercept, λ, zero(eltype(H)))
end


"""
$SIGNATURES

Soft-thresholding S_η(z).
"""
soft_thresh(z, η) = sign(z) * max(abs(z) - η, 0)


"""
$SIGNATURES

Threshold the number if its absolute value is too close to zero.
"""
clip(z, τ) = ifelse(abs(z) < τ, τ, z)


"""
$SIGNATURES

Return λ if penalize intercept otherwise 0, useful in computations of Hessian.
"""
λ_if_penalize_intercept(glr, λ) = ifelse(glr.penalize_intercept, λ, zero(λ))

"""
$SIGNATURES

Return a view of θ if the last element should not be penalized.
"""
@inline function view_θ(glr, θ)
	f = glr.fit_intercept && !glr.penalize_intercept
	f && return view(θ, 1:length(θ)-1)
	θ
end
