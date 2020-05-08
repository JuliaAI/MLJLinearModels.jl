const R = MLJLinearModels
const CI = get(ENV, "CI", "false") == "true"

DO_COMPARISONS = DO_COMPARISONS && !CI
DO_COMPARISONS && (using PyCall; using RCall)
SKLEARN_LM = nothing
PY_RND     = nothing
if DO_COMPARISONS
    SKLEARN_LM = pyimport("sklearn.linear_model")
    PY_RND     = pyimport("random")
    QUANTREG   = rimport("quantreg")
end

m(s, p=true) = println("\n== $s ==" * ifelse(p, "\n", ""))
mm(s) = println("\n > $s < \n")

nnz(θ) = sum(abs.(θ) .> 0)

"""Make portion s of vector 0."""
sparsify!(θ, s, r) = (θ .*= (rand(r, length(θ)) .< s))

"""Add outliers to portion s of vector."""
function outlify(y, s, r=StableRNG(123511))
    n = length(y)
    return y .+ 20 * randn(r, n) .* (rand(r, n) .< s)
end

"""Generate continuous (X, y) with and without intercept."""
function generate_continuous(n, p; seed=61234, sparse=1)
    r = StableRNG(seed)
    X  = randn(r, n, p)
    X1 = R.augment_X(X, true)
    θ  = randn(r, p)
    θ1 = randn(r, p+1)
    sparse < 1 && begin
        sparsify!(θ, sparse, r)
        sparsify!(θ1, sparse, r)
    end
    y  = X*θ + 0.1 * randn(r, n)
    y1 = X1*θ1 + 0.1 * randn(r, n)
    return ((X, y, θ), (X1, y1, θ1))
end

"""Generate continuous X and binary y with and without intercept."""
function generate_binary(n, p; seed=1345123, sparse=1)
    r = StableRNG(seed)
    X  = randn(r, n, p)
    X1 = R.augment_X(X, true)
    θ  = randn(r, p)
    θ1 = randn(r, p+1)
    sparse < 1 && begin
        sparsify!(θ, sparse, r)
        sparsify!(θ1, sparse, r)
    end
    y  = rand(r, n) .< R.σ.(X*θ)
    y  = y .* ones(Int, n) .- .!y .* ones(Int, n)
    y1 = rand(r, n) .< R.σ.(X1*θ1)
    y1 = y1 .* ones(Int, n) .- .!y1 .* ones(Int, n)
    return ((X, y, θ), (X1, y1, θ1))
end

"""Simple function to sample from a multinomial."""
function multi_rand(Mp, r)
    # Mp[i, :] sums to 1
    n, c = size(Mp)
    be   = reshape(rand(r, length(Mp)), n, c)
    y    = zeros(Int, n)
    @inbounds for i in eachindex(y)
        rp = 1.0
        for k in 1:c-1
            if (be[i, k] < Mp[i, k] / rp)
                y[i] = k
                break
            end
            rp -= Mp[i, k]
        end
    end
    y[y .== 0] .= c
    return y
end

"""Generate continuous X and multiclass y with and without intercept."""
function generate_multiclass(n, p, c; seed=53412224, sparse=1)
    r = StableRNG(seed)
    X   = randn(r, n, p)
    X1  = R.augment_X(X, true)
    θ   = randn(r, p * c)
    θ1  = randn(r, (p+1) * c)
    sparse < 1 && begin
        sparsify!(θ, sparse, r)
        sparsify!(θ1, sparse, r)
    end
    y   = zeros(Int, n)
    y1  = zeros(Int, n)

    P   = R.apply_X(X, θ, c)
    M   = exp.(P)
    Mn  = M ./ sum(M, dims=2)
    P1  = R.apply_X(X, θ1, c)
    M1  = exp.(P1)
    Mn1 = M1 ./ sum(M1, dims=2)

    y  = multi_rand(Mn, r)
    y1 = multi_rand(Mn1, r)
    return ((X, y, θ), (X1, y1, θ1))
end
