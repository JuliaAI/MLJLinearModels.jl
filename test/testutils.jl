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
sparsify!(θ, s) = (θ .*= (rand(length(θ)) .< s))

"""Add outliers to portion s of vector."""
outlify(y, s) = (n = length(y); y .+ 20 * randn(n) .* (rand(n) .< s))

"""Generate continuous (X, y) with and without intercept."""
function generate_continuous(n, p; seed=0, sparse=1)
    Random.seed!(seed)
    X  = randn(n, p)
    X_ = R.augment_X(X, true)
    θ  = randn(p)
    θ1 = randn(p+1)
    sparse < 1 && begin
        sparsify!(θ, sparse)
        sparsify!(θ1, sparse)
    end
    y  = X*θ + 0.1 * randn(n)
    y1 = X_*θ1 + 0.1 * randn(n)
    return ((X, y, θ), (X_, y1, θ1))
end

"""Generate continuous X and binary y with and without intercept."""
function generate_binary(n, p; seed=0, sparse=1)
    Random.seed!(seed)
    X  = randn(n, p)
    X_ = R.augment_X(X, true)
    θ  = randn(p)
    θ1 = randn(p+1)
    sparse < 1 && begin
        sparsify!(θ, sparse)
        sparsify!(θ1, sparse)
    end
    y  = rand(n) .< R.σ.(X*θ)
    y  = y .* ones(Int, n) .- .!y .* ones(Int, n)
    y1 = rand(n) .< R.σ.(X_*θ1)
    y1 = y1 .* ones(Int, n) .- .!y1 .* ones(Int, n)
    return ((X, y, θ), (X_, y1, θ1))
end

"""Simple function to sample from a multinomial."""
function multi_rand(Mp)
    # Mp[i, :] sums to 1
    n, c = size(Mp)
    be   = reshape(rand(length(Mp)), n, c)
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
function generate_multiclass(n, p, c; seed=0, sparse=1)
    Random.seed!(seed)
    X   = randn(n, p)
    X_  = R.augment_X(X, true)
    θ   = randn(p * c)
    θ1  = randn((p+1) * c)
    sparse < 1 && begin
        sparsify!(θ, sparse)
        sparsify!(θ1, sparse)
    end
    y   = zeros(Int, n)
    y1  = zeros(Int, n)

    P   = R.apply_X(X, θ, c)
    M   = exp.(P)
    Mn  = M ./ sum(M, dims=2)
    P1  = R.apply_X(X, θ1, c)
    M1  = exp.(P1)
    Mn1 = M1 ./ sum(M1, dims=2)

    y  = multi_rand(Mn)
    y1 = multi_rand(Mn1)
    return ((X, y, θ), (X_, y1, θ1))
end
