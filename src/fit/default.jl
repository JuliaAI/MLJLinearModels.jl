export fit

# Default solvers

# TODO: in the future, have cases where if the things are too big, take another
# default. Also should check if p > n in which case should do dual stuff (or
# other appropriate alternative)

# Linear, Ridge
_solver(::GLR{L2Loss,<:L2R}, np::NTuple{2,Int}) = Analytical()

# Logistic, Multinomial
_solver(::GLR{LogisticLoss,<:L2R},      np::NTuple{2,Int}) =  LBFGS()
_solver(::GLR{<:MultinomialLoss,<:L2R}, np::NTuple{2,Int}) = LBFGS()

# Lasso, ElasticNet, Logistic, Multinomial
function _solver(glr::GLR{<:SmoothLoss,<:ENR}, np::NTuple{2,Int})
    (is_l1(glr.penalty) || is_elnet(glr.penalty)) && return FISTA()
    @error "Not yet implemented."
end

# Robust, Quantile
_solver(::GLR{<:RobustLoss,<:L2R}, np::NTuple{2,Int}) = LBFGS()

# Fallback NOTE: should revisit bc with non-smooth, wouldn't work probably
# PGD/PSGD depending on how much data there is
_solver(::GLR, np::NTuple{2,Int}) = @error "Not yet implemented."


"""
$SIGNATURES

Fit a generalised linear regression model using an appropriate solver based on
the loss and penalty of the model. A method can, in some cases, be specified.
"""
function fit(::Type{T}, glr::GLR, X::AbstractMatrix{<:Real}, y::AVR; data=nothing,
             solver::Solver=_solver(glr, size(X))) where {T<:Real}
    if hasproperty(solver, :gram) && solver.gram
        # interpret X,y as X'X, X'y
        data = verify_or_construct_gramian(glr, X, y, data)
        p = size(data.XX, 2)
        return _fit(glr, solver, data.XX, data.Xy, (; dims=(data.n, p, 0)))
    else
        check_nrows(X, y)
        n, p = size(X)
        c = getc(glr, y)
        return _fit(T, glr, solver, X, y, scratch(n, p, c, i=glr.fit_intercept))
    end
end

function fit(glr::GLR, X::AbstractMatrix{<:Real}, y::AVR; kwargs...)
    return fit(eltype(X), glr, X, y; kwargs...)
end

fit(glr::GLR; kwargs...) = fit(glr, zeros((0,0)), zeros((0,)); kwargs...)


function scratch(n, p, c=0; i=false)
    p_ = p + Int(i)
    s = (n=zeros(n), n2=zeros(n), n3=zeros(n), p=zeros(p_), dims=(n,p_,c))
    if !iszero(c)
        s = (s..., nc=zeros(n,c), nc2=zeros(n,c), nc3=zeros(n,c),
                   nc4=zeros(n,c), pc=zeros(p_,c))
    end
    return s
end
scratch(X; kw...) = scratch(size(X)...; kw...)

npc(s) = s.dims
