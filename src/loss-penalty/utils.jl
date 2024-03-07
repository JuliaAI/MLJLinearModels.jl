is_l1(p::Penalty) = isa(p, L1R)
function is_elnet(cp::CompositePenalty)
    length(cp.penalties) == 2 || return false
    isa(cp.penalties[1], L2R) && isa(cp.penalties[2], L1R) && return true
    isa(cp.penalties[1], L1R) && isa(cp.penalties[2], L2R) && return true
    return false
end

get_l1(p::ScaledPenalty{L1Penalty}) = p
get_l1(cp::CompositePenalty) = cp.penalties[findfirst(e->isa(e, L1R), cp.penalties)]
get_l2(p::ScaledPenalty{L1Penalty}) = NoPenalty()
get_l2(cp::CompositePenalty) = cp.penalties[findfirst(e->isa(e, L2R), cp.penalties)]

getscale_l1(p1::Union{NoPenalty,L1R}) = p1 |> getscale
getscale_l1(cp::CompositePenalty) = is_elnet(cp) ? cp |> get_l1 |> getscale :
                                                   @error "Case not implemented."
getscale_l2(p1::Union{NoPenalty,L1R}) = 0.0
getscale_l2(p2::L2R) = p2 |> getscale
getscale_l2(cp::CompositePenalty) = is_elnet(cp) ? cp |> get_l2 |> getscale :
                                                   @error "Case not implemented."

function get_penalty_scale(::Type{T}, glr, n) where {T<:Real}
    return getscale(glr.penalty) * ifelse(glr.scale_penalty_with_samples, T(n), T(1.0))
end

function get_penalty_scale(glr, n)
    return get_penalty_scale(eltype(getscale(glr.penalty)), glr, n)
end

function get_penalty_scale_l2(::Type(T), glr, n) where {T<:Real}
    return T(getscale_l2(glr.penalty)) * ifelse(glr.scale_penalty_with_samples, T(n), T(1.0))
end

function get_penalty_scale_l2(glr, n)
    return get_penalty_scale_l2(eltype(getscale(glr.penalty)), glr, n)
end

function get_penalty_scale_l1(::Type{T}, glr, n) where {T<:Real}
    return getscale_l1(glr.penalty) * ifelse(glr.scale_penalty_with_samples, T(n), T(1.0))
end

function get_penalty_scale_l1(glr, n)
    return return get_penalty_scale_l1(eltype(getscale(glr.penalty)), glr, n)
end
