struct GatheringCallback{F}
    x :: Vector{Vector{F}}
    fx :: Vector{Vector{F}}
    critvals :: Vector{F}
end
function GatheringCallback(mop::AbstractMOP)
    return GatheringCallback(float_type(mop))
end
function GatheringCallback(::Type{F}) where F<:Number
    x = Vector{F}[]
    fx = Vector{F}[]
    critvals = F[]
    return GatheringCallback(x, fx, critvals)
end
function (callback::GatheringCallback)(it_index, carrays, mop, step_cache)
    push!(callback.x, copy(carrays.x))
    push!(callback.fx, copy(carrays.fx))
    push!(callback.critvals, criticality(carrays, step_cache))
end
_x_mat(callback)=reduce(hcat, callback.x)
_fx_mat(callback)=reduce(hcat, callback.fx)