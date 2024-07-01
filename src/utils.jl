import Accessors: @set
struct GatheringCallback{F, step_metaType}
    x :: Vector{Vector{F}}
    fx :: Vector{Vector{F}}
    critvals :: Vector{F}
    step_meta :: step_metaType
end
function GatheringCallback(mop::AbstractMOP)
    return GatheringCallback(float_type(mop))
end
function GatheringCallback(::Type{F}) where F<:Number
    x = Vector{F}[]
    fx = Vector{F}[]
    critvals = F[]
    step_meta = nothing
    return GatheringCallback(x, fx, critvals, step_meta)
end

function Base.empty!(cback::GatheringCallback)
    @unpack x, fx, critvals, step_meta = cback
    empty!(x)
    empty!(fx)
    empty!(critvals)
    if isa(step_meta, AbstractVector)
        empty!(step_meta)
    end
end
function initialize_callback(callback::GatheringCallback, carrays, mop, step_cache)
    step_metaType = metadata_type(step_cache)
    if !(step_metaType <: Nothing)
        cback = @set callback.step_meta = Vector{step_metaType}()
    else
        cback = callback
    end
    empty!(cback)
    return cback
end

function exec_callback(callback::GatheringCallback, it_index, carrays, mop, step_cache, stop_code)
    stop_code == STOP_MAX_ITER && return nothing
    stop_code == STOP_CRIT_TOL_ABS && return nothing
    push!(callback.x, copy(carrays.x))
    push!(callback.fx, copy(carrays.fx))
    push!(callback.critvals, criticality(carrays, step_cache))
    
    step_metaType = metadata_type(step_cache)
    if !(step_metaType <: Nothing)
        push!(callback.step_meta, metadata(step_cache))
    end
    return nothing
end
_x_mat(callback)=reduce(hcat, callback.x)
_fx_mat(callback)=reduce(hcat, callback.fx)