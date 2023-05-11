module MultiobjectiveNonlinearCG

using Parameters: @with_kw

const MIN_PRECISION = Float32;

struct MetaData{P}
    dim_in :: Int
    dim_out :: Int
    precision :: P
    num_iter :: Base.RefValue{Int}
end

# Callback Interface
abstract type AbstractCallback end
abstract type AbstractCallbackCache end

function init_callback_cache(::AbstractCallback, descent_cache, x, fx, d, DfxT, objf!, jacT!, meta)::AbstractCallbackCache
    return nothing
end

function callback_before_iteration!(::AbstractCallbackCache, descent_cache, it_index, x, fx, DfxT, d, objf!, jacT!, meta)::Tuple{String, Int}
    return "", 0
end

function callback_after_iteration!(::AbstractCallbackCache, descent_cache, it_index, x, fx, DfxT, d, objf!, jacT!, meta)::Tuple{String, Int}
    return "", 0
end

# optional
before_priority(::AbstractCallback)::Int = typemax(Int)
after_priority(::AbstractCallback)::Int = typemax(Int)

# derived functions
import Logging
Base.@kwdef struct CallbackHandler
    loglevel :: Logging.LogLevel = Logging.Info
    before_sorting :: Vector{Int}
    after_sorting :: Vector{Int}
end

function print_msg(cb_handler, msg) 
    if !isempty(msg)
        return Logging.@logmsg(cb_handler.loglevel, msg)
    end
end

function init_callbacks(
    callbacks, loglevel, max_iter, descent_cache, x, fx, d, DfxT, objf!, jacT!, meta
)
    max_cb = MaxIterStopping(max_iter)
    cbs = vcat(max_cb, callbacks)
    before_sorting = sortperm(cbs; by=before_priority)
    after_sorting = sortperm(cbs; by=after_priority)
    init_fn(cb) = init_callback_cache(cb, descent_cache, x, fx, d, DfxT, objf!, jacT!, meta)
    cb_caches = map(init_fn, cbs)
    
    return CallbackHandler(; loglevel, before_sorting, after_sorting), cb_caches
end

function handle_callbacks!(cb_caches, cb_handler, cache_sorting, cb_eval_func!, cb_args...)::Int
    for cache in cb_caches[cache_sorting]
        msg, stop_code = cb_eval_func!(cache, cb_args...)
        print_msg(cb_handler, msg)
        if stop_code < 0
            return stop_code
        end        
    end
    return 0
end

function do_callbacks_before_iteration!(cb_caches, cb_handler, cb_args...)
    return handle_callbacks!(cb_caches, cb_handler, cb_handler.before_sorting, callback_before_iteration!, cb_args...)
end
function do_callbacks_after_iteration!(cb_caches, cb_handler, cb_args...)
    return handle_callbacks!(cb_caches, cb_handler, cb_handler.after_sorting, callback_after_iteration!, cb_args...)
end

include("callbacks.jl")

const DEFAULT_CALLBACKS = [
    IterInfoLogger(), 
    CriticalityStop(; eps_crit = eps(MIN_PRECISION(1e-1))),
]

abstract type AbstractDirRule end
abstract type AbstractDirCache end

# mandatory
function init_cache(::AbstractDirRule, x, fx, DfxT, d, objf!, jacT!, meta)::AbstractDirCache
    return nothing
end

# mandatory
function first_step!(descent_cache::AbstractDirCache, d, x, fx, DfxT, objf!, jacT!, meta)::Nothing
    @error "No implementation of `first_step!` for `$(descent_cache)`."
end

# optional, but likely needed
function step!(descent_cache::AbstractDirCache, d, x, fx, DfxT, objf!, jacT!, meta)::Nothing
    return first_step!(descent_cache, d, x, fx, DfxT, objf!, jacT!, meta)
end

# optional
function criticality(descent_cache::AbstractDirCache)::Number
    return Inf
end

function info_msg(descent_cache::AbstractDirCache)::String
    return ""
end

include("dir_rules/all_rules.jl")

function initialize_for_optimization(
    x0::AbstractVector{X}, fx0::AbstractVector{Y}, DfxT0::AbstractMatrix{D}, max_iter
) where{X, Y, D}
    T = Base.promote_type(MIN_PRECISION, X, Y, D)
    x = T.(x0)
    fx = T.(fx0)
    DfxT = T.(DfxT0)

    # read meta data
    dim_in = length(x)
    dim_out = length(fx)
    meta = MetaData(dim_in, dim_out, T, Ref(0))

    return x, fx, DfxT, meta
end

function __optimize(
    x0::AbstractVector{X}, fx0::AbstractVector{Y}, 
    DfxT0::AbstractMatrix{<:Real}, objf!, jacT!;
    ensure_fx0=true,
    ensure_DfxT0=true,
    max_iter=1_000,
    descent_rule=SteepestDescentRule(FixedStepsizeRule()),
    callbacks = DEFAULT_CALLBACKS,
    loglevel = Logging.Info
) where {X<:Number, Y<:Number}

    if isnothing(loglevel)
        loglevel = Logging.LogLevel(typemin(Int32))
    end

    x, fx, DfxT, meta = initialize_for_optimization(x0, fx0, DfxT0, max_iter)
    stop_code = typemin(Int)
    it_index = 0
    if max_iter > 0
        it_index += 1
        # if needed, perform first evaluation(s) -- preparing for first iteration
        # !!! note
        #     We could choose to delay evaluation until after the callbacks have been
        #     evaluated. By doing it before, on the other hand, we can 
        #     a) provide `init_cache` with a valid Jacobian, should it be needed,
        #     b) provide `init_callbacks` with a Jacobian 
        #        (however, this seems not too important),
        #     c) enable callbacks to abort based on the derivative values 
        #        -- even in iteration 1
        ensure_fx0 && objf!(fx, x)
        ensure_DfxT0 && jacT!(DfxT, x)
        
        # allocate direction vector `d` and the `descent_cache` before entering the main loop
        d = similar(x)
        descent_cache = init_cache(descent_rule, x, fx, DfxT, d, objf!, jacT!, meta)
        
        cb_handler, cb_caches = init_callbacks(
            callbacks, loglevel, max_iter, descent_cache, x, fx, d, DfxT, objf!, jacT!, meta
        )

        stop_code = do_callbacks_before_iteration!(
            cb_caches, cb_handler, descent_cache, 1, x, fx, DfxT, d, objf!, jacT!, meta)
        
        if stop_code >= 0
            #jacT!(DfxT, x)
            first_step!(descent_cache, d, x, fx, DfxT, objf!, jacT!, meta)
            x .+= d
            objf!(fx, x)
    
            stop_code = do_callbacks_after_iteration!(
                cb_caches, cb_handler, descent_cache, 1, x, fx, DfxT, d, objf!, jacT!, meta)
        else
            it_index -= 1
        end
    end
    while stop_code >= 0 
        it_index += 1
        stop_code = do_callbacks_before_iteration!(
            cb_caches, cb_handler, descent_cache, it_index, x, fx, DfxT, d, objf!, jacT!, meta)
        if stop_code < 0
            it_index -= 1
            break
        end
        
        jacT!(DfxT, x)
        step!(descent_cache, d, x, fx, DfxT, objf!, jacT!, meta)
        x .+= d
        objf!(fx, x)

        stop_code = do_callbacks_after_iteration!(
            cb_caches, cb_handler, descent_cache, it_index, x, fx, DfxT, d, objf!, jacT!, meta)
        stop_code < 0 && break
    end
    
    meta.num_iter[] = it_index
    return x, fx, stop_code, meta
end

function _optimize(
    x0::AbstractVector{X}, objf, jacT, 
    objf_iip::Val{true}, jacT_iip::Union{Val{false}, Val{true}};
    kwargs...
) where {X<:Real}
    error("""
        The objective function is specified to be mutating, i.e., it has signature
        `objf!(fx, x)`. We need a pre-allocated container for the objective values `fx`."""
    )
end

function _optimize(
    x0::AbstractVector{X}, objf, jacT, 
    objf_iip::Val{false}, jacT_iip::Union{Val{false}, Val{true}};
    kwargs...
) where {X<:Real}
    fx0 = objf(x0)
    return _optimize(x0, fx0, objf, jacT, objf_iip, jacT_iip; ensure_fx0=false, kwargs...)
end

function _optimize(
    x0::AbstractVector{X}, fx0::AbstractVector{Y}, objf, jacT, 
    objf_iip::Union{Val{false}, Val{true}}, jacT_iip::Union{Val{false}, Val{true}};
    kwargs...
) where {X<:Real, Y<:Real}
    dim_in = length(x0)
    dim_out = length(fx0)

    T = Base.promote_type(MIN_PRECISION, X, Y)
    DfxT0 = zeros(T, dim_in, dim_out)

    return _optimize(x0, fx0, DfxT0, objf, jacT, objf_iip, jacT_iip; ensure_DfxT0=true, kwargs...)
end

function _optimize(
    x0::AbstractVector{X}, fx0::AbstractVector{Y}, DfxT0::AbstractMatrix{D}, objf, jacT, 
    objf_iip::Val{false}, jacT_iip::Union{Val{true},Val{false}};
    kwargs...
) where {X<:Real, Y<:Real, D<:Real}
    function objf!(y, x)
        y .= objf(x)
        return nothing
    end    
    return _optimize(x0, fx0, DfxT0, objf!, jacT, Val(true), jacT_iip; kwargs...)
end

function _optimize(
    x0::AbstractVector{X}, fx0::AbstractVector{Y}, DfxT0::AbstractMatrix{D}, objf!, jacT, 
    objf_iip::Val{true}, jacT_iip::Val{false};
    kwargs...
) where {X<:Real, Y<:Real, D<:Real}
    function jacT!(DfxT, x)
        DfxT .= jacT(x)
        return nothing
    end
    return _optimize(x0, fx0, DfxT0, objf!, jacT!, objf_iip, Val(true); kwargs...)
end

function _optimize(
    x0::AbstractVector{X}, fx0::AbstractVector{Y}, DfxT0::AbstractMatrix{D}, objf!, jacT!, 
    objf_iip::Val{true}, jacT_iip::Val{true};
    kwargs...
) where {X<:Real, Y<:Real, D<:Real}
    return __optimize(x0, fx0, DfxT0, objf!, jacT!; kwargs...)
end

function optimize(args...; objf_iip::Bool=true, jacT_iip::Bool=true, kwargs...)
    return _optimize(args..., Val(objf_iip), Val(jacT_iip); kwargs...)
end

end