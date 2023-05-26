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
function init_cache(::AbstractDirRule, x, fx, DfxT, d, objf!, jacT!, objf_and_jacT!, meta)::AbstractDirCache
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
    x0::AbstractVector{X}, fx0::AbstractVector{Y}, DfxT0::AbstractMatrix{D}
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

function _optimize(
    x0::AbstractVector{X}, fx0::AbstractVector{Y}, 
    DfxT0::AbstractMatrix{<:Real}, objf!, jacT!,
    objf_and_jacT!;
    ensure_fx0=true,
    ensure_DfxT0=true,
    max_iter=1_000,
    descent_rule=SteepestDescentRule(StandardArmijoRule()),
    callbacks = DEFAULT_CALLBACKS,
    loglevel = Logging.Info
) where {X<:Number, Y<:Number}

    if isnothing(loglevel)
        loglevel = Logging.LogLevel(typemin(Int32))
    end

    if isnothing(objf_and_jacT!)
        objf_and_jacT! = function(y, D, x)
            objf!(y, x)
            jacT!(D, x)
            return nothing
        end
    end

    x, fx, DfxT, meta = initialize_for_optimization(x0, fx0, DfxT0)
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
        if ensure_fx0
            if ensure_DfxT0
                objf_and_jacT!(fx, DfxT, x)
                ensure_DfxT0 = false
            else
                objf!(fx, x)
            end
        end
        ensure_DfxT0 && jacT!(DfxT, x)
        
        # allocate direction vector `d` and the `descent_cache` before entering the main loop
        d = similar(x)
        descent_cache = init_cache(descent_rule, x, fx, DfxT, d, objf!, jacT!, objf_and_jacT!, meta)
        
        cb_handler, cb_caches = init_callbacks(
            callbacks, loglevel, max_iter, descent_cache, x, fx, d, DfxT, objf!, jacT!, meta
        )

        stop_code = do_callbacks_before_iteration!(
            cb_caches, cb_handler, descent_cache, 1, x, fx, DfxT, d, objf!, jacT!, meta)
        
        if stop_code >= 0
            first_step!(descent_cache, d, x, fx, DfxT, objf!, jacT!, meta)
            x .+= d
            objf_and_jacT!(fx, DfxT, x)
    
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
        
        step!(descent_cache, d, x, fx, DfxT, objf!, jacT!, meta)
        x .+= d
        objf_and_jacT!(fx, DfxT, x)

        stop_code = do_callbacks_after_iteration!(
            cb_caches, cb_handler, descent_cache, it_index, x, fx, DfxT, d, objf!, jacT!, meta)
        stop_code < 0 && break
    end
    
    meta.num_iter[] = it_index
    return x, fx, stop_code, meta
end

function make_mutating(func)
    return function(y, args...)
        y .= func(args...)
    end
end

function optimize(x0, objf, jac; kwargs...)
    return optimize(x0; objf, jac, kwargs...)
end

function optimize(x0, objf_and_jac; kwargs...)
    return optimize(x0; objf_and_jac, kwargs...)
end

function optimize(x0, objf, jac, objf_and_jac; kwargs...)
    return optimize(x0; objf, jac, objf_and_jac, kwargs...)
end

function optimize(
    x0::AbstractVector{X}; 
    objf = nothing, jac = nothing,
    objf_and_jac = nothing,
    objf_and_jac_is_mutating = false,
    jac_is_transposed=true,
    objf_is_mutating::Bool=false, jac_is_mutating::Bool=false,
    fx0::Union{AbstractVector{<:Real}, Nothing}=nothing,
    Dfx0::Union{AbstractMatrix{<:Real}, Nothing}=nothing,
    ensure_fx0::Bool=false,
    ensure_Dfx0::Bool=false,
    kwargs...
) where {X<:Real}

    J = jac
    if !isnothing(jac)
        if !jac_is_transposed
            if jac_is_mutating
                J = (y, D, x) -> jac(y, D', x)
            else
                J = (x,) -> jac(x)'
            end
        end
    end

    fJ = objf_and_jac
    if !isnothing(objf_and_jac)
        if !jac_is_transposed
            if objf_and_jac_is_mutating
                fJ = (y, D, x) -> jac(y, D', x)
            else
                fJ = (x,) -> jac(x)'
            end
        end
    end

    if !isnothing(Dfx0)
        if !jac_is_transposed
            Dfx0 = copy(Dfx0')
        end
    end
    
    f = objf
    if isnothing(f)
        if !isnothing(fJ)
            @warn ("inferring objective from `objf_and_jac`")
            if objf_and_jac_is_mutating
                if isnothing(Dfx0)
                    if !isnothing(fx0)
                        Dfx0 = similar(fx0, length(x0), length(fx0))
                    elseif !isnothing(J)
                        if !jac_is_mutating
                            Dfx0 = J(x0)
                            ensure_Dfx0 = false
                        end
                    end
                end
                if !isnothing(Dfx0)
                    _D = similar(Dfx0)
                    f = function (y, x)
                        fJ(y, _D, x)
                        return nothing
                    end
                    objf_is_mutating = true
                end
            else
                f = x -> first(fJ(x))
                objf_is_mutating = false
            end
        end
    end
    isnothing(f) && error("No objective provided.")

    if isnothing(J)
        if !isnothing(fJ)
            @warn ("inferring jacobian from `objf_and_jac`")
            if objf_and_jac_is_mutating
                if isnothing(fx0)
                    if !isnothing(Dfx0)
                        fx0 = similar(vec(Dfx0[:, 1]))
                        ensure_fx0 = true
                    elseif !objf_is_mutating
                        fx0 = f(x0)
                        ensure_fx0 = false
                    end
                end
                if !isnothing(fx0)
                    _fx0 = similar(fx0)
                    J = function (D, x)
                        fJ(_fx0, D, x)
                        return nothing
                    end
                end
            else
                J = x -> last(fJ(x))
            end
        end
    end
    isnothing(J) && error("No jacobian provided.")

    f! = if objf_is_mutating
        f
    else
        make_mutating(f)
    end

    J! = if jac_is_mutating
        J
    else
        make_mutating(J)
    end

    fJ! = if isnothing(fJ)
        function (y, D, x)
            f!(y, x)
            J!(D, x)
            return nothing
        end
    elseif objf_and_jac_is_mutating
        fJ
    else
        make_mutating(fJ)
    end

    if isnothing(fx0)
        if !isnothing(Dfx0)
            fx0 = similar(vec(Dfx0[:, 1]))
            ensure_fx0 = true
        elseif !objf_and_jac_is_mutating && !isnothing(fJ)
            fx0, Dfx0 = fJ(x0)
            ensure_fx0 = false
            ensure_Dfx0 = false
        elseif !objf_is_mutating
            fx0 = f(x0)
            ensure_fx0 = false
        elseif !jac_is_mutating
            Dfx0 = J(x0)
            fx0 = similar(vec(Dfx0[:, 1]))
            ensure_fx0 = true
        end
    end
    isnothing(fx0) && error("For a mutating objective function, please provide a pre-allocated objective vector with kwarg `fx0`.")

    if isnothing(Dfx0)
        T = Base.promote_type(MIN_PRECISION, X, eltype(fx0))
        Dfx0 = zeros(T, length(x0), length(fx0))
        ensure_Dfx0 = true
    else
        @assert size(Dfx0) == (length(x0), length(fx0)) "Transposed jacobian array `DfxT0` must have size ($(length(x0)), $(length(fx0)))."
    end

    ensure_DfxT0 = ensure_Dfx0
    return _optimize(x0, fx0, Dfx0, f!, J!, fJ!; ensure_fx0, ensure_DfxT0, kwargs...)
end

#=
function improve(
    x0::AbstractVector{X}, fx0::AbstractVector{Y}, 
    objf!, jacT!; DfxT0=nothing, kwargs...
) where{X<:Real, Y<:Real}
    if isnothing(DfxT0)
        T = Base.promote_type(MIN_PRECISION, X, Y)
        DfxT_0 = zeros(T, length(x0), length(fx_0))
    else
        @assert size(DfxT0) == (length(x0), length(fx_0)) "Transposed jacobian array `DfxT0` must have size ($(length(x0)), $(length(fx_0)))."
        DfxT_0 = DfxT0
    end

    return _optimize(x0, fx0, DfxT_0, objf!, jacT!)
end
=#

end