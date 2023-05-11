
using Printf

const STOP_CODE_DICT = Dict{Int, Symbol}(
    -1 => :RelativeStoppingX,
    -2 => :RelativeStoppingFx,
    -3 => :CriticalityStop,
    -4 => :MaxIterStopping
)

struct IterInfoLogger <: AbstractCallback end
struct IterInfoLoggerCache <: AbstractCallbackCache end

# make sure, this callback get's executed first in each
# iteration -- to have the messages, even if other criteria
# want to stop
before_priority(::IterInfoLogger) = -100
after_priority(::IterInfoLogger) = -100

function init_callback_cache(
    ::IterInfoLogger, descent_cache, x, fx, d, DfxT, objf!, jacT!, meta
)::IterInfoLoggerCache
    return IterInfoLoggerCache()
end

function pretty_vec(vec; max_entries=nothing)
    num_entries = max_entries isa Integer ? max_entries : length(vec)
    num_entries = min(num_entries, length(vec))
    return "[" * join([@sprintf("%3.3e", e) for e in vec[1:num_entries]], ", ") * "]"
end

function callback_before_iteration!(::IterInfoLoggerCache, descent_cache, it_index, x, fx, DfxT, d, objf!, jacT!, meta)::Tuple{String, Int}
    msg = @sprintf("""
    
    Starting iteration %d.
    * Current iterate is %s
    * Current values are %s""",
    it_index, pretty_vec(x), pretty_vec(fx))
    
    return msg, 0
end

function callback_after_iteration!(::IterInfoLoggerCache, descent_cache, it_index, x, fx, DfxT, d, objf!, jacT!, meta)::Tuple{String, Int}
    dmsg = info_msg(descent_cache)
    msg = isempty(dmsg) ? "" : @sprintf("""
    
    Done with iteration %d.
    %s""", it_index, info_msg(descent_cache))
    
    return msg, 0
end

"""
    RelativeStoppingX(eps, w, p)

Stop, if at the end of an iteration it holds that
```math
‖ w ∘ (xₖ₊₁ - xₖ) ‖ₚ ≤ ε ‖ w ∘ xₖ₊₁ ‖ₚ.
```
It is a relative stopping criterion in decision space, based on a weighted p-norm.
"""
Base.@kwdef struct RelativeStoppingX{
    E<:Real, W<:Union{Real, AbstractVector{<:Real}}, P<:Real
} <: AbstractCallback
    eps :: E
    w :: W = 1
    p :: P = Inf
    print_message :: Bool = true
end

struct RelativeStoppingXCache{E, W, P} <: AbstractCallbackCache
    eps :: E
    w :: W
    p :: P
    print_message :: Bool
end

function init_callback_cache(
    sc::RelativeStoppingX{E, W, P}, descent_cache, x, fx, d, DfxT, objf!, jacT!, meta
)::RelativeStoppingXCache{E, W, P} where{E, W, P}
    return RelativeStoppingXCache(sc.eps, sc.w, sc.p, sc.print_message)
end

function callback_after_iteration!(sc::RelativeStoppingXCache, descent_cache, it_index, x, fx, DfxT, d, objf!, jacT!, meta)::Tuple{String, Int}
    lhs = LA.norm( sc.w .* d )
    rhs = sc.eps * LA.norm( sc.w .* x )
    do_stop = lhs <= rhs

    if do_stop
        msg = !sc.print_message ? "" : @sprintf(
            """
                Stopping after iteration %d because a relative stopping criterion was imposed on `x`. And we have
                %.4e = ‖ w ∘ Δx ‖ₚ ≤ ε ‖ w ∘ xₖ ‖ₚ = %.4e
            """, it_index, lhs, rhs
        )
        return msg, -1
    else
        return "", 0
    end
end

"""
    RelativeStoppingFx(eps, w, p)

Stop, if at the end of an iteration it holds that
```math
‖ w ∘ (f(xₖ₊₁) - f(xₖ)) ‖ₚ ≤ ε ‖ w ∘ f(xₖ₊₁) ‖ₚ.
```
It is a relative stopping criterion in objective space, based on a weighted p-norm.
"""
Base.@kwdef struct RelativeStoppingFx{
    E<:Real, W<:Union{Real, AbstractVector{<:Real}}, P<:Real
} <: AbstractCallback
    eps :: E
    w :: W = 1
    p :: P = Inf
    print_message :: Bool = true
end

struct RelativeStoppingFxCache{E, W, P, F} <: AbstractCallbackCache
    eps :: E
    w :: W
    p :: P
    fx_old :: F
    print_message :: Bool
end

function init_callback_cache(
    sc::RelativeStoppingFx{E, W, P}, descent_cache, x, fx, d, DfxT, objf!, jacT!, meta
)::RelativeStoppingFxCache{E, W, P} where{E, W, P}
    return RelativeStoppingFxCache(sc.eps, sc.w, sc.p, copy(fx), sc.print_message)
end

function callback_before_iteration!(sc::RelativeStoppingFxCache, descent_cache, it_index, x, fx, DfxT, d, objf!, jacT!, meta)::Tuple{String, Int}
    sc.fx_old .= fx
    return "", 0
end

function callback_after_iteration!(sc::RelativeStoppingFxCache, descent_cache, it_index, x, fx, DfxT, d, objf!, jacT!, meta)::Tuple{String, Int}
    Δf = fx .- sc.fx_old
    lhs = LA.norm( sc.w .* Δf )
    rhs = sc.eps * LA.norm( sc.w .* x )
    do_stop = lhs <= rhs

    if do_stop
        msg = !sc.print_message ? "" : @sprintf(
            """
                Stopping after iteration %d because a relative stopping criterion was imposed on `f(x)`. And we have
                %.4e = ‖ w ∘ Δf ‖ₚ ≤ ε ‖ w ∘ f(xₖ) ‖ₚ = %.4e
            """, it_index, lhs, rhs
        )
        return msg, -2
    else
        return "", 0
    end
end

Base.@kwdef struct CriticalityStop{E} <: AbstractCallback
    eps_crit :: E
    print_message :: Bool = true
end

struct CriticalityStopCache{E} <: AbstractCallbackCache
    eps_crit :: E
    print_message :: Bool
end

function init_callback_cache(
    sc::CriticalityStop{E}, descent_cache, x, fx, d, DfxT, objf!, jacT!, meta
)::CriticalityStopCache{E} where E
    return CriticalityStopCache(sc.eps_crit, sc.print_message)
end

function callback_after_iteration!(sc::CriticalityStopCache, descent_cache, it_index, x, fx, DfxT, d, objf!, jacT!, meta)::Tuple{String, Int}
    ω = criticality(descent_cache)
    do_stop = ω <= sc.eps_crit

    if do_stop 
        msg = !sc.print_message ? "" : @sprintf("""
            Stopping after iteration %d because an absolute stopping criterion was placed on the criticality.
            We have %.4e = ω <= ε_crit = %.4ef.
        """, it_index, ω, sc.eps_crit)
        return msg, -3
    else
        return "", 0
    end
end

struct MaxIterStopping <: AbstractCallback
    max_iter :: Int
end

struct MaxIterStoppingCache <: AbstractCallbackCache
    max_iter :: Int
end

function init_callback_cache(
    m::MaxIterStopping, descent_cache, x, fx, d, DfxT, objf!, jacT!, meta
)::MaxIterStoppingCache
    return MaxIterStoppingCache(m.max_iter)
end

before_priority(::MaxIterStopping) = -101
after_priority(::MaxIterStopping) = -100

function callback_before_iteration!(m::MaxIterStoppingCache, descent_cache, it_index, x, fx, DfxT, d, objf!, jacT!, meta)::Tuple{String, Int}
    do_stop = it_index > m.max_iter

    if do_stop 
        return "Stopping, because the maximum number of iterations is $(m.max_iter).", -4
    else
        return "", 0
    end
end


struct GatheringCallbackCache{T} <: AbstractCallbackCache
    x_arr :: Vector{Vector{T}}
    fx_arr :: Vector{Vector{T}}
    DfxT_arr :: Vector{Matrix{T}}
    d_arr :: Vector{Vector{T}}
end

function GatheringCallbackCache(T::Type)
    x_arr = Vector{T}[]
    fx_arr = copy(x_arr)
    DfxT_arr = Matrix{T}[]
    d_arr = copy(x_arr)
    return GatheringCallbackCache(x_arr, fx_arr, DfxT_arr, d_arr)
end

struct GatheringCallback{CT} <: AbstractCallback 
    cache::CT
end

function GatheringCallback(T::Type)
    return GatheringCallback(GatheringCallbackCache(T))
end

function init_callback_cache(
    m::GatheringCallback, descent_cache, x, fx, d, DfxT, objf!, jacT!, meta
)
    return m.cache
end

function callback_before_iteration!(m::GatheringCallbackCache, descent_cache, it_index, x, fx, DfxT, d, objf!, jacT!, meta)::Tuple{String, Int}
    if it_index == 1
        push!(m.x_arr, copy(x))
        push!(m.fx_arr, copy(fx))
        push!(m.DfxT_arr, copy(DfxT))
        push!(m.d_arr, copy(d))
    end
    return "", 0    
end

function callback_after_iteration!(m::GatheringCallbackCache, descent_cache, it_index, x, fx, DfxT, d, objf!, jacT!, meta)::Tuple{String, Int}
    push!(m.x_arr, copy(x))
    push!(m.fx_arr, copy(fx))
    push!(m.DfxT_arr, copy(DfxT))
    push!(m.d_arr, copy(d))
    return "", 0    
end