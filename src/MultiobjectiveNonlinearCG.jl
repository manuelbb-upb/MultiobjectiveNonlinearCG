module MultiobjectiveNonlinearCG

using Parameters: @with_kw

const MIN_PRECISION = Float32;

struct MetaData{P}
    dim_in :: Int
    dim_out :: Int
    precision :: P
end

abstract type AbstractStoppingCriterion end
abstract type AbstractStoppingCache end

struct DummyStoppingCache <: AbstractStoppingCache end

function init_stopping_cache(::AbstractStoppingCriterion, descent_cache, x, fx, DfxT, d, objf!, jacT!, meta)::AbstractStoppingCache
    return nothing
end

function stop_before_iteration!(::AbstractStoppingCache, descent_cache, it_index, x, fx, DfxT, d, objf!, jacT!, meta)::Bool
    return false
end

function stop_after_iteration!(::AbstractStoppingCache, descent_cache, it_index, x, fx, DfxT, d, objf!, jacT!, meta)::Bool
    return false
end

using Printf

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
} <: AbstractStoppingCriterion
    eps :: E
    w :: W = 1
    p :: P = Inf
    print_message :: Bool = true
end

struct RelativeStoppingXCache{E, W, P} <: AbstractStoppingCache
    eps :: E
    w :: W
    p :: P
    print_message :: Bool
end

function init_stopping_cache(sc::RelativeStoppingX, descent_cache, x, fx, DfxT, d, objf!, jacT!, meta)::AbstractStoppingCache
    return RelativeStoppingXCache(sc.eps, sc.w, sc.p, sc.print_message)
end

function stop_after_iteration!(sc::RelativeStoppingXCache, descent_cache, it_index, x, fx, DfxT, d, objf!, jacT!, meta)::Bool
    lhs = LA.norm( sc.w .* d )
    rhs = sc.eps * LA.norm( sc.w .* x )
    do_stop = lhs <= rhs

    if do_stop && sc.print_message
        @printf(
            """
                Stopping after iteration %d because a relative stopping criterion was imposed on `x`. And we have
                %.4e = ‖ w ∘ Δx ‖ₚ ≤ ε ‖ w ∘ xₖ ‖ₚ = %.4e
            """, it_index, lhs, rhs
        )
    end

    return do_stop
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
} <: AbstractStoppingCriterion
    eps :: E
    w :: W = 1
    p :: P = Inf
    print_message :: Bool = true
end

struct RelativeStoppingFxCache{E, W, P, F} <: AbstractStoppingCache
    eps :: E
    w :: W
    p :: P
    fx_old :: F
    print_message :: Bool
end

function init_stopping_cache(sc::RelativeStoppingFx, descent_cache, x, fx, DfxT, d, objf!, jacT!, meta)::AbstractStoppingCache
    return RelativeStoppingFxCache(sc.eps, sc.w, sc.p, copy(fx), sc.print_message)
end

function stop_before_iteration!(sc::RelativeStoppingFxCache, descent_cache, it_index, x, fx, DfxT, d, objf!, jacT!, meta)::Bool
    sc.fx_old .= fx
    return false
end

function stop_after_iteration!(sc::RelativeStoppingFxCache, descent_cache, it_index, x, fx, DfxT, d, objf!, jacT!, meta)::Bool
    Δf = fx .- sc.fx_old
    lhs = LA.norm( sc.w .* Δf )
    rhs = sc.eps * LA.norm( sc.w .* x )
    do_stop = lhs <= rhs

    if do_stop && sc.print_message
        @printf(
            """
                Stopping after iteration %d because a relative stopping criterion was imposed on `f(x)`. And we have
                %.4e = ‖ w ∘ Δf ‖ₚ ≤ ε ‖ w ∘ f(xₖ) ‖ₚ = %.4e
            """, it_index, lhs, rhs
        )
    end

    return do_stop
end

Base.@kwdef struct CriticalityStop{E} <: AbstractStoppingCriterion
    eps_crit :: E
    print_message :: Bool = true
end

struct CriticalityStopCache{E} <: AbstractStoppingCache
    eps_crit :: E
    print_message :: Bool
end

function init_stopping_cache(sc::CriticalityStop, descent_cache, x, fx, DfxT, d, objf!, jacT!, meta)::AbstractStoppingCache
    return CriticalityStopCache(sc.eps_crit, sc.print_message)
end

function stop_after_iteration!(sc::CriticalityStopCache, descent_cache, it_index, x, fx, DfxT, d, objf!, jacT!, meta)
    ω = criticality(descent_cache)
    do_stop = ω <= sc.eps_crit

    if do_stop && sc.print_message
        @printf("""
            Stopping after iteration %d because an absolute stopping criterion was placed on the criticality.
            We have %.4e = ω <= ε_crit = %.4ef.
        """, it_index, ω, sc.eps_crit)
    end

    return do_stop
end

function stop_because_of_criteria!(stopping_caches, crit_eval_func, args...)
    for cache in stopping_caches
        if crit_eval_func(cache, args...)
            return true
        end
    end
    return false
end

function stop_before_iteration!(stopping_caches::AbstractVector, args...) :: Bool
    return stop_because_of_criteria!(stopping_caches, stop_before_iteration!, args...)
end
function stop_after_iteration!(stopping_caches::AbstractVector, args...) :: Bool
    return stop_because_of_criteria!(stopping_caches, stop_after_iteration!, args...)
end

const DEFAULT_STOPPING_CRITERIA = [CriticalityStop(; eps_crit = eps(MIN_PRECISION(1e-3))),]

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

function typed_init_stopping_caches(stopping_criteria, descent_cache, x, fx, DfxT, d, objf!, jacT!, meta)
    isempty(stopping_criteria) && return Vector{DummyStoppingCache}()
    stopping_caches = [ 
        init_stopping_cache(crit, descent_cache, x, fx, DfxT, d, objf!, jacT!, meta)
            for crit in stopping_criteria
    ]
    return stopping_caches
    # T = Union{ (typeof(cache) for cache in stopping_caches)... }
    # return Vector{T}(stopping_caches)
end

include("dir_rules/all_rules.jl")

function optimize(
    x0 :: AbstractVector{X}, fx0::AbstractVector{Y}, objf!, jacT!;
    max_iter=100,
    descent_rule=SteepestDescentRule(FixedStepsizeRule()),
    stopping_criteria = DEFAULT_STOPPING_CRITERIA
) where {X<:Number, Y<:Number}

    # initialize/prealloc iterates:
    T = Base.promote_type(MIN_PRECISION, X, Y)
    x = T.(x0)
    fx = T.(fx0)

    max_iter <= 0 && @goto returnResults

    # read meta data
    dim_in = length(x)
    dim_out = length(fx)
    precision = T
    meta = MetaData(dim_in, dim_out, precision)

    # prealloc arrays for transposed jacobian and step `d`
    DfxT = zeros(precision, dim_in, dim_out)
    d = similar(x)

    # perform first iteration, prealloc `stopping_caches` and `descent_cache`.
    # we want to do this before the main loop    
    ## set jacobian and create descent cache
    
    descent_cache = init_cache(descent_rule, x, fx, DfxT, d, objf!, jacT!, meta)
    stopping_caches = typed_init_stopping_caches(stopping_criteria, descent_cache, x, fx, DfxT, d, objf!, jacT!, meta)
    
    if stop_before_iteration!(
        stopping_caches, descent_cache, 1, x, fx, DfxT, d, objf!, jacT!, meta
    )
        @goto returnResults
    end
    
    jacT!(DfxT, x)
    first_step!(descent_cache, d, x, fx, DfxT, objf!, jacT!, meta)
    x .+= d
    objf!(fx, x)
    
    if stop_after_iteration!(
        stopping_caches, descent_cache, 1, x, fx, DfxT, d, objf!, jacT!, meta
    )
        @goto returnResults
    end
    
    for it_ind=2:max_iter
        if stop_before_iteration!(stopping_caches, descent_cache, it_ind, x, fx, DfxT, d, objf!, jacT!, meta)
            break
        end
        jacT!(DfxT, x)
        step!(descent_cache, d, x, fx, DfxT, objf!, jacT!, meta)
        x .+= d
        objf!(fx, x)

        if stop_after_iteration!(
            stopping_caches, descent_cache, it_ind, x, fx, DfxT, d, objf!, jacT!, meta
        )
            break
        end  
    end

    @label returnResults
    return x, fx
end

end