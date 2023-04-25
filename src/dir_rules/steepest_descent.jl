
Base.@kwdef struct SteepestDescentRule{SR<:AbstractStepsizeRule, E<:Number} <: AbstractDirRule
    stepsize_strategy :: SR

    eps_crit :: E = 100 * nextfloat(zero(MIN_PRECISION))
end

struct SteepestDescentCache{C, S, E} <: AbstractDirCache 
    criticality :: C

    stepsize_cache :: S

    eps_crit :: E
end

function init_cache(descent_rule::SteepestDescentRule, x, fx, DfxT, d, objf!, jacT!, meta)
    T = meta.precision
    stepsize_cache = init_stepsize_cache(descent_rule.stepsize_strategy, x, fx, DfxT, d, objf!, jacT!, meta)
    return SteepestDescentCache(
        Ref(zero(T)), 
        stepsize_cache,
        T(descent_rule.eps_crit)
    )
end

function first_step!(descent_cache::SteepestDescentCache, d, x, fx, DfxT, objf!, jacT!, meta)
    return first_step_steepest_descent(descent_cache, d, x, fx, DfxT, objf!, jacT!, meta)
end

function first_step_steepest_descent!(descent_cache, d, x, fx, DfxT, objf!, jacT!, meta)
    d .= -frank_wolfe_multidir_dual(eachcol(DfxT))

    ω = LA.norm(d, 2)
    descent_cache.criticality[] = ω

    apply_stepsize!(d, descent_cache.stepsize_cache, descent_cache, x, fx, DfxT, objf!, jacT!, meta)

    return nothing
end

function stop_after(descent_cache::SteepestDescentCache, it_index, d, x, fx, DfxT, objf!, jacT!, meta)
    return descent_cache.criticality[] <= descent_cache.eps_crit
end