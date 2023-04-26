
Base.@kwdef struct SteepestDescentRule{SR<:AbstractStepsizeRule} <: AbstractDirRule
    stepsize_rule :: SR
end

struct SteepestDescentCache{C, S} <: AbstractDirCache 
    sd_norm_squared :: C
    stepsize_cache :: S
end

function init_cache(descent_rule::SteepestDescentRule, x, fx, DfxT, d, objf!, jacT!, meta)
    T = meta.precision
    stepsize_cache = init_stepsize_cache(descent_rule.stepsize_rule, x, fx, DfxT, d, objf!, jacT!, meta)
    return SteepestDescentCache(
        Ref(zero(T)), 
        stepsize_cache,
    )
end

function set_steepest_descent!(descent_cache, d, x, fx, DfxT, objf!, jacT!, meta)
    d .= -frank_wolfe_multidir_dual(eachcol(DfxT))

    ω = LA.norm(d, 2)
    descent_cache.sd_norm_squared[] = ω

    return nothing
end

function first_step!(descent_cache::SteepestDescentCache, d, x, fx, DfxT, objf!, jacT!, meta)
    set_steepest_descent!(descent_cache, d, x, fx, DfxT, objf!, jacT!, meta)
    apply_stepsize!(d, descent_cache.stepsize_cache, descent_cache, x, fx, DfxT, objf!, jacT!, meta)
    return nothing
end

function criticality(descent_cache::SteepestDescentCache) 
    return descent_cache.sd_norm_squared
end